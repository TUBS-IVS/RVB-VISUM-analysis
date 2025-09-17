from __future__ import annotations

import asyncio
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
import contextlib
import os
import logging
from tqdm import tqdm
import subprocess

import pandas as pd

from gtfs_routing.transit_router import TransitRouter
from gtfs_routing.gtfs_setup import AppConfig, run_grashopper
from results_processing import (
    results_to_dataframe_with_options,
    parse_graphhopper_response,
    select_best_path,
)

import routing_helpers as rh

############################################################
# NEW WORKFLOW: schedule-driven routing                     #
# We read a pre-built schedule parquet with columns:        #
#  - od_index, pair_id, origin_endpoint_index,              #
#    destination_endpoint_index, departure_time             #
#  - origin_lon_wgs84, origin_lat_wgs84, dest_lon_wgs84,    #
#    dest_lat_wgs84 (required)                              #
# and possibly original meta columns (usage_rebus, etc.)    #
############################################################

@dataclass
class RoutingMain:
    gh_port: int = 8989
    date: str = "2025-05-13"               # base date (UTC)
    start_hour: int = 6
    end_hour: int = 22
    interval_minutes: int = 15
    test_limit: Optional[int] = None        # limit number of unique od_index pairs
    max_retries: int = 3
    save_raw: bool = False                  # optionally pickle raw slice results
    resume_slices: bool = True              # reuse cached per-slice fastest/allpaths parquet (if future extension)
    schedule_file: Optional[str] = None     # explicit path to schedule parquet
    auto_start: bool = True                # start backend if not running (also used for automatic restart)
    scenario_name: str = "scenario_V10_2025"  # output/<scenario_name>
    # new toggles for caching and outputs
    cache_slices: bool = True                   # write per-slice parquet caches
    output_parquet: bool = True                 # write final parquet outputs
    output_csv: bool = False                    # write final csv outputs
    phase: str = "both"                         # 'route', 'enrich', or 'both'
    slice_batch_size: int = 250                 # routing chunk size per slice for progress updates
    quiet_backend: bool = True                  # suppress GraphHopper stdout/stderr

    def __post_init__(self):
        self.repo_root = rh.repo_root()
        self.output_dir = self.repo_root / "output" / self.scenario_name
        self.input_dir = self.repo_root / "input"
        self.graphhopper_dir = self.input_dir / "graphhopper"
        self.gtfs_dir = self.input_dir / "gtfs-data" / "2025(V10)"
        self.gtfs_zip = self.gtfs_dir / "VM_RVB_Basisszenario2025_mDTicket_V10_init_LB GTFS_250827.zip"
        self.routes_txt = self.gtfs_dir / "routes.txt"
        # per-slice cache directory
        self.slice_cache_dir = self.output_dir / "slice_cache"
        self.slice_cache_dir.mkdir(parents=True, exist_ok=True)
        # raw cache for routing-only results (JSON.gz per slice)
        self.raw_cache_dir = self.output_dir / "raw_cache"
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- cache path helpers ----------
    def _raw_cache_path(self, departure_time: str) -> Path:
        date_compact = self.date.replace('-', '')
        t_safe = rh.sanitize_time_for_filename(departure_time)
        base_suffix = f"{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
        test_tag = f"_test{self.test_limit}" if (self.test_limit and self.test_limit > 0) else ""
        base = f"slice_{date_compact}_{base_suffix}{test_tag}_{t_safe}"
        return self.raw_cache_dir / f"{base}_raw.json.gz"

    def _wait_for_backend(self, timeout: float = 180.0, poll: float = 2.0) -> bool:
        """Poll until backend responds or timeout reached."""
        start = time.time()
        while time.time() - start < timeout:
            if rh.is_backend_ready(self.gh_port, timeout=1.0):
                return True
            time.sleep(poll)
        return False

    def _start_backend(self) -> None:
        java_bin = rh.resolve_java_bin()
        print(f"Using Java binary: {java_bin}")
        cache_dir = self.graphhopper_dir / "graph-cache"
        building = not cache_dir.exists()
        print(f"Starting GraphHopper on port {self.gh_port} ... ({'quiet' if self.quiet_backend else 'verbose'} mode; {'initial build' if building else 'reuse cache'})")
        # Silent logback (no console spam) if quiet mode
        silent_logback = self.graphhopper_dir / "logback-silent.xml"
        if self.quiet_backend and not silent_logback.exists():
            silent_logback.write_text(
                """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<configuration>\n  <root level=\"ERROR\"/>\n</configuration>\n""",
                encoding="utf-8",
            )
        java_opts = [
            "-Xms16g","-Xmx64g","-XX:+UseG1GC","-XX:MaxGCPauseMillis=200","-XX:ActiveProcessorCount=12","-Xss256k",
            "-XX:ReservedCodeCacheSize=256m","-XX:MaxDirectMemorySize=512m",
            "-Dlogging.level.root=ERROR","-Dlogging.level.com.graphhopper=ERROR","-Dlogging.level.org.springframework=ERROR",
        ]
        if self.quiet_backend:
            java_opts.append(f"-Dlogback.configurationFile={silent_logback.as_posix()}")
        cfg = AppConfig(
            project_root=self.repo_root,
            graphhopper_dir=self.graphhopper_dir,
            gh_config_path=self.graphhopper_dir / "config.yml",
            gtfs_input_zip=self.gtfs_zip,
            output_base=self.repo_root / "output",
            scenario_name=self.scenario_name,
            gh_jar_path=self.graphhopper_dir / "graphhopper-web-10.2.jar",
            gh_cache_dir=self.graphhopper_dir / "graph-cache",
            gh_port=self.gh_port,
            java_bin=java_bin,
            java_opts=java_opts,
        )
        # Delegate suppression to run_grashopper(quiet=...); still wrap in devnull to hide any stray prints
        if self.quiet_backend:
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                run_grashopper(cfg, quiet=True)
        else:
            run_grashopper(cfg, quiet=False)
        timeout = 3600 if building else 300  # allow longer for initial import
        print(f"Waiting for GraphHopper to become ready (timeout {timeout//60} min)...")
        if self._wait_for_backend(timeout=timeout):
            print("GraphHopper backend is ready.")
        else:
            print(f"Warning: GraphHopper did not open port {self.gh_port} within timeout.")

    # ---------- Schedule handling ----------
    def _expected_schedule_path(self) -> Path:
        date_compact = self.date.replace('-', '')
        return self.output_dir / (
            f"df_routing_OD_schedule_{date_compact}_{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m_wgs84.parquet"
        )

    def _resolve_schedule_path(self) -> Path:
        # 1. If explicit path provided and exists, use it
        if self.schedule_file:
            candidate = Path(self.schedule_file)
            if candidate.exists():
                return candidate.resolve()
            # 2. Treat provided string as filename within scenario output dir
            in_scenario = self.output_dir / candidate.name
            if in_scenario.exists():
                return in_scenario.resolve()
            # 3. Fuzzy search (substring) inside scenario output dir
            fuzzy_matches = [p for p in self.output_dir.glob(f"*{candidate.name}*") if p.is_file()]
            if fuzzy_matches:
                # pick most recent
                fuzzy_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if len(fuzzy_matches) > 1:
                    print("Multiple schedule matches found, using most recent:")
                    for fm in fuzzy_matches:
                        print(f"  - {fm.name}")
                return fuzzy_matches[0].resolve()
            raise FileNotFoundError(f"Could not resolve schedule file '{self.schedule_file}' in scenario dir {self.output_dir}")
        # 4. Default expected path
        expected = self._expected_schedule_path()
        if expected.exists():
            return expected.resolve()
        # 5. Pattern fallback using date/hour window prefix
        prefix = f"df_routing_OD_schedule_{self.date.replace('-', '')}_{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
        candidates = list(self.output_dir.glob(f"{prefix}*.parquet"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print("Expected schedule not found, using closest match candidates:")
            for c in candidates:
                print(f"  - {c.name}")
            return candidates[0].resolve()
        raise FileNotFoundError(f"Schedule parquet not found. Looked for explicit file or pattern '{prefix}*.parquet' in {self.output_dir}")

    def _load_schedule(self) -> pd.DataFrame:
        path = self._resolve_schedule_path()
        print(f"Using schedule file: {path}")
        df = pd.read_parquet(path)
        required = {
            'od_index','departure_time','origin_lon_wgs84','origin_lat_wgs84',
            'dest_lon_wgs84','dest_lat_wgs84','origin_endpoint_index','destination_endpoint_index'
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Schedule file missing required columns: {missing}")
        # Ensure departure_time sorted
        try:
            df['departure_time'] = df['departure_time'].astype(str)
        except Exception:
            pass
        return df

    def _unique_time_slices(self, schedule: pd.DataFrame) -> List[str]:
        return sorted(schedule['departure_time'].unique())

    def _restrict_test_limit(self, schedule: pd.DataFrame) -> pd.DataFrame:
        if self.test_limit is None or self.test_limit <= 0:
            return schedule
        # keep first N unique od_index values
        keep_od = list(dict.fromkeys(schedule['od_index']))[: self.test_limit]
        out = schedule[schedule['od_index'].isin(keep_od)].copy()
        return out

    # ---------- per-slice cache helpers ----------
    def _slice_cache_paths(self, departure_time: str) -> Tuple[Path, Path]:
        date_compact = self.date.replace('-', '')
        t_safe = rh.sanitize_time_for_filename(departure_time)
        base_suffix = f"{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
        test_tag = f"_test{self.test_limit}" if (self.test_limit and self.test_limit > 0) else ""
        base = f"slice_{date_compact}_{base_suffix}{test_tag}_{t_safe}"
        return (
            self.slice_cache_dir / f"{base}_allpaths.parquet",
            self.slice_cache_dir / f"{base}_fastest.parquet",
        )

    def run(self) -> None:
        # Load schedule and derive slices
        full_schedule = self._load_schedule()
        print(f"Loaded schedule rows (full): {len(full_schedule)}")
        time_slices_full = self._unique_time_slices(full_schedule)
        total_unique_od_full = full_schedule['od_index'].nunique()
        print(f"Full unique OD pairs: {total_unique_od_full}; time slices: {len(time_slices_full)}")
        schedule = self._restrict_test_limit(full_schedule)
        restricted_unique_od = schedule['od_index'].nunique()
        if self.test_limit:
            print(f"Applied test_limit -> rows now: {len(schedule)} (unique od_index: {restricted_unique_od})")
        time_slices = self._unique_time_slices(schedule)  # should match full
        if not time_slices:
            print("No time slices to process.")
            return
        print(f"Time slices (used): {len(time_slices)} ({time_slices[0]} .. {time_slices[-1]})")

        # Common name parts
        date_compact = self.date.replace('-', '')
        base_suffix = f"{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
        test_tag = f"_test{self.test_limit}" if (self.test_limit and self.test_limit > 0) else ""

        # Phase 1: ROUTING -> write raw JSON.gz per slice (resume if exists)
        if self.phase in ("route", "both"):
            # Backend availability only needed for routing
            if not rh.is_backend_ready(self.gh_port, timeout=2.0):
                if self.auto_start:
                    print("Backend not running – starting GraphHopper...")
                    self._start_backend()
                    if not rh.is_backend_ready(self.gh_port, timeout=2.0):
                        print("Backend still not reachable after start attempt. Aborting routing phase.")
                        return
                else:
                    print("Backend not running – please start manually or pass --auto-start.")
                    return
            else:
                print(f"GraphHopper running on port {self.gh_port}.")

            router = TransitRouter(port=self.gh_port)
            routing_wall_start = time.time()
            accumulated_routing_seconds = 0.0
            successful_requests = 0
            expected_requests_restricted = restricted_unique_od * len(time_slices)
            expected_requests_full = total_unique_od_full * len(time_slices_full)

            # background writer for raw cache files
            def _write_raw_cache(path: Path, results_obj) -> None:
                with gzip.open(path, 'wt', encoding='utf-8') as f:
                    json.dump(results_obj, f)

            write_executor = ThreadPoolExecutor(max_workers=4)
            write_futures: List[Tuple[object, Path]] = []  # (future, path)

            # overall progress across slices (position 0) + persistent per-slice bar (position 1)
            with tqdm(total=len(time_slices), desc="Slices", unit="slice", dynamic_ncols=True, position=0, leave=True) as pbar_slices:
                pbar_slice = tqdm(total=0, desc="Slice -", unit="pair", dynamic_ncols=True, position=1, leave=True)
                devnull = open(os.devnull, 'w')
                try:
                    for idx, t in enumerate(time_slices, start=1):
                        raw_path = self._raw_cache_path(t)
                        slice_df = schedule[schedule['departure_time'] == t].copy()
                        if slice_df.empty:
                            pbar_slices.update(1)
                            continue
                        if raw_path.exists():
                            successful_requests += len(slice_df)
                            pbar_slices.update(1)
                            continue
                        coords_list = list(zip(
                            slice_df['origin_lat_wgs84'],
                            slice_df['origin_lon_wgs84'],
                            slice_df['dest_lat_wgs84'],
                            slice_df['dest_lon_wgs84'],
                        ))
                        # configure per-slice progress bar
                        pbar_slice.reset()
                        pbar_slice.total = len(coords_list)
                        pbar_slice.set_description(f"Slice {t}")
                        pbar_slice.refresh()
                        # chunk coords to allow progress updates
                        results_all = []
                        for start in range(0, len(coords_list), self.slice_batch_size):
                            sub = coords_list[start:start + self.slice_batch_size]
                            attempt = 0
                            sub_results = None
                            t0 = time.time()
                            while attempt < self.max_retries:
                                attempt += 1
                                try:
                                    # silence any print/log output from deep inside routing
                                    old_disable = logging.root.manager.disable
                                    logging.disable(logging.CRITICAL)
                                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                                        sub_results = asyncio.run(router.batch_pt_routes_safe(sub, departure_time=t))
                                    logging.disable(old_disable)
                                    break
                                except Exception as e:
                                    logging.disable(old_disable if 'old_disable' in locals() else logging.NOTSET)
                                    # Auto-restart logic
                                    if self.auto_start and not rh.is_backend_ready(self.gh_port, timeout=1.0):
                                        print(f"Routing batch failure (attempt {attempt}) - backend down? Restarting GraphHopper...")
                                        self._start_backend()
                                        if self._wait_for_backend():
                                            print("Backend restarted and ready.")
                                            # Re-create router instance (socket/session state may reset)
                                            router = TransitRouter(port=self.gh_port)
                                        else:
                                            print("Backend restart timeout; will continue retries.")
                                    if attempt >= self.max_retries:
                                        print(f"Batch failed after {attempt} attempts: {e}")
                                        break
                                    delay = min(30, 2 * attempt)
                                    pbar_slice.set_postfix_str(f"retry {attempt} in {delay}s")
                                    time.sleep(delay)
                            if sub_results is None:
                                pbar_slice.update(len(sub))
                                continue
                            results_all.extend(sub_results)
                            accumulated_routing_seconds += (time.time() - t0)
                            successful_requests += len(sub)
                            pbar_slice.update(len(sub))

                        if not results_all:
                            pbar_slices.update(1)
                            continue
                        # queue raw cache write (non-blocking)
                        fut = write_executor.submit(_write_raw_cache, raw_path, results_all)
                        write_futures.append((fut, raw_path))
                        pbar_slices.update(1)
                finally:
                    try:
                        devnull.close()
                    except Exception:
                        pass
                    pbar_slice.close()

            # Routing runtime estimation (when test subset)
            if False and self.test_limit and restricted_unique_od < total_unique_od_full and successful_requests > 0:
                # estimation prints disabled to keep bars clean
                pass

            # ensure all queued writes are finished before proceeding or exiting
            if write_futures:
                for fut, path in write_futures:
                    try:
                        fut.result()
                    except Exception as e:
                        # keep bars clean; show minimal error hint in postfix
                        pbar_slices.set_postfix_str(f"write error: {path.name}")
                write_executor.shutdown(wait=False)

            if self.phase == "route":
                # keep output clean; bars reflect completion
                return

        # Phase 2: ENRICH -> read raw caches, write per-slice parquet (dataset), no single-file aggregation
        slices_written = 0

        for idx, t in enumerate(time_slices, start=1):
            raw_path = self._raw_cache_path(t)
            slice_df = schedule[schedule['departure_time'] == t].copy()
            if slice_df.empty:
                continue
            if not raw_path.exists():
                print(f"[{idx}/{len(time_slices)}] Missing raw cache for slice {t} -> {raw_path.name}; skipping.")
                continue
            print(f"[{idx}/{len(time_slices)}] Enrich slice {t} from raw cache: {raw_path.name}")
            try:
                with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
                    results = json.load(f)
            except Exception as e:
                print(f"  -> raw cache read error ({raw_path.name}): {e}")
                continue

            # Build all-paths rows
            slice_all_rows_local: List[dict] = []
            for local_idx, raw in enumerate(results):
                if not raw:
                    continue
                paths = raw.get('paths', []) or []
                best = select_best_path(paths)
                best_id = id(best) if best else None
                meta_row = slice_df.iloc[local_idx]
                for path_idx, p in enumerate(paths):
                    parsed = parse_graphhopper_response(p) or {}
                    route_sequence = []
                    has_pt_leg = False
                    for leg in (p.get('legs') or []):
                        if (leg.get('type') or '').lower() == 'pt':
                            has_pt_leg = True
                            rid = leg.get('route_id')
                            if rid is not None:
                                route_sequence.append(str(rid))
                    if not has_pt_leg:
                        route_sequence = ['walk']
                    row = {
                        'departure_time': t,
                        'slice_row_index': local_idx,
                        'od_index': meta_row.get('od_index'),
                        'pair_id': meta_row.get('pair_id'),
                        'origin_endpoint_index': meta_row.get('origin_endpoint_index'),
                        'destination_endpoint_index': meta_row.get('destination_endpoint_index'),
                        'origin_lon_wgs84': meta_row.get('origin_lon_wgs84'),
                        'origin_lat_wgs84': meta_row.get('origin_lat_wgs84'),
                        'dest_lon_wgs84': meta_row.get('dest_lon_wgs84'),
                        'dest_lat_wgs84': meta_row.get('dest_lat_wgs84'),
                        'is_best': id(p) == best_id,
                        'path_index': path_idx,
                        'gh_time': parsed.get('gh_time'),
                        'pure_pt_travel_time': parsed.get('pure_pt_travel_time'),
                        'total_walking_distance': parsed.get('total_walking_distance'),
                        'walking_dist_to_first_pt_station': parsed.get('walking_dist_to_first_pt_station'),
                        'transfers': parsed.get('transfers'),
                        'has_pt': parsed.get('has_pt'),
                        'n_pt_legs': parsed.get('n_pt_legs'),
                        'route_id_0': parsed.get('route_id_0'),
                        'route_id_1': parsed.get('route_id_1'),
                        'route_id_2': parsed.get('route_id_2'),
                        'trip_headsign_0': parsed.get('trip_headsign_0'),
                        'trip_headsign_1': parsed.get('trip_headsign_1'),
                        'trip_headsign_2': parsed.get('trip_headsign_2'),
                        'first_leg_departure_time': parsed.get('first_leg_departure_time'),
                        'final_arrival_time': parsed.get('final_arrival_time'),
                        'first_station_name': parsed.get('first_station_name'),
                        'first_station_coord': parsed.get('first_station_coord'),
                        'last_station_name': parsed.get('last_station_name'),
                        'last_station_coord': parsed.get('last_station_coord'),
                        'transfer_stations': parsed.get('transfer_stations'),
                        'transfer_coords': parsed.get('transfer_coords'),
                        'transfer_wait_times': parsed.get('transfer_wait_times'),
                        'route_sequence': route_sequence,
                    }
                    slice_all_rows_local.append(row)
            # no global accumulation; we only write per-slice parquet

            # Fastest dataframe using helper
            fast_df = results_to_dataframe_with_options(
                results,
                slice_departure_time=t,
                origin_lats= slice_df['origin_lat_wgs84'].values,
                origin_lons= slice_df['origin_lon_wgs84'].values,
                dest_lats= slice_df['dest_lat_wgs84'].values,
                dest_lons= slice_df['dest_lon_wgs84'].values,
            )
            fast_df['slice_row_index'] = fast_df['od_index']
            fast_df['od_index'] = slice_df['od_index'].values
            if 'pair_id' in slice_df.columns:
                fast_df['pair_id'] = slice_df['pair_id'].values
            fast_df['origin_endpoint_index'] = slice_df['origin_endpoint_index'].values
            fast_df['destination_endpoint_index'] = slice_df['destination_endpoint_index'].values
            if 'origin_lon_wgs84' in slice_df.columns:
                fast_df['origin_lon_wgs84'] = slice_df['origin_lon_wgs84'].values
                fast_df['origin_lat_wgs84'] = slice_df['origin_lat_wgs84'].values
                fast_df['dest_lon_wgs84'] = slice_df['dest_lon_wgs84'].values
                fast_df['dest_lat_wgs84'] = slice_df['dest_lat_wgs84'].values
            fast_df['departure_time'] = t
            # no global accumulation; we only write per-slice parquet

            # write per-slice parquet caches during enrichment (optional)
            if self.cache_slices:
                path_all, path_fast = self._slice_cache_paths(t)
                try:
                    pd.DataFrame(slice_all_rows_local).to_parquet(path_all, index=False)
                    fast_df.to_parquet(path_fast, index=False)
                    slices_written += 1
                    print(f"  -> cached slice parquet: {path_all.name}, {path_fast.name}")
                except Exception as e:
                    print(f"  -> slice cache error: {e}")

        if slices_written == 0:
            print("No per-slice parquet written in enrichment phase (missing raw caches or empty slices).")
            return

        # Straight-line distance handled inside results_to_dataframe_with_options (no path/straight ratio kept).

        # Final note: per-slice parquet dataset is ready under slice_cache; read via glob/dataset
        print(f"Per-slice parquet dataset ready under: {self.slice_cache_dir}")

        print("Done.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schedule-driven PT routing (all paths + fastest)")
    parser.add_argument("--port", type=int, default=8989, help="GraphHopper port")
    parser.add_argument("--date", type=str, default="2025-05-13", help="Date (YYYY-MM-DD)")
    parser.add_argument("--start-hour", type=int, default=6, help="Start hour (UTC)")
    parser.add_argument("--end-hour", type=int, default=22, help="End hour (UTC)")
    parser.add_argument("--interval", type=int, default=15, help="Interval minutes (used to derive schedule filename if not provided)")
    parser.add_argument("--schedule-file", type=str, default=None, help="Schedule parquet: absolute path OR filename/substring searched in scenario output dir")
    parser.add_argument("--scenario", type=str, default="scenario_V10_2025", help="Scenario subdirectory under ./output")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit number of unique od_index pairs")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per slice on failure")
    parser.add_argument("--save-raw", action="store_true", help="(Reserved) Save raw responses – currently ignored in new workflow")
    parser.add_argument("--output", choices=["parquet", "csv", "both"], default="parquet", help="Final output format(s)")
    parser.add_argument("--phase", choices=["route", "enrich", "both"], default="both", help="Run only routing, only enrichment, or both")
    # Replace old --auto-start flag with paired flags to keep default True
    parser.add_argument("--auto-start", dest="auto_start", action="store_true", default=True, help="(default) Automatically start/restart backend if not running")
    parser.add_argument("--no-auto-start", dest="auto_start", action="store_false", help="Disable automatic backend start/restart")
    parser.add_argument("--no-quiet-backend", action="store_true", help="Do not suppress GraphHopper stdout/stderr")
    args = parser.parse_args()
    out_parquet = args.output in ("parquet", "both")
    out_csv = args.output in ("csv", "both")

    RoutingMain(
        gh_port=args.port,
        date=args.date,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        interval_minutes=args.interval,
        schedule_file=args.schedule_file,
        scenario_name=args.scenario,
        test_limit=args.test_limit,
        max_retries=args.max_retries,
        save_raw=args.save_raw,
        output_parquet=out_parquet,
        output_csv=out_csv,
        phase=args.phase,
        auto_start=args.auto_start,
        quiet_backend=not args.no_quiet_backend,
    ).run()