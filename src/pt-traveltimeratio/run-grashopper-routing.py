from __future__ import annotations

import asyncio
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import os
import shutil

import pandas as pd
import numpy as np
import requests

from gtfs_routing.transit_router import TransitRouter
from gtfs_routing.gtfs_setup import AppConfig, run_grashopper, GTFSRouteLookup
from results_processing import (
    results_to_dataframe_with_options,
    parse_graphhopper_response,
    select_best_path,
)

BBox = Tuple[float, float, float, float]


def _resolve_java_bin() -> Path:
    env = os.environ.get("GH_JAVA_BIN")
    if env:
        p = Path(env)
        if p.exists():
            return p
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        cand = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if cand.exists():
            return cand
    if os.name == "nt":
        candidates: List[Path] = []
        pf_vars = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramW6432")]
        patterns = [
            "Eclipse Adoptium/jdk*/*/bin/java.exe",
            "Eclipse Adoptium/jdk*/bin/java.exe",
            "AdoptOpenJDK/jdk*/bin/java.exe",
            "Java/jdk*/bin/java.exe",
            "Zulu/zulu*/bin/java.exe",
        ]
        for base_str in pf_vars:
            if not base_str:
                continue
            base = Path(base_str)
            for pattern in patterns:
                for match in base.glob(pattern):
                    candidates.append(match)
        if candidates:
            def _sort_key(p: Path):
                return str(p.parent)
            best = sorted(candidates, key=_sort_key, reverse=True)[0]
            if best.exists():
                return best
    which = shutil.which("java")
    if which:
        return Path(which)
    return Path("java")

def _repo_root() -> Path:
    cwd = Path.cwd().resolve()
    here = cwd
    for _ in range(8):
        if (here / "input" / "shapes" / "pt-data").exists():
            return here
        if here.parent == here:
            break
        here = here.parent
    raise RuntimeError("Could not find the repo root (input/shapes/pt-data missing). Please run this script from within the repository.")

def _health_urls(port: int) -> Sequence[str]:
    base = f"http://localhost:{port}"
    return (f"{base}/health", f"{base}/actuator/health")

def _is_backend_ready(port: int, timeout: float = 0.0) -> bool:
    import time
    end = time.time() + max(0.0, timeout)
    while True:
        for url in _health_urls(port):
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                pass
        if timeout <= 0 or time.time() >= end:
            return False
        time.sleep(1.0)

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
    auto_start: bool = False                # start backend if not running
    scenario_name: str = "scenario_V10_2025"  # output/<scenario_name>

    def __post_init__(self):
        self.repo_root = _repo_root()
        self.output_dir = self.repo_root / "output" / self.scenario_name
        self.input_dir = self.repo_root / "input"
        self.graphhopper_dir = self.input_dir / "graphhopper"
        self.gtfs_dir = self.input_dir / "gtfs-data" / "2025(V10)"
        self.gtfs_zip = self.gtfs_dir / "VM_RVB_Basisszenario2025_mDTicket_V10_init_LB GTFS_250827.zip"
        self.routes_txt = self.gtfs_dir / "routes.txt"

    def _prompt_start_backend(self) -> bool:
        ans = input("GraphHopper backend is not running. Start it now? [y/N]: ").strip().lower()
        return ans in ("y", "yes")

    def _start_backend(self) -> None:
        java_bin = _resolve_java_bin()
        print(f"Using Java binary: {java_bin}")
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
            java_opts=["-Xms8g", "-Xmx110g"],
        )
        run_grashopper(cfg)

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

    def run(self) -> None:
        # Backend availability
        if not _is_backend_ready(self.gh_port, timeout=2.0):
            if self.auto_start:
                print("Backend not running – starting GraphHopper...")
                self._start_backend()
            else:
                print("Backend not running – please start manually or pass --auto-start (not implemented flag).")
        else:
            print(f"GraphHopper running on port {self.gh_port}.")

        # Load schedule (full) then restrict for test limit
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
        print(f"Time slices (used): {len(time_slices)} ({time_slices[0]} .. {time_slices[-1]})")

        # Router + optional GTFS lookup
        router = TransitRouter(port=self.gh_port)
        try:
            gtfs_lookup = GTFSRouteLookup(self.gtfs_zip)
        except Exception as e:
            print(f"Warning: GTFSRouteLookup failed ({e}); continuing without route info enrichment.")
            gtfs_lookup = None

        all_paths_rows: List[dict] = []
        fastest_rows: List[pd.DataFrame] = []

        # Timing / estimation helpers
        routing_wall_start = time.time()
        accumulated_routing_seconds = 0.0  # sum of pure routing calls
        successful_requests = 0  # number of (od,slice) requests actually returned a result list (even empty paths)
        expected_requests_restricted = restricted_unique_od * len(time_slices)
        expected_requests_full = total_unique_od_full * len(time_slices_full)

        date_compact = self.date.replace('-', '')
        base_suffix = f"{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
        test_tag = f"_test{self.test_limit}" if (self.test_limit and self.test_limit > 0) else ""

        for idx, t in enumerate(time_slices, start=1):
            slice_df = schedule[schedule['departure_time'] == t].copy()
            if slice_df.empty:
                continue
            # Build coordinate list (lat, lon order expected by router)
            coords_list = list(zip(
                slice_df['origin_lat_wgs84'],
                slice_df['origin_lon_wgs84'],
                slice_df['dest_lat_wgs84'],
                slice_df['dest_lon_wgs84'],
            ))
            print(f"[{idx}/{len(time_slices)}] Routing slice {t} -> {len(coords_list)} pairs")
            # Retry loop
            results = None
            slice_route_start = time.time()
            for attempt in range(1, self.max_retries + 1):
                try:
                    results = asyncio.run(router.batch_pt_routes_safe(coords_list, departure_time=t))
                    break
                except Exception as e:
                    if attempt >= self.max_retries:
                        print(f"  -> FAILED slice {t} after {attempt} attempts: {e}")
                    else:
                        delay = min(30, 2 * attempt)
                        print(f"  -> attempt {attempt} error: {e} (retry in {delay}s)")
                        time.sleep(delay)
            if results is None:
                continue
            slice_route_end = time.time()
            accumulated_routing_seconds += (slice_route_end - slice_route_start)
            successful_requests += len(coords_list)
            # ---- All paths table ----
            for local_idx, raw in enumerate(results):
                if not raw:
                    continue
                paths = raw.get('paths', []) or []
                best = select_best_path(paths)
                best_id = id(best) if best else None
                meta_row = slice_df.iloc[local_idx]
                for path_idx, p in enumerate(paths):
                    parsed = parse_graphhopper_response(p) or {}
                    # Build route sequence for this path
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
                    all_paths_rows.append({
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
                        # removed unreliable gh_distance per user request
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
                    })
            # ---- Fastest (with option lists) ----
            fast_df = results_to_dataframe_with_options(
                results,
                slice_departure_time=t,
                origin_lats=slice_df['origin_lat_wgs84'].values,
                origin_lons=slice_df['origin_lon_wgs84'].values,
                dest_lats=slice_df['dest_lat_wgs84'].values,
                dest_lons=slice_df['dest_lon_wgs84'].values,
            )
            # Re-map od_index to global; keep local position
            fast_df['slice_row_index'] = fast_df['od_index']
            fast_df['od_index'] = slice_df['od_index'].values
            if 'pair_id' in slice_df.columns:
                fast_df['pair_id'] = slice_df['pair_id'].values
            fast_df['origin_endpoint_index'] = slice_df['origin_endpoint_index'].values
            fast_df['destination_endpoint_index'] = slice_df['destination_endpoint_index'].values
            # carry coordinates
            if 'origin_lon_wgs84' in slice_df.columns:
                fast_df['origin_lon_wgs84'] = slice_df['origin_lon_wgs84'].values
                fast_df['origin_lat_wgs84'] = slice_df['origin_lat_wgs84'].values
                fast_df['dest_lon_wgs84'] = slice_df['dest_lon_wgs84'].values
                fast_df['dest_lat_wgs84'] = slice_df['dest_lat_wgs84'].values
            fast_df['departure_time'] = t
            fastest_rows.append(fast_df)

        if not fastest_rows and not all_paths_rows:
            print("No results produced.")
            return

        df_all_paths = pd.DataFrame(all_paths_rows)
        df_fastest = pd.concat(fastest_rows, ignore_index=True) if fastest_rows else pd.DataFrame()

        print(f"All paths rows: {len(df_all_paths)} | Fastest rows: {len(df_fastest)}")

    # Straight-line distance handled inside results_to_dataframe_with_options (no path/straight ratio kept).

        # Estimation block (only when test_limit set and we processed subset)
        if self.test_limit and restricted_unique_od < total_unique_od_full and successful_requests > 0:
            wall_total = time.time() - routing_wall_start
            # Use pure accumulated routing time for per-request average
            per_request_seconds = accumulated_routing_seconds / successful_requests
            estimated_total_seconds = per_request_seconds * expected_requests_full
            remaining_seconds = max(0.0, estimated_total_seconds - accumulated_routing_seconds)

            def _fmt(sec: float) -> str:
                m, s = divmod(int(sec + 0.5), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            print("\n=== Runtime Estimation (test mode) ===")
            print(f"Processed unique OD pairs: {restricted_unique_od} / {total_unique_od_full} ({restricted_unique_od/total_unique_od_full:.2%})")
            print(f"Time slices: {len(time_slices)}")
            print(f"Successful routed requests (pairs * slices): {successful_requests} / {expected_requests_restricted}")
            print(f"Accumulated routing call time: {_fmt(accumulated_routing_seconds)}")
            print(f"Wall clock elapsed (includes overhead): {_fmt(wall_total)}")
            print(f"Avg seconds per request (routing only): {per_request_seconds:.3f}s")
            print(f"Estimated full requests: {expected_requests_full}")
            print(f"Estimated total routing time (pure): {_fmt(estimated_total_seconds)}")
            print(f"Estimated remaining (pure) if full run: {_fmt(remaining_seconds)}")

        base_name_all = f"pt_routing_allpaths_{date_compact}_{base_suffix}{test_tag}"
        base_name_fast = f"pt_routing_fastest_{date_compact}_{base_suffix}{test_tag}"

        # Save
        try:
            df_all_paths.to_parquet(self.output_dir / f"{base_name_all}.parquet", index=False)
            df_fastest.to_parquet(self.output_dir / f"{base_name_fast}.parquet", index=False)
            print("Saved parquet outputs.")
        except Exception as e:
            print(f"Parquet save error: {e}")
        try:
            df_all_paths.to_csv(self.output_dir / f"{base_name_all}.csv", index=False)
            df_fastest.to_csv(self.output_dir / f"{base_name_fast}.csv", index=False)
            print("Saved CSV outputs.")
        except Exception as e:
            print(f"CSV save error: {e}")

        # --- Summary stats ---
        try:
            print("\n=== Summary Metrics ===")
            # Core counts
            unique_od_processed = df_fastest['od_index'].nunique() if 'od_index' in df_fastest.columns else None
            print(f"Unique OD pairs processed (fastest table): {unique_od_processed}")
            print(f"Time slices processed: {len(time_slices)}")
            # All-paths richness
            if not df_all_paths.empty:
                try:
                    avg_paths_per_request = (
                        df_all_paths.groupby(['od_index','departure_time']).size().mean()
                    )
                except Exception:
                    avg_paths_per_request = None
                print(f"Average alternative paths per request: {avg_paths_per_request:.2f}" if avg_paths_per_request is not None else "Average alternative paths per request: n/a")
            # Numeric helpers
            def _fmt_mean(df, col, factor=1.0, suffix=''):
                if col in df.columns and not df[col].dropna().empty:
                    return f"{df[col].mean():.2f}{suffix}" if factor == 1.0 else f"{(df[col].mean()/factor):.2f}{suffix}"
                return "n/a"
            # Travel times (seconds -> minutes)
            avg_gh_min = _fmt_mean(df_fastest, 'gh_time', 60.0, ' min')
            med_gh_min = (
                f"{(df_fastest['gh_time'].median()/60.0):.2f} min" if 'gh_time' in df_fastest.columns and not df_fastest['gh_time'].dropna().empty else 'n/a'
            )
            # Avg pure PT travel time only for rows that actually used PT
            if 'has_pt' in df_fastest.columns and 'pure_pt_travel_time' in df_fastest.columns:
                mask_pt_time = df_fastest['has_pt'] == True
                if mask_pt_time.any():
                    avg_pt_travel_min = f"{(df_fastest.loc[mask_pt_time, 'pure_pt_travel_time'].mean()/60.0):.2f} min"
                    med_pt_travel_min = f"{(df_fastest.loc[mask_pt_time, 'pure_pt_travel_time'].median()/60.0):.2f} min"
                else:
                    avg_pt_travel_min = "n/a"
                    med_pt_travel_min = "n/a"
            else:
                avg_pt_travel_min = "n/a"
                med_pt_travel_min = "n/a"
            # Walking (split by mode usage) & transfers
            if 'has_pt' in df_fastest.columns and 'total_walking_distance' in df_fastest.columns:
                walk_col = 'total_walking_distance'
                series_walk = df_fastest[walk_col]
                # PT rows
                mask_pt_rows = df_fastest['has_pt'] == True
                if mask_pt_rows.any():
                    avg_walk_dist_pt = f"{series_walk[mask_pt_rows].mean():.2f} m"
                else:
                    avg_walk_dist_pt = "n/a"
                # Pure walk rows
                mask_walk_only = df_fastest['has_pt'] == False
                if mask_walk_only.any():
                    avg_walk_dist_walk_only = f"{series_walk[mask_walk_only].mean():.2f} m"
                else:
                    avg_walk_dist_walk_only = "n/a"
                # Overall (optional)
                if not series_walk.dropna().empty:
                    avg_walk_dist_overall = f"{series_walk.mean():.2f} m"
                else:
                    avg_walk_dist_overall = "n/a"
            else:
                avg_walk_dist_pt = avg_walk_dist_walk_only = avg_walk_dist_overall = "n/a"
            avg_transfers = _fmt_mean(df_fastest, 'transfers')
            # Average PT legs only among rows that actually used PT (exclude pure walk)
            if 'has_pt' in df_fastest.columns and 'n_pt_legs' in df_fastest.columns and not df_fastest['n_pt_legs'].dropna().empty:
                mask_pt = df_fastest['has_pt'] == True  # explicit True to avoid truthiness surprises
                if mask_pt.any():
                    avg_pt_legs = f"{df_fastest.loc[mask_pt, 'n_pt_legs'].mean():.2f}"
                else:
                    avg_pt_legs = "n/a"
            else:
                avg_pt_legs = "n/a"
            # Share with PT
            share_has_pt = (
                f"{(df_fastest['has_pt'].mean()*100):.1f}%" if 'has_pt' in df_fastest.columns and not df_fastest['has_pt'].dropna().empty else 'n/a'
            )
            print(f"Avg total GH time (fastest)      : {avg_gh_min} (median {med_gh_min})")
            print(f"Avg pure PT travel time          : {avg_pt_travel_min} (median {med_pt_travel_min})")
            print(f"Avg total walking distance (all) : {avg_walk_dist_overall}")
            print(f"Avg walking distance (PT rows)   : {avg_walk_dist_pt}")
            print(f"Avg walking distance (walk-only) : {avg_walk_dist_walk_only}")
            print(f"Avg transfers (fastest paths)    : {avg_transfers}")
            # If derived number_of_transfers exists, report its mean similarly (PT only)
            if 'number_of_transfers' in df_fastest.columns:
                if 'has_pt' in df_fastest.columns:
                    mask_pt2 = df_fastest['has_pt'] == True
                    if mask_pt2.any():
                        avg_number_transfers = f"{df_fastest.loc[mask_pt2, 'number_of_transfers'].mean():.2f}"
                    else:
                        avg_number_transfers = "n/a"
                else:
                    avg_number_transfers = f"{df_fastest['number_of_transfers'].mean():.2f}" if not df_fastest['number_of_transfers'].empty else 'n/a'
                print(f"Avg derived transfers (best seq): {avg_number_transfers}")
            # Circuity metrics
            if 'straight_line_distance_m' in df_fastest.columns:
                sld = df_fastest['straight_line_distance_m']
                if not sld.dropna().empty:
                    overall = sld.mean()
                    print(f"Avg straight-line distance (all) : {overall:.2f} m")
                    if 'has_pt' in df_fastest.columns:
                        sld_pt = sld[df_fastest['has_pt'] == True]
                        sld_walk = sld[df_fastest['has_pt'] == False]
                        if not sld_pt.dropna().empty:
                            print(f"  -> with PT                     : {sld_pt.mean():.2f} m")
                        if not sld_walk.dropna().empty:
                            print(f"  -> walk-only                   : {sld_walk.mean():.2f} m")
            # Aggregate total transfer wait time
            if 'total_transfer_wait_time' in df_fastest.columns:
                mask_pt_wait = df_fastest['has_pt'] == True if 'has_pt' in df_fastest.columns else pd.Series([True]*len(df_fastest))
                waits = df_fastest.loc[mask_pt_wait, 'total_transfer_wait_time'].dropna() if 'total_transfer_wait_time' in df_fastest.columns else pd.Series(dtype=float)
                if not waits.empty:
                    print(f"Avg total transfer wait (PT rows): {waits.mean():.2f} s (median {waits.median():.2f} s)")
            print(f"Avg # PT legs (only PT rows)     : {avg_pt_legs}")
            print(f"Share with any PT (has_pt)       : {share_has_pt}")
        except Exception as e:
            print(f"Summary metrics failed: {e}")

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
    args = parser.parse_args()

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
    ).run()
