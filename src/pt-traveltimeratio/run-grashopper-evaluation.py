from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

import routing_helpers as rh
from results_processing import (
    results_to_dataframe_with_options,
    parse_graphhopper_response,
    select_best_path,
)


def _sanitize_time_for_filename(t: str) -> str:
    """Alias to routing_helpers implementation for consistent filenames."""
    return rh.sanitize_time_for_filename(t)


@dataclass
class EvaluationMain:
    """Evaluate previously routed raw cache slices and export results.

    - Reads output/<scenario>/raw_cache/*.json.gz created by the routing phase
    - Uses the schedule parquet to align meta columns (od_index, endpoints, coords)
    - Optionally writes per-slice parquet caches (allpaths + fastest)
    - Streams the fastest rows into a single CSV without loading all slices in memory
    """

    date: str
    start_hour: int
    end_hour: int
    interval_minutes: int
    scenario_name: str = "scenario_V10_2025"
    schedule_file: str | None = None
    cache_slices: bool = False
    export_best_csv: bool = True

    def __post_init__(self):
        self.repo_root = rh.repo_root()
        self.output_dir = self.repo_root / "output" / self.scenario_name
        self.slice_cache_dir = self.output_dir / "slice_cache"
        self.raw_cache_dir = self.output_dir / "raw_cache"
        if self.cache_slices:
            self.slice_cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- paths ----------
    def _raw_cache_path(self, departure_time: str) -> Path:
        date_compact = self.date.replace('-', '')
        t_safe = _sanitize_time_for_filename(departure_time)
        base_suffix = f"{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
        base = f"slice_{date_compact}_{base_suffix}_{t_safe}"
        return self.raw_cache_dir / f"{base}_raw.json.gz"

    def _slice_cache_paths(self, departure_time: str) -> Tuple[Path, Path]:
        date_compact = self.date.replace('-', '')
        t_safe = _sanitize_time_for_filename(departure_time)
        base_suffix = f"{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
        base = f"slice_{date_compact}_{base_suffix}_{t_safe}"
        return (
            self.slice_cache_dir / f"{base}_allpaths.parquet",
            self.slice_cache_dir / f"{base}_fastest.parquet",
        )

    def _final_fastest_csv_path(self) -> Path:
        return self.output_dir / "df_routing_OD.csv"

    # ---------- schedule ----------
    def _expected_schedule_path(self) -> Path:
        date_compact = self.date.replace('-', '')
        return self.output_dir / (
            f"df_routing_OD_schedule_{date_compact}_{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m_wgs84.parquet"
        )

    def _resolve_schedule_path(self) -> Path:
        if self.schedule_file:
            cand = Path(self.schedule_file)
            if cand.exists():
                return cand.resolve()
            in_scenario = self.output_dir / cand.name
            if in_scenario.exists():
                return in_scenario.resolve()
        expected = self._expected_schedule_path()
        if expected.exists():
            return expected.resolve()
        # fallback by prefix
        prefix = f"df_routing_OD_schedule_{self.date.replace('-', '')}_{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
        candidates = sorted(self.output_dir.glob(f"{prefix}*.parquet"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0].resolve()
        raise FileNotFoundError(f"Schedule parquet not found in {self.output_dir}")

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
        df['departure_time'] = df['departure_time'].astype(str)
        return df

    # ---------- processing ----------
    def _process_slice(self, t: str, slice_df: pd.DataFrame, out_csv: Path | None) -> int:
        raw_path = self._raw_cache_path(t)
        if not raw_path.exists():
            print(f"  - missing raw: {raw_path.name}")
            return 0
        try:
            with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            print(f"  - read error {raw_path.name}: {e}")
            return 0

        # all-paths rows (optional parquet cache)
        if self.cache_slices:
            slice_all_rows_local: List[dict] = []
            for local_idx, raw in enumerate(results):
                if not raw:
                    continue
                paths = raw.get('paths', []) or []
                best = select_best_path(paths)
                best_id = id(best) if best else None
                meta = slice_df.iloc[local_idx]
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
                    slice_all_rows_local.append({
                        'departure_time': t,
                        'slice_row_index': local_idx,
                        'od_index': meta.get('od_index'),
                        'pair_id': meta.get('pair_id'),
                        'origin_endpoint_index': meta.get('origin_endpoint_index'),
                        'destination_endpoint_index': meta.get('destination_endpoint_index'),
                        'origin_lon_wgs84': meta.get('origin_lon_wgs84'),
                        'origin_lat_wgs84': meta.get('origin_lat_wgs84'),
                        'dest_lon_wgs84': meta.get('dest_lon_wgs84'),
                        'dest_lat_wgs84': meta.get('dest_lat_wgs84'),
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
                    })

            path_all, path_fast = self._slice_cache_paths(t)
            try:
                pd.DataFrame(slice_all_rows_local).to_parquet(path_all, index=False)
            except Exception as e:
                print(f"  - write allpaths parquet error: {e}")

        # fastest dataframe
        fast_df = results_to_dataframe_with_options(
            results,
            slice_departure_time=t,
            origin_lats=slice_df['origin_lat_wgs84'].values,
            origin_lons=slice_df['origin_lon_wgs84'].values,
            dest_lats=slice_df['dest_lat_wgs84'].values,
            dest_lons=slice_df['dest_lon_wgs84'].values,
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

        # optional per-slice fastest parquet
        if self.cache_slices:
            _, path_fast = self._slice_cache_paths(t)
            try:
                fast_df.to_parquet(path_fast, index=False)
            except Exception as e:
                print(f"  - write fastest parquet error: {e}")

        # append to one CSV if requested
        written = 0
        if out_csv is not None:
            header = not out_csv.exists()
            try:
                fast_df.to_csv(out_csv, mode='a', header=header, index=False)
                written = len(fast_df)
            except Exception as e:
                print(f"  - csv append error: {e}")
        return written

    def run(self) -> None:
        schedule = self._load_schedule()
        time_slices: List[str] = sorted(schedule['departure_time'].astype(str).unique())
        print(f"Time slices: {len(time_slices)} ({time_slices[0]} .. {time_slices[-1]})")

        out_csv = self._final_fastest_csv_path() if self.export_best_csv else None
        if out_csv and out_csv.exists():
            try:
                out_csv.unlink()
            except Exception:
                pass

        total = 0
        with tqdm(total=len(time_slices), desc="Evaluate", unit="slice", dynamic_ncols=True) as pbar:
            for t in time_slices:
                slice_df = schedule[schedule['departure_time'] == t].copy()
                wrote = self._process_slice(t, slice_df, out_csv)
                total += wrote
                pbar.set_postfix_str(f"total rows: {total}")
                pbar.update(1)

        if out_csv:
            print(f"Final fastest CSV: {out_csv} ({total} rows)")
        if self.cache_slices:
            print(f"Slice parquet dataset: {self.slice_cache_dir}")


def test_raw_files(scenario: str, date: str, start_hour: int, end_hour: int, interval: int) -> None:
    """Check all raw_cache/*.json.gz for corruption and print a report."""
    repo_root = rh.repo_root()
    output_dir = repo_root / "output" / scenario
    raw_cache_dir = output_dir / "raw_cache"
    files = sorted(raw_cache_dir.glob("*.json.gz"))
    print(f"Testing {len(files)} raw cache files in {raw_cache_dir}...")
    bad = []
    for f in tqdm(files, desc="Test raw", unit="file", dynamic_ncols=True):
        try:
            with gzip.open(f, 'rt', encoding='utf-8') as fh:
                json.load(fh)
        except Exception as e:
            print(f"  - Corrupt: {f.name} ({e})")
            bad.append(f.name)
    print(f"\nChecked {len(files)} files. Corrupt: {len(bad)}")
    if bad:
        print("Corrupt files:")
        for b in bad:
            print(f"  {b}")
    else:
        print("All files OK.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GraphHopper routing raw caches slice-by-slice")
    parser.add_argument("--scenario", type=str, default="scenario_V10_2025", help="Scenario under ./output")
    parser.add_argument("--date", type=str, required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--start-hour", type=int, required=True, help="Start hour (UTC)")
    parser.add_argument("--end-hour", type=int, required=True, help="End hour (UTC)")
    parser.add_argument("--interval", type=int, default=15, help="Interval minutes")
    parser.add_argument("--schedule-file", type=str, default=None, help="Schedule parquet (absolute or in scenario dir)")
    parser.add_argument("--cache-slices", action="store_true", help="Write per-slice parquet caches (allpaths + fastest)")
    parser.add_argument("--no-export-best-csv", action="store_true", help="Do not write the single fastest CSV")
    parser.add_argument("--test-raw", action="store_true", help="Only test raw_cache/*.json.gz for corruption and print a report")

    args = parser.parse_args()
    if args.test_raw:
        test_raw_files(args.scenario, args.date, args.start_hour, args.end_hour, args.interval)
        return
    EvaluationMain(
        date=args.date,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        interval_minutes=args.interval,
        scenario_name=args.scenario,
        schedule_file=args.schedule_file,
        cache_slices=args.cache_slices,
        export_best_csv=not args.no_export_best_csv,
    ).run()


if __name__ == "__main__":
    main()
