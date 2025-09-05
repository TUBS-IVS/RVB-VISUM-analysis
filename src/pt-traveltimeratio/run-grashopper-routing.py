from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import random
from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import requests

from shapely import wkt
from shapely.geometry import Point
# Optional CRS transform support
try:
    from pyproj import Transformer  # type: ignore
    _PYPROJ = True
except Exception:
    Transformer = None  # type: ignore
    _PYPROJ = False
# Shapely compatibility: from_wkb exists in Shapely 2.x; older versions use wkb.loads(hex=True)
try:
    from shapely import from_wkb as _from_wkb  # Shapely 2.x
    def _from_wkb_hex(s: str):
        return _from_wkb(bytes.fromhex(s))
except Exception:  # pragma: no cover
    from shapely import wkb as _wkb           # Shapely 1.x
    def _from_wkb_hex(s: str):
        return _wkb.loads(s, hex=True)

from gtfs_routing.transit_router import TransitRouter
from gtfs_routing.gtfs_setup import AppConfig, run_grashopper, GTFSRouteLookup
from results_processing import results_to_od_dataframe


BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


def _repo_root() -> Path:
    """Search upwards from CWD for a directory containing input/shapes/pt-data.
    This makes the script runnable from anywhere.
    """
    cwd = Path.cwd().resolve()
    here = cwd
    for _ in range(8):
        if (here / "input" / "shapes" / "pt-data").exists():
            return here
        if here.parent == here:
            break
        here = here.parent
    raise RuntimeError("Konnte Repo-Root nicht finden (input/shapes/pt-data fehlt). Bitte Skript im Repo ausführen.")


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


def _restore_geometry_columns(df: pd.DataFrame, src_epsg: Optional[str] = None, target_epsg: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    # Simple: transform endpoints from src_epsg to target_epsg if provided
    transformer = None
    if src_epsg and target_epsg and src_epsg != target_epsg and _PYPROJ:
        try:
            from pyproj import Transformer as _Transformer  # type: ignore
            transformer = _Transformer.from_crs(src_epsg, target_epsg, always_xy=True)
        except Exception as e:
            print(f"Warning: failed to create CRS transformer {src_epsg}->{target_epsg}: {e}")
            transformer = None

    # Restore endpoints from WKT lists
    for col in ["connector_far_endpoints_o", "connector_far_endpoints_d"]:
        wkt_col = col + "_wkt"
        if wkt_col in df.columns:
            def _restore_endpoints(val):
                if pd.isna(val) or not val:
                    return []
                try:
                    geom = wkt.loads(val)
                except Exception:
                    return []
                # If MultiPoint/GeometryCollection: use .geoms via getattr to keep linters happy
                geoms_attr = getattr(geom, "geoms", None)
                pts: List
                if geoms_attr is not None:
                    try:
                        pts = list(geoms_attr)
                    except Exception:
                        return []
                else:
                    # Single Point or other geometry: wrap into list
                    pts = [geom]
                # Optional transform to target CRS
                if transformer is not None:
                    out: List[Point] = []
                    for p in pts:
                        try:
                            x, y = transformer.transform(p.x, p.y)
                            out.append(Point(x, y))
                        except Exception:
                            continue
                    pts = out
                return pts
            df[col] = df[wkt_col].map(_restore_endpoints)

    # Restore any *_wkb geometries to shapely geometries (hex strings)
    wkb_cols = [c for c in df.columns if c.endswith("_wkb")]
    for col in wkb_cols:
        orig = col.removesuffix("_wkb") if hasattr(col, "removesuffix") else col.replace("_wkb", "")
        df[orig] = df[col].map(lambda s: _from_wkb_hex(s) if pd.notna(s) else None)
    return df


@dataclass
class RoutingMain:
    gh_port: int = 8989
    date: str = "2025-05-13"  # YYYY-MM-DD (UTC)
    start_hour: int = 6
    end_hour: int = 22
    interval_minutes: int = 15
    test_limit: Optional[int] = None  # if set, limit number of OD pairs per slice
    bbox: BBox = (10.45, 52.22, 10.60, 52.32)  # (min_lon, min_lat, max_lon, max_lat)
    src_epsg: str = "EPSG:25832"  # source CRS for endpoint points (ETRS89 / UTM zone 32N)
    auto_start: bool = False

    def __post_init__(self):
        self.repo_root = _repo_root()
        self.input_dir = self.repo_root / "input"
    # Parquet outputs are stored under the scenario directory
        self.output_dir = self.repo_root / "output" / "scenario_V10_2025"
        self.graphhopper_dir = self.input_dir / "graphhopper"
        self.gtfs_dir = self.input_dir / "gtfs-data" / "2025(V10)"
        self.gtfs_zip = self.gtfs_dir / "VM_RVB_Basisszenario2025_mDTicket_V10_init_LB GTFS_250827.zip"
        self.routes_txt = self.gtfs_dir / "routes.txt"

    def _prompt_start_backend(self) -> bool:
        ans = input("GraphHopper backend is not running. Start it now? [y/N]: ").strip().lower()
        return ans in ("y", "yes")

    def _start_backend(self) -> None:
        cfg = AppConfig(
            project_root=self.repo_root,
            graphhopper_dir=self.graphhopper_dir,
            gh_config_path=self.graphhopper_dir / "config.yml",
            gtfs_input_zip=self.gtfs_zip,
            output_base=self.repo_root / "output",
            scenario_name="scenario_V10_2025",
            gh_jar_path=self.graphhopper_dir / "graphhopper-web-10.2.jar",
            gh_cache_dir=self.graphhopper_dir / "graph-cache",
            gh_port=self.gh_port,
            # Adjust java_bin if your JDK lives elsewhere
            java_bin=Path(r"C:\Program Files\Eclipse Adoptium\jdk-21.0.3.9-hotspot\bin\java.exe"),
            java_opts=["-Xms8g", "-Xmx110g"],
        )
        # This will update config.yml, start GH, and wait until ready
        run_grashopper(cfg)

    def _load_routing_df(self) -> pd.DataFrame:
        pq = self.output_dir / "df_routing_OD.parquet"
        if not pq.exists():
            raise FileNotFoundError(f"Parquet not found: {pq}")
        df = pd.read_parquet(pq)
        # Restore endpoints and transform to WGS84 directly here
        df = _restore_geometry_columns(df, src_epsg=self.src_epsg, target_epsg="EPSG:4326")
        print(f"Reloaded rows: {len(df)}")
        return df

    def _build_coords_for_df(self, df: pd.DataFrame) -> List[Tuple[float, float, float, float]]:
        coords_list: List[Tuple[float, float, float, float]] = []
        if df.empty:
            return coords_list
        o_col = "connector_far_endpoints_o"
        d_col = "connector_far_endpoints_d"
        if o_col not in df.columns or d_col not in df.columns:
            return coords_list
        for row in df.itertuples(index=False):
            origins = getattr(row, o_col, None) or []
            dests = getattr(row, d_col, None) or []
            if not origins or not dests:
                continue
            for o in origins:
                for d in dests:
                    try:
                        coords_list.append((o.y, o.x, d.y, d.x))
                    except Exception:
                        continue
        return coords_list

    def _build_coords_and_meta_for_df(self, df: pd.DataFrame) -> Tuple[List[Tuple[float, float, float, float]], pd.DataFrame]:
        """Build coords and an aligned metadata frame per OD pair.
        Metadata includes row-level attributes and the origin/destination endpoint indices used.
        """
        coords_list: List[Tuple[float, float, float, float]] = []
        meta_rows: List[dict] = []
        if df.empty:
            return coords_list, pd.DataFrame()

        # Expected optional columns from OD relation DF
        opt_cols = [
            "origins", "destinations", "usage_rebus", "pt_transfers", "pt_trips",
            "origin_name", "destination_name", "origin_geometry", "destination_geometry",
            "connector_far_endpoints_o", "connector_far_endpoints_d",
        ]

        for row_id, row in df.iterrows():
            origins = row.get("connector_far_endpoints_o") or []
            dests = row.get("connector_far_endpoints_d") or []
            if not origins or not dests:
                continue

            # Base metadata copied for each OD pair from this row
            base = {c: row.get(c, None) for c in opt_cols if c in df.columns}
            # Convenient counts
            base["n_connectors_o"] = len(origins)
            base["n_connectors_d"] = len(dests)
            # Preserve row id for traceability (index of df)
            base["row_id"] = row_id

            for o_idx, o in enumerate(origins):
                for d_idx, d in enumerate(dests):
                    try:
                        coords_list.append((o.y, o.x, d.y, d.x))
                    except Exception:
                        continue
                    meta = dict(base)
                    meta["origin_endpoint_index"] = o_idx
                    meta["destination_endpoint_index"] = d_idx
                    meta_rows.append(meta)

        # Assign od_index aligned with coords_list order
        for i, m in enumerate(meta_rows):
            m["od_index"] = i
        meta_df = pd.DataFrame(meta_rows)
        return coords_list, meta_df

    def _count_expected_pairs(self, df: pd.DataFrame) -> int:
        """Count sum over rows of len(origins) * len(dests)."""
        if df.empty:
            return 0
        o_col = "connector_far_endpoints_o"
        d_col = "connector_far_endpoints_d"
        if o_col not in df.columns or d_col not in df.columns:
            return 0
        total = 0
        for row in df.itertuples(index=False):
            o = getattr(row, o_col, None) or []
            d = getattr(row, d_col, None) or []
            try:
                total += (len(o) * len(d))
            except Exception:
                continue
        return total

    def _time_slices(self, date_iso: str, start_h: int, end_h: int, interval_minutes: int) -> List[str]:
        # Build inclusive steps from start_h:00 to end_h:00 on the given date in UTC (Z)
        date_part = date_iso.split("T")[0]
        start_dt = datetime.fromisoformat(f"{date_part}T{start_h:02d}:00:00+00:00")
        end_dt = datetime.fromisoformat(f"{date_part}T{end_h:02d}:00:00+00:00")
        out: List[str] = []
        cur = start_dt
        step = max(1, int(interval_minutes))
        while cur <= end_dt:
            out.append(cur.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            cur += timedelta(minutes=step)
        return out

    def _random_coords_in_bbox(self, n: int = 2) -> List[Tuple[float, float, float, float]]:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        coords: List[Tuple[float, float, float, float]] = []
        for _ in range(n):
            origin_lat = random.uniform(min_lat, max_lat)
            origin_lon = random.uniform(min_lon, max_lon)
            dest_lat = random.uniform(min_lat, max_lat)
            dest_lon = random.uniform(min_lon, max_lon)
            coords.append((origin_lat, origin_lon, dest_lat, dest_lon))
        return coords

    # Removed old first-row helper; we batch all OD pairs per time slice now

    def run(self) -> None:
        # 1) Ensure backend is up
        if not _is_backend_ready(self.gh_port, timeout=2.0):
            print("GraphHopper backend is not running. Starting it now...")
            self._start_backend()
        else:
            print(f"GraphHopper is running on port {self.gh_port}.")

        # Start-up log with key settings and inputs
        pq_path = self.output_dir / "df_routing_OD.parquet"
        print("\n=== Routing run configuration ===")
        print(f"Date: {self.date}  Window: {self.start_hour:02d}:00 - {self.end_hour:02d}:00  Interval: {self.interval_minutes} min")
        print(f"Port: {self.gh_port}  Source CRS: {self.src_epsg}  Test limit: {self.test_limit}")
        print(f"Parquet: {pq_path}  Exists: {pq_path.exists()}")
        print(f"GTFS ZIP: {self.gtfs_zip}  Exists: {self.gtfs_zip.exists()}")
        print(f"routes.txt: {self.routes_txt}  Exists: {self.routes_txt.exists()}")

        # 2) Load and restore df_routing_OD
        df_routing_OD = self._load_routing_df()
        print(
            f"Columns present: connector_far_endpoints_o: {'connector_far_endpoints_o' in df_routing_OD.columns}, "
            f"connector_far_endpoints_d: {'connector_far_endpoints_d' in df_routing_OD.columns}"
        )

        # 2b) Compute expected counts and time slices
        expected_pairs = self._count_expected_pairs(df_routing_OD)
        time_slices = self._time_slices(self.date, self.start_hour, self.end_hour, self.interval_minutes)
        effective_pairs = min(expected_pairs, self.test_limit) if (self.test_limit is not None and self.test_limit > 0) else expected_pairs
        expected_total_requests = effective_pairs * len(time_slices)
        print(f"Expected OD pairs: {expected_pairs}; time slices: {len(time_slices)}; total requests (effective): {expected_total_requests}")

        # 3) Build complete coords list once (all origins x destinations per row) with metadata
        coords_list, od_meta = self._build_coords_and_meta_for_df(df_routing_OD)
        if self.test_limit is not None and self.test_limit > 0:
            coords_list = coords_list[: self.test_limit]
            od_meta = od_meta.iloc[: self.test_limit].copy()
            od_meta["od_index"] = range(len(od_meta))
        print(f"Total OD coordinate pairs: {len(coords_list)}")
        assert len(coords_list) == effective_pairs, (
            f"Mismatch in OD pairs: built {len(coords_list)} vs expected {effective_pairs} (before routing)"
        )
        if not coords_list:
            print("No coordinate pairs to route. Exiting.")
            return

        # 4) Iterate time slices from start_hour to end_hour
        print(f"Time slices to process: {len(time_slices)}")
        router = TransitRouter(port=self.gh_port)
        # optional GTFS lookup for route short names & modes
        gtfs_lookup = None
        try:
            gtfs_lookup = GTFSRouteLookup(self.gtfs_zip)
        except Exception as e:
            print(f"Warning: GTFSRouteLookup not available ({e}); proceeding without line/mode enrichment.")
        all_slices: List[pd.DataFrame] = []
        for idx, t in enumerate(time_slices, start=1):
            assert len(coords_list) == effective_pairs, (
                f"Slice {idx}: OD pair count changed unexpectedly ({len(coords_list)} != {effective_pairs})"
            )
            print(f"[{idx}/{len(time_slices)}] Routing time slice {t} with {len(coords_list)} pairs...")
            try:
                results = asyncio.run(router.batch_pt_routes_safe(coords_list, departure_time=t))
                print(f"  -> routes: {len(results)}")
                # Convert results to flat OD DataFrame for this slice
                df_slice = results_to_od_dataframe(results, gtfs_route_lookup=gtfs_lookup, coords_list=coords_list)
                # Join with OD metadata to include relation attributes and connector indices
                if not od_meta.empty:
                    df_slice = df_slice.merge(od_meta, on="od_index", how="left")
                df_slice["departure_time"] = t
                # Build a readable OD id: O{origin_idx+1}_D{dest_idx+1}, fallback OD_{od_index+1}
                try:
                    import numpy as np  # optional; used for isnan check
                except Exception:
                    np = None  # type: ignore
                def _mk_od_id(r):
                    oi = r.get("origin_endpoint_index")
                    di = r.get("destination_endpoint_index")
                    try:
                        if oi is not None and di is not None and (np is None or (not np.isnan(oi) and not np.isnan(di))):
                            return f"O{int(oi)+1}_D{int(di)+1}"
                    except Exception:
                        pass
                    try:
                        return f"OD_{int(r.get('od_index', 0))+1}"
                    except Exception:
                        return "OD_UNKNOWN"
                df_slice["od_id"] = df_slice.apply(_mk_od_id, axis=1)

                # Drop geometry and connector endpoint list/count columns
                drop_cols = [
                    "origin_geometry", "destination_geometry",
                    "connector_far_endpoints_o", "connector_far_endpoints_d",
                    "n_connectors_o", "n_connectors_d",
                ]
                df_slice = df_slice.drop(columns=[c for c in drop_cols if c in df_slice.columns], errors="ignore")

                # Reorder columns so OD info appears first
                preferred_order = [
                    "od_index", "od_id",
                    "origins", "destinations", "usage_rebus", "pt_transfers", "pt_trips",
                    "origin_name", "destination_name",
                    "origin_endpoint_index", "destination_endpoint_index",
                    "row_id",
                    "origin_lat", "origin_lon", "dest_lat", "dest_lon",
                    "departure_time",
                ]
                present_first = [c for c in preferred_order if c in df_slice.columns]
                remaining = [c for c in df_slice.columns if c not in present_first]
                df_slice = df_slice[present_first + remaining]
                all_slices.append(df_slice)
                print(f"  -> df_slice rows: {len(df_slice)}; columns: {len(df_slice.columns)}")
                # TODO: persist per-slice results if needed (e.g., Parquet/CSV)
            except Exception as e:
                print(f"  -> failed time slice {t}: {e}")

        # 5) Build final DataFrame from all slices
        if all_slices:
            df_all = pd.concat(all_slices, ignore_index=True)
            print(f"\nFinal results DataFrame: {df_all.shape[0]} rows, {df_all.shape[1]} columns across {len(time_slices)} slices.")
            # Persist final results as Parquet and CSV
            date_compact = self.date.replace("-", "")
            suffix = f"{self.start_hour:02d}-{self.end_hour:02d}_{self.interval_minutes}m"
            test_tag = f"_test{self.test_limit}" if (self.test_limit is not None and self.test_limit > 0) else ""
            base_name = f"pt_routing_results_{date_compact}_{suffix}{test_tag}"
            parquet_path = self.output_dir / f"{base_name}.parquet"
            csv_path = self.output_dir / f"{base_name}.csv"
            # For safe serialization, convert geometry/list object columns to strings
            df_save = df_all.copy()
            for col in [
                "origin_geometry", "destination_geometry",
                "connector_far_endpoints_o", "connector_far_endpoints_d",
            ]:
                if col in df_save.columns:
                    def _to_str(v):
                        if v is None:
                            return None
                        try:
                            # shapely geometry
                            wkt_attr = getattr(v, "wkt", None)
                            if wkt_attr is not None:
                                return v.wkt
                        except Exception:
                            pass
                        try:
                            return str(v)
                        except Exception:
                            return None
                    df_save[col] = df_save[col].map(_to_str)
            try:
                df_save.to_parquet(parquet_path, index=False)
                print(f"Saved Parquet -> {parquet_path}")
            except Exception as e:
                print(f"Failed to save Parquet: {e}")
            try:
                df_save.to_csv(csv_path, index=False)
                print(f"Saved CSV     -> {csv_path}")
            except Exception as e:
                print(f"Failed to save CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PT routing batches for a day in 15-min steps.")
    parser.add_argument("--port", type=int, default=8989, help="GraphHopper port (default: 8989)")
    parser.add_argument("--date", type=str, default="2025-05-13", help="Date (YYYY-MM-DD, UTC)")
    parser.add_argument("--start-hour", type=int, default=6, help="Start hour (0-23), default 6")
    parser.add_argument("--end-hour", type=int, default=22, help="End hour (0-23), default 22")
    parser.add_argument("--interval-minutes", type=int, default=15, help="Interval minutes, default 15")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit number of OD pairs for test runs")
    args = parser.parse_args()

    RoutingMain(
        gh_port=args.port,
        date=args.date,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        interval_minutes=args.interval_minutes,
        test_limit=args.test_limit,
    ).run()
