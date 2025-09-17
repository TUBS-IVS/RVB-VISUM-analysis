# -*- coding: utf-8 -*-
"""
Centralized setup for GTFS + GraphHopper + data loading.

- One place for all paths and knobs (AppConfig)
- Clean helpers to update GraphHopper config, run GraphHopper,
  and load buildings in a bounding box
- Robust GTFS route lookup that reads routes.txt from a GTFS ZIP
"""
from __future__ import annotations

import os
import pickle
import yaml
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

# External modules expected in your environment
from .gtfs_modifier import GTFSModifier
from .graphhoppermanager import GraphHopperManager

# ==============================
# Configuration
# ==============================

@dataclass
class AppConfig:
    # -------------------------------------------------------------------------
    # Project roots
    # -------------------------------------------------------------------------
    # Repository root inferred from this file location
    project_root: Path = Path(__file__).resolve().parents[4]
    # Default GH directory under input/
    graphhopper_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4] / "input" / "graphhopper")

    # -------------------------------------------------------------------------
    # GraphHopper config.yml
    # -------------------------------------------------------------------------
    gh_config_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4] / "input" / "graphhopper" / "config.yml")

    # -------------------------------------------------------------------------
    # GTFS input
    # -------------------------------------------------------------------------
    # Fallback default; prefer overriding in runner
    gtfs_input_zip: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4] / "input" / "gtfs-data" / "2019(V9)" / "GTFS_VISUM.zip")

    # -------------------------------------------------------------------------
    # Output base directory
    # -------------------------------------------------------------------------
    output_base: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4] / "output")

    # -------------------------------------------------------------------------
    # Work CRS for routing
    # -------------------------------------------------------------------------
    work_crs: str = "EPSG:4326"

    # -------------------------------------------------------------------------
    # Bounding box [south, west, north, east] in WGS84
    # -------------------------------------------------------------------------
    bbox_wgs84: Tuple[float, float, float, float] = (52.15, 10.40, 52.40, 10.70)

    # -------------------------------------------------------------------------
    # Scenario settings
    # -------------------------------------------------------------------------
    scenario_name: str = "scenario_V10_2025"

    # Toggle to use cached results
    use_cache: bool = True

    # -------------------------------------------------------------------------
    # GraphHopper runtime settings
    # -------------------------------------------------------------------------
    gh_jar_path: Optional[Path] = None   # default resolved lazily
    gh_cache_dir: Optional[Path] = None  # default resolved lazily
    gh_port: int = 8989

    # -------------------------------------------------------------------------
    # Java runtime settings
    # -------------------------------------------------------------------------
    java_bin: Optional[Path] = None      # set explicit JDK (e.g. JDK 21)
    java_opts: List[str] = field(default_factory=lambda: ["-Xmx110g"])
    # Example: ["-Xms32g", "-Xmx110g"]

    # -------------------------------------------------------------------------
    # Optional environment override for PROJ (GDAL / pyproj)
    # -------------------------------------------------------------------------
    proj_lib: Optional[Path] = None

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    def scenario_output_dir(self) -> Path:
        """Return the output directory for the current scenario."""
        return self.output_base / self.scenario_name

    def ensure_dirs(self) -> None:
        """Ensure scenario output directory exists."""
        self.scenario_output_dir().mkdir(parents=True, exist_ok=True)

    @property
    def resolved_jar_path(self) -> Path:
        """Return GraphHopper JAR path, falling back to default location."""
        return self.gh_jar_path or (self.graphhopper_dir / "graphhopper-web-10.2.jar")

    @property
    def resolved_cache_dir(self) -> Path:
        """Return GraphHopper cache dir, falling back to default location."""
        return self.gh_cache_dir or (self.graphhopper_dir / "graph-cache")


# ==============================
# GraphHopper helpers
# ==============================

def update_gtfs_path_in_config(cfg: AppConfig, gtfs_path: Path) -> None:
    """Update graphhopper/config.yml so 'graphhopper.gtfs.file' points to the provided GTFS zip."""
    config_path = cfg.gh_config_path
    if not config_path.exists():
        raise FileNotFoundError(f"GraphHopper config.yml not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("graphhopper", {})
    # Prefer project-root relative paths starting with 'input/' for portability
    project_root = cfg.project_root
    def _rel_to_root(p: Path) -> str:
        try:
            rel = p.relative_to(project_root)
            return str(rel).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")
    data["graphhopper"]["gtfs.file"] = _rel_to_root(gtfs_path)
    data["graphhopper"]["graph.location"] = _rel_to_root(cfg.resolved_cache_dir)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Updated GraphHopper GTFS path in config.yml -> {gtfs_path}")


def run_grashopper(
    cfg: AppConfig,
    move_stops: Optional[List[Tuple[str, float, float]]] = None,
    add_stops: Optional[List[Tuple[str, float, float, str, Optional[str]]]] = None,
    remove_stops: Optional[List[str]] = None,
    *,
    quiet: bool = False,
) -> None:
    """
    Run GraphHopper for a scenario, with optional GTFS modifications.

    Steps:
    1. Apply GTFS modifications (move/add/remove stops) if provided.
    2. Save modified GTFS or use input GTFS if no modifications.
    3. Update GraphHopper config.yml to point to the correct GTFS.
    4. Start or rebuild GraphHopper using GraphHopperManager.
    5. Wait until GraphHopper /health endpoint reports ready.
    """

    # -------------------------------------------------------------------------
    # 0) Optional PROJ_LIB override (for pyproj/GeoPandas if CRS errors occur)
    # -------------------------------------------------------------------------
    if cfg.proj_lib:
        os.environ["PROJ_LIB"] = str(cfg.proj_lib)

    # Ensure the scenario output directory exists
    cfg.ensure_dirs()
    scenario_output_dir = cfg.scenario_output_dir()

    # -------------------------------------------------------------------------
    # 1) GTFS modifications (if any)
    # -------------------------------------------------------------------------
    modifications_required = any([move_stops, add_stops, remove_stops])

    if modifications_required:
        print("Modifying GTFS...")
        # Use input GTFS as base for modifications
        mod = GTFSModifier(gtfs_zip_path=str(cfg.gtfs_input_zip))

        # Move stops
        if move_stops:
            for sid, lat, lon in move_stops:
                print(f" - Moving stop {sid} to ({lat}, {lon})")
                mod.move_stop(sid, lat, lon)

        # Add virtual stops
        if add_stops:
            for entry in add_stops:
                print(f" - Adding stop {entry[0]} at ({entry[1]}, {entry[2]})")
                mod.add_virtual_stop(*entry)

        # Remove stops
        if remove_stops:
            for sid in remove_stops:
                print(f" - Removing stop {sid}")
                mod.remove_stop(sid)

        # Save modified GTFS to scenario output dir
        gtfs_path = scenario_output_dir / "gtfs_modified.zip"
        mod.save(str(gtfs_path))
        mod.cleanup()
        print(f"GTFS modifications written to {gtfs_path}")

    else:
        print("No GTFS modifications required. Using existing GTFS input.")
        gtfs_path = cfg.gtfs_input_zip

    # -------------------------------------------------------------------------
    # 2) Update GraphHopper config.yml with correct GTFS path
    # -------------------------------------------------------------------------
    update_gtfs_path_in_config(cfg, gtfs_path)

    # -------------------------------------------------------------------------
    # 3) Configure GraphHopperManager from AppConfig
    # -------------------------------------------------------------------------
    gh = GraphHopperManager(
        config_path=str(cfg.gh_config_path),
        graph_cache_dir=str(cfg.resolved_cache_dir),
        jar_path=str(cfg.resolved_jar_path),
        java_bin=str(cfg.java_bin) if cfg.java_bin else None,
        java_mem=cfg.java_opts,  # Liste von Flags (["-Xms32g", "-Xmx110g"])
        port=cfg.gh_port,
    )

    log_path = str(cfg.graphhopper_dir / "gh.log")

    # -------------------------------------------------------------------------
    # 4) Start strategy
    # -------------------------------------------------------------------------
    print(f"Checking GraphHopper status on port {cfg.gh_port} ...")

    # Always clear cache first 
    print(f"Clearing cache directory before start: {cfg.resolved_cache_dir}")
    gh.clear_cache()

    if gh.is_ready(timeout=3):
        if not quiet:
            print(f"GraphHopper is already running on http://{gh.host}:{gh.port} (cache reused).")
    else:
        if not quiet:
            print("Starting GraphHopper process...")
            print(f"  JAR file   : {cfg.resolved_jar_path}")
            print(f"  Config file: {cfg.gh_config_path}")
            print(f"  Log file   : {log_path}")
        gh.start(log_file=log_path, quiet=quiet)

    # -------------------------------------------------------------------------
    # 5) Wait for readiness
    # -------------------------------------------------------------------------
    if not gh.wait_until_ready(timeout=1800):
        raise RuntimeError("GraphHopper did not become ready.")
    if not quiet:
        print("GraphHopper is ready.")


# ==============================
# GTFS route lookup
# ==============================

class GTFSRouteLookup:
    """Resolve route_id to (route_short_name, mode_string)."""

    GTFS_ROUTE_TYPES: Dict[int, str] = {
        0: "Tram / Light Rail",
        1: "Subway / Metro",
        2: "Rail",
        3: "Bus",
        4: "Ferry",
        5: "Cable Car",
        6: "Gondola / Suspended Cable Car",
        7: "Funicular",
    }

    def __init__(self, gtfs_zip_path: Path):
        self.routes = self._load_routes(gtfs_zip_path)

    def _load_routes(self, gtfs_zip_path: Path) -> pd.DataFrame:
        if not gtfs_zip_path.exists():
            raise FileNotFoundError(f"GTFS ZIP not found at {gtfs_zip_path}")

        with zipfile.ZipFile(gtfs_zip_path) as zf:
            name_candidates = [n for n in zf.namelist() if n.lower().endswith("routes.txt")]
            if not name_candidates:
                raise FileNotFoundError("routes.txt not found inside GTFS ZIP")
            routes_name = name_candidates[0]
            with zf.open(routes_name) as f:
                df = pd.read_csv(f, dtype={"route_id": str, "route_type": "Int64", "route_short_name": str, "route_long_name": str})

        if "route_short_name" not in df.columns and "route_long_name" in df.columns:
            df["route_short_name"] = df["route_long_name"]
        for col in ["route_id", "route_short_name"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        if "route_type" not in df.columns:
            df["route_type"] = pd.Series([pd.NA] * len(df), dtype="Int64")

        return df[["route_id", "route_short_name", "route_type"]].copy()

    def get_route_info(self, route_id: str) -> Tuple[Optional[str], Optional[str]]:
        if route_id is None:
            return None, None
        m = self.routes[self.routes["route_id"] == str(route_id)]
        if m.empty:
            return None, None
        short = m.iloc[0]["route_short_name"]
        rtype = m.iloc[0]["route_type"]
        mode = self.GTFS_ROUTE_TYPES.get(int(rtype)) if pd.notna(rtype) else None
        return short, mode
