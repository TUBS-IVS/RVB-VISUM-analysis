from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _sec(ms: Optional[float]) -> Optional[float]:
    if ms is None:
        return None
    try:
        return float(ms) / 1000.0
    except Exception:
        return None


def _parse_single_path(path: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single GraphHopper PT path into a flat dictionary with key metrics.
    Units: seconds for times, meters for distances.
    """
    legs: List[Dict[str, Any]] = path.get("legs", []) or []

    pure_pt_travel_time_ms: float = 0.0
    total_walking_distance_m: float = 0.0
    walking_to_first_pt_station_m: Optional[float] = None
    first_station_name: Optional[str] = None
    first_station_coord: Optional[Sequence[float]] = None  # [lon, lat]
    first_pt_found = False

    trip_headsigns: List[Optional[str]] = []
    route_ids: List[Optional[str]] = []
    pt_modes: List[Optional[str]] = []  # filled via lookup later
    pt_lines_shortname: List[Optional[str]] = []  # filled via lookup later

    first_leg_departure_time: Optional[str] = None
    final_arrival_time: Optional[str] = None

    for i, leg in enumerate(legs):
        if i == 0:
            first_leg_departure_time = leg.get("departure_time")
        if i == len(legs) - 1:
            final_arrival_time = leg.get("arrival_time")

        ltype = leg.get("type")
        if ltype == "walk":
            dist = leg.get("distance")
            if dist is not None:
                try:
                    total_walking_distance_m += float(dist)
                except Exception:
                    pass
            if not first_pt_found:
                # distance up to the first PT leg
                if walking_to_first_pt_station_m is None:
                    walking_to_first_pt_station_m = 0.0
                if dist is not None:
                    try:
                        walking_to_first_pt_station_m += float(dist)
                    except Exception:
                        pass
        elif ltype == "pt":
            tt = leg.get("travel_time")
            if tt is not None:
                try:
                    pure_pt_travel_time_ms += float(tt)
                except Exception:
                    pass
            if not first_pt_found:
                first_station_name = leg.get("departure_location")
                coords = (leg.get("geometry") or {}).get("coordinates")
                if coords:
                    first_station_coord = coords[0]
                first_pt_found = True
            if "trip_headsign" in leg:
                trip_headsigns.append(leg.get("trip_headsign"))
            if "route_id" in leg:
                rid = leg.get("route_id")
                route_ids.append(str(rid) if rid is not None else None)

    total_time_ms = path.get("time")
    gh_distance_m = path.get("distance")
    transfers = path.get("transfers")

    entry: Dict[str, Any] = {
        "first_station_name": first_station_name,
        "first_station_x": (first_station_coord[0] if first_station_coord else None),
        "first_station_y": (first_station_coord[1] if first_station_coord else None),
        "gh_time_s": _sec(total_time_ms),
        "pure_pt_travel_time_s": _sec(pure_pt_travel_time_ms),
        "gh_distance_m": gh_distance_m,
        "transfers": transfers,
        "walking_distance_to_first_station_m": walking_to_first_pt_station_m,
        "total_walking_distance_m": total_walking_distance_m,
        # placeholders to be filled after GTFS lookup
        "route_ids": route_ids,
        "trip_headsigns": trip_headsigns,
        "pt_lines_shortname": pt_lines_shortname,
        "pt_modes": pt_modes,
        "first_leg_departure_time": first_leg_departure_time,
        "final_arrival_time": final_arrival_time,
    }
    return entry


def _apply_gtfs_lookup(option: Dict[str, Any], gtfs_route_lookup: Any) -> None:
    if gtfs_route_lookup is None:
        return
    route_ids: List[Optional[str]] = option.get("route_ids") or []
    shortnames: List[Optional[str]] = []
    modes: List[Optional[str]] = []
    for rid in route_ids:
        try:
            short, mode = gtfs_route_lookup.get_route_info(rid)
        except Exception:
            short, mode = None, None
        shortnames.append(short)
        modes.append(mode)
    option["pt_lines_shortname"] = shortnames
    option["pt_modes"] = modes


def _flatten_selected(option: Dict[str, Any], idx: int,
                      coords: Optional[Tuple[float, float, float, float]],
                      walking_speed_mps: float) -> Dict[str, Any]:
    """
    Flatten the selected option to a single-row dict with stable column names.
    """
    route_ids: List[Optional[str]] = option.get("route_ids") or []
    shortnames: List[Optional[str]] = option.get("pt_lines_shortname") or []
    modes: List[Optional[str]] = option.get("pt_modes") or []
    heads: List[Optional[str]] = option.get("trip_headsigns") or []

    total_walk_m = option.get("total_walking_distance_m")
    normal_walking_time_s = (float(total_walk_m) / walking_speed_mps) if total_walk_m is not None else None
    pt_time_s = option.get("pure_pt_travel_time_s") or 0.0
    normal_total_time_s = (normal_walking_time_s + pt_time_s) if normal_walking_time_s is not None else None

    row: Dict[str, Any] = {
        "od_index": idx,
        "first_station_name": option.get("first_station_name"),
        "first_station_x": option.get("first_station_x"),
        "first_station_y": option.get("first_station_y"),
        "gh_time_s": option.get("gh_time_s"),
        "pure_pt_travel_time_s": option.get("pure_pt_travel_time_s"),
        "gh_distance_m": option.get("gh_distance_m"),
        "transfers": option.get("transfers"),
        "walking_distance_to_first_station_m": option.get("walking_distance_to_first_station_m"),
        "total_walking_distance_m": option.get("total_walking_distance_m"),
        "normal_walking_time_s": normal_walking_time_s,
        "normal_total_time_s": normal_total_time_s,
        "first_leg_departure_time": option.get("first_leg_departure_time"),
        "final_arrival_time": option.get("final_arrival_time"),
        # up to 3 PT lines (extend if needed)
        "pt_line_id_0": route_ids[0] if len(route_ids) > 0 else None,
        "pt_line_id_1": route_ids[1] if len(route_ids) > 1 else None,
        "pt_line_id_2": route_ids[2] if len(route_ids) > 2 else None,
        "pt_line_name_0": shortnames[0] if len(shortnames) > 0 else None,
        "pt_line_name_1": shortnames[1] if len(shortnames) > 1 else None,
        "pt_line_name_2": shortnames[2] if len(shortnames) > 2 else None,
        "pt_mode_0": modes[0] if len(modes) > 0 else None,
        "pt_mode_1": modes[1] if len(modes) > 1 else None,
        "pt_mode_2": modes[2] if len(modes) > 2 else None,
        "trip_headsign_0": heads[0] if len(heads) > 0 else None,
        "trip_headsign_1": heads[1] if len(heads) > 1 else None,
        "trip_headsign_2": heads[2] if len(heads) > 2 else None,
    }

    if coords is not None:
        o_lat, o_lon, d_lat, d_lon = coords
        row.update({
            "origin_lat": o_lat,
            "origin_lon": o_lon,
            "dest_lat": d_lat,
            "dest_lon": d_lon,
        })
    return row


def results_to_od_dataframe(results: Sequence[Optional[Dict[str, Any]]],
                            gtfs_route_lookup: Optional[Any] = None,
                            coords_list: Optional[Sequence[Tuple[float, float, float, float]]] = None,
                            walking_speed_mps: float = 1.33) -> pd.DataFrame:
    """
    Convert a list of GraphHopper PT results (one per OD) into a flat DataFrame
    with one row per OD pair, selecting the best option per OD for a single
    walking speed (normal person).

    - results: output from TransitRouter.batch_pt_routes_safe (...)
    - gtfs_route_lookup: optional object exposing get_route_info(route_id) -> (short_name, mode)
    - coords_list: optional list of (o_lat, o_lon, d_lat, d_lon); when provided
                   these are included in the output rows aligned by index
    - walking_speed_mps: used to compute normal walking time and total time
    """
    rows: List[Dict[str, Any]] = []

    for idx, res in enumerate(results):
        coords = coords_list[idx] if coords_list is not None and idx < len(coords_list) else None
        if not res:
            # No result for this OD
            rows.append(_flatten_selected({
                "gh_time_s": None,
                "pure_pt_travel_time_s": None,
                "gh_distance_m": None,
                "transfers": None,
                "total_walking_distance_m": None,
                "walking_distance_to_first_station_m": None,
                "route_ids": [],
                "trip_headsigns": [],
                "pt_lines_shortname": [],
                "pt_modes": [],
                "first_leg_departure_time": None,
                "final_arrival_time": None,
            }, idx, coords, walking_speed_mps))
            continue

        paths = res.get("paths") or []
        options = [_parse_single_path(p) for p in paths]

        # Enrich with GTFS lookup if available
        for opt in options:
            _apply_gtfs_lookup(opt, gtfs_route_lookup)

        # Select best option using walking_speed_mps
        best_opt: Optional[Dict[str, Any]] = None
        best_total: float = float("inf")
        for opt in options:
            tw = opt.get("total_walking_distance_m")
            if tw is None:
                continue
            pt_s = opt.get("pure_pt_travel_time_s") or 0.0
            try:
                total = float(tw) / walking_speed_mps + float(pt_s)
            except Exception:
                continue
            if total < best_total:
                best_total = total
                best_opt = opt

        # If nothing selected (e.g., all missing walk), fallback to first option if present
        if best_opt is None:
            best_opt = options[0] if options else {
                "gh_time_s": None,
                "pure_pt_travel_time_s": None,
                "gh_distance_m": None,
                "transfers": None,
                "total_walking_distance_m": None,
                "walking_distance_to_first_station_m": None,
                "route_ids": [],
                "trip_headsigns": [],
                "pt_lines_shortname": [],
                "pt_modes": [],
                "first_leg_departure_time": None,
                "final_arrival_time": None,
            }

        row = _flatten_selected(best_opt, idx, coords, walking_speed_mps)
        # Diagnostics: number of options
        row["n_options"] = len(options)
        rows.append(row)

    return pd.DataFrame(rows)
