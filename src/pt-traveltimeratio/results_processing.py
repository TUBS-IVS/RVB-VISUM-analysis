from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import os
import pandas as pd
import numpy as np


def parse_graphhopper_response(path: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal parser for a single GraphHopper PT path (baseline version)."""
    pure_travel_time = 0.0
    total_walking_distance = 0.0
    walking_to_first_pt_station: Optional[float] = None
    first_station_name: Optional[str] = None
    first_station_coord: Optional[Sequence[float]] = None
    last_station_name: Optional[str] = None
    last_station_coord: Optional[Sequence[float]] = None
    trip_headsigns: List[Optional[str]] = []
    route_ids: List[Optional[str]] = []
    first_pt_found = False
    first_leg_departure_time: Optional[str] = None
    final_arrival_time: Optional[str] = None
    pt_legs: List[Dict[str, Any]] = []
    transfer_stations: List[Optional[str]] = []  # station names where transfer occurs (departure of next PT leg)
    transfer_coords: List[Optional[Sequence[float]]] = []  # coordinates associated with transfer station
    transfer_wait_times: List[Optional[float]] = []  # seconds waiting between PT legs
    total_transfer_wait_time: Optional[float] = None

    legs = path.get("legs", []) or []

    for i, leg in enumerate(legs):
        if i == 0:
            first_leg_departure_time = leg.get("departure_time")
        if i == len(legs) - 1:
            final_arrival_time = leg.get("arrival_time")

        ltype = (leg.get("type") or "").lower()
        if ltype == "walk":
            dist = leg.get("distance", 0) or 0
            try:
                dist_f = float(dist)
            except Exception:
                dist_f = 0.0
            total_walking_distance += dist_f
            # accumulate ALL walking until first PT leg is encountered
            if not first_pt_found:
                if walking_to_first_pt_station is None:
                    walking_to_first_pt_station = 0.0
                walking_to_first_pt_station += dist_f
        elif ltype == "pt":
            tt = leg.get("travel_time", 0) or 0
            try:
                pure_travel_time += float(tt)
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
                route_ids.append(leg.get("route_id"))
            pt_legs.append(leg)

    # Derive last station info (arrival of last PT leg)
    if pt_legs:
        last_leg = pt_legs[-1]
        # Try multiple candidate keys for arrival stop name (GraphHopper variations / robustness)
        _arrival_name = None
        candidate_keys = [
            "arrival_location","arrival_name","arrival_station","arrival_stop",
            "to_name","to_location","to","end_location","endName","end_name",
        ]
        for key in candidate_keys:
            val = last_leg.get(key)
            if val:
                _arrival_name = val
                break
        # Fallback: inspect list of stops if present
        if not _arrival_name:
            for list_key in ["stops","stop_list","stopTimes","stop_times","stations"]:
                stops = last_leg.get(list_key)
                if isinstance(stops, list) and stops:
                    # iterate reverse to get final stop
                    for stop in reversed(stops):
                        if isinstance(stop, dict):
                            for k in ["name","stop_name","station_name","label","id"]:
                                v = stop.get(k)
                                if v:
                                    _arrival_name = v
                                    break
                        if _arrival_name:
                            break
                if _arrival_name:
                    break
        last_station_name = _arrival_name
        last_coords = (last_leg.get("geometry") or {}).get("coordinates")
        if last_coords:
            last_station_coord = last_coords[-1]
        # Final fallback: if still None, reuse departure_location of last PT leg
        if last_station_name is None:
            dep_loc = last_leg.get("departure_location")
            if dep_loc:
                last_station_name = dep_loc
        # Optional debug once
        if last_station_name is None and os.environ.get("DEBUG_LAST_STATION") == "1":
            try:
                import json
                print("[DEBUG] last PT leg keys:", list(last_leg.keys()))
                for lk in ["stops","stop_list","stopTimes","stop_times","stations"]:
                    if lk in last_leg:
                        print(f"[DEBUG] sample {lk} entry:", (last_leg[lk][0] if isinstance(last_leg.get(lk), list) and last_leg[lk] else None))
            except Exception:
                pass

    # Derive transfer stations & wait times between consecutive PT legs
    if len(pt_legs) > 1:
        from datetime import datetime
        def _parse_time(ts: Optional[str]) -> Optional[datetime]:
            if not ts:
                return None
            try:
                # Allow both 'Z' suffix and naive
                if ts.endswith('Z'):
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                return datetime.fromisoformat(ts)
            except Exception:
                return None
        for i in range(len(pt_legs) - 1):
            cur_leg = pt_legs[i]
            nxt_leg = pt_legs[i+1]
            # Transfer station taken as departure_location of next leg
            t_station = nxt_leg.get("departure_location")
            coords_next = (nxt_leg.get("geometry") or {}).get("coordinates")
            t_coord = coords_next[0] if coords_next else None
            dep_time_next = _parse_time(nxt_leg.get("departure_time"))
            arr_time_cur = _parse_time(cur_leg.get("arrival_time"))
            wait_sec: Optional[float]
            if dep_time_next and arr_time_cur:
                wait_sec = (dep_time_next - arr_time_cur).total_seconds()
                if wait_sec is not None and wait_sec < 0:
                    # Guard against negative due to clock issues
                    wait_sec = None
            else:
                wait_sec = None
            transfer_stations.append(t_station)
            transfer_coords.append(t_coord)
            transfer_wait_times.append(wait_sec)
        # Compute aggregate wait if any values present
        valid_waits = [w for w in transfer_wait_times if isinstance(w, (int, float))]
        if valid_waits:
            total_transfer_wait_time = float(sum(valid_waits))

    total_time_ms = path.get("time", 0) or 0
    gh_distance = path.get("distance")
    # Normalize transfers: GraphHopper uses -1 for 'no PT legs' -> map to 0 for clarity
    raw_transfers = path.get("transfers")
    if raw_transfers is None or (isinstance(raw_transfers, (int, float)) and raw_transfers < 0):
        transfers_norm = 0
    else:
        transfers_norm = raw_transfers

    # Count PT legs actually present
    n_pt_legs = len([r for r in route_ids if r is not None])
    has_pt = n_pt_legs > 0

    return {
        "transfers": transfers_norm,
        "gh_time": float(total_time_ms) / 1000.0,
        "pure_pt_travel_time": pure_travel_time / 1000.0,
        "total_walking_distance": total_walking_distance,
        "walking_dist_to_first_pt_station": walking_to_first_pt_station,
        "first_station_name": first_station_name,
        "first_station_coord": first_station_coord,
    "last_station_name": last_station_name,
    "last_station_coord": last_station_coord,
    "transfer_stations": transfer_stations if transfer_stations else None,
    "transfer_coords": transfer_coords if transfer_coords else None,
    "transfer_wait_times": transfer_wait_times if transfer_wait_times else None,
        "total_transfer_wait_time": total_transfer_wait_time,
        "trip_headsign_0": trip_headsigns[0] if len(trip_headsigns) > 0 else None,
        "trip_headsign_1": trip_headsigns[1] if len(trip_headsigns) > 1 else None,
        "trip_headsign_2": trip_headsigns[2] if len(trip_headsigns) > 2 else None,
        "route_id_0": route_ids[0] if len(route_ids) > 0 else None,
        "route_id_1": route_ids[1] if len(route_ids) > 1 else None,
        "route_id_2": route_ids[2] if len(route_ids) > 2 else None,
        "first_leg_departure_time": first_leg_departure_time,
        "final_arrival_time": final_arrival_time,
        "has_pt": has_pt,
        "n_pt_legs": n_pt_legs,
    }


def select_best_path(paths: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not paths:
        return None
    best = None
    best_time = float("inf")
    for p in paths:
        t = p.get("time")
        try:
            t_sec = float(t) / 1000.0 if t is not None else float("inf")
        except Exception:
            t_sec = float("inf")
        if t_sec < best_time:
            best_time = t_sec
            best = p
    return best


def results_to_dataframe(results: List[Optional[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for od_index, res in enumerate(results):
        if not res:
            rows.append({
                "od_index": od_index,
                "gh_time": None,
                "pure_pt_travel_time": None,
                "total_walking_distance": None,
                "walking_dist_to_first_pt_station": None,
                "first_station_name": None,
                "first_station_coord": None,
                "trip_headsign_0": None,
                "trip_headsign_1": None,
                "trip_headsign_2": None,
                "route_id_0": None,
                "route_id_1": None,
                "route_id_2": None,
                "transfers": None,
                "first_leg_departure_time": None,
                "final_arrival_time": None,
            })
            continue
        paths = res.get("paths") or []
        best_path = select_best_path(paths)
        if best_path is None:
            rows.append({
                "od_index": od_index,
                "gh_time": None,
                "pure_pt_travel_time": None,
                "total_walking_distance": None,
                "walking_dist_to_first_pt_station": None,
                "first_station_name": None,
                "first_station_coord": None,
                "trip_headsign_0": None,
                "trip_headsign_1": None,
                "trip_headsign_2": None,
                "route_id_0": None,
                "route_id_1": None,
                "route_id_2": None,
                "transfers": None,
                "first_leg_departure_time": None,
                "final_arrival_time": None,
            })
            continue
        parsed = parse_graphhopper_response(best_path)
        parsed["od_index"] = od_index
        rows.append(parsed)
    return pd.DataFrame(rows)


def results_to_dataframe_with_options(
    results: List[Optional[Dict[str, Any]]],
    slice_departure_time: Optional[str] = None,
    walk_token: str = "walk",
    origin_lats: Optional[Sequence[float]] = None,
    origin_lons: Optional[Sequence[float]] = None,
    dest_lats: Optional[Sequence[float]] = None,
    dest_lons: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Extended variant returning fastest path metrics PLUS per-option lists.

            Adds (per OD index):
      - options_count: number of path alternatives returned by GraphHopper
      - departure_times_list: first PT leg departure time per path (fallback slice_departure_time)
      - total_times_list: total travel time seconds for each path (path['time']/1000)
      - routes_sequences: list of lists of route_ids for each path (PT legs only). Walk-only -> [walk_token].
      - best_route_sequence: sequence for the selected fastest path
            - number_of_transfers: derived transfers (len(best_route_sequence)-1) if PT else 0
                - straight_line_distance_m (great-circle, if coordinates provided)

    Parameters
    ----------
    results : list
        Raw GraphHopper responses (one per OD) each containing a 'paths' list.
    slice_departure_time : str, optional
        Fallback departure time if a path has no PT legs (user slice time).
    walk_token : str
        Placeholder token inserted when a path has no PT legs.
    """
    out_rows: List[Dict[str, Any]] = []
    for od_index, res in enumerate(results):
        paths = (res or {}).get("paths") or []
        options_count = len(paths)
        departure_times_list: List[str] = []
        total_times_list: List[Optional[float]] = []  # seconds
    # (Removed unreliable per-path distance list; keep only straight-line later)
        routes_sequences: List[List[str]] = []

        # Collect per-path lists
        for p in paths:
            legs = p.get("legs", []) or []
            first_dep: Optional[str] = None  # capture first leg departure (walk OR pt)
            route_seq: List[str] = []
            has_pt = False
            for leg in legs:
                if first_dep is None:
                    first_dep = leg.get("departure_time")
                ltype = (leg.get("type") or "").lower()
                if ltype == "pt":
                    has_pt = True
                    rid = leg.get("route_id")
                    if rid is not None:
                        route_seq.append(str(rid))
            if not has_pt:  # pure walk path -> mark with walk token
                route_seq = [walk_token]
            dep_val = first_dep or slice_departure_time or "UNKNOWN"
            departure_times_list.append(dep_val)
            # Total time (ms -> s)
            try:
                total_times_list.append(float(p.get("time", 0) or 0) / 1000.0)
            except Exception:
                total_times_list.append(None)
            routes_sequences.append(route_seq)

        # Determine best path (fastest) using existing selector
        best_path = select_best_path(paths)
        if best_path:
            parsed_best = parse_graphhopper_response(best_path)
            # Build best route sequence
            best_seq: List[str] = []
            for leg in best_path.get("legs", []) or []:
                if (leg.get("type") or "").lower() == "pt":
                    rid = leg.get("route_id")
                    if rid is not None:
                        best_seq.append(str(rid))
            if not best_seq:
                best_seq = [walk_token]
        else:
            parsed_best = {
                "gh_time": None,
                "pure_pt_travel_time": None,
                "total_walking_distance": None,
                "walking_dist_to_first_pt_station": None,
                "first_station_name": None,
                "first_station_coord": None,
                "last_station_name": None,
                "last_station_coord": None,
                "last_station_name": None,
                "last_station_coord": None,
                "trip_headsign_0": None,
                "trip_headsign_1": None,
                "trip_headsign_2": None,
                "route_id_0": None,
                "route_id_1": None,
                "route_id_2": None,
                "transfers": None,
                "first_leg_departure_time": None,
                "final_arrival_time": None,
                "has_pt": False,
                "n_pt_legs": 0,
                "total_transfer_wait_time": None,
            }
            best_seq = []

        # Derived number_of_transfers
        if parsed_best.get("has_pt"):
            pt_leg_count = len([x for x in best_seq if x and x != walk_token])
            number_of_transfers = max(0, pt_leg_count - 1)
        else:
            number_of_transfers = 0

        row = {
            "od_index": od_index,
            **parsed_best,
            "options_count": options_count,
            "departure_times_list": departure_times_list,
            "total_times_list": total_times_list,
            "routes_sequences": routes_sequences,
            "best_route_sequence": best_seq,
            "number_of_transfers": number_of_transfers,
        }

    # Optional straight-line distance (circuity ratio removed on request)
        try:
            if (origin_lats is not None and origin_lons is not None and dest_lats is not None and dest_lons is not None
        and len(origin_lats) == len(results)):
                lat1 = float(origin_lats[od_index]); lon1 = float(origin_lons[od_index])
                lat2 = float(dest_lats[od_index]); lon2 = float(dest_lons[od_index])
                rlat1 = np.radians(lat1); rlon1 = np.radians(lon1)
                rlat2 = np.radians(lat2); rlon2 = np.radians(lon2)
                dlat = rlat2 - rlat1; dlon = rlon2 - rlon1
                a = np.sin(dlat/2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                straight = 6371000.0 * c
                row['straight_line_distance_m'] = straight
        except Exception:
            pass
        out_rows.append(row)

    return pd.DataFrame(out_rows)


__all__ = [
    "parse_graphhopper_response",
    "select_best_path",
    "results_to_dataframe",
    "results_to_dataframe_with_options",
    "results_to_od_dataframe",
    "results_to_legacy_dataframe",
]


def results_to_od_dataframe(results, **_ignored):  # type: ignore
    """Compatibility wrapper matching previous interface used by routing script.

    Parameters
    ----------
    results : list
        List of raw GraphHopper responses (one per OD) where each item is a dict
        with a "paths" list, or None if routing failed.
    **_ignored : Any
        Additional legacy keyword arguments (gtfs_route_lookup, coords_list, etc.)
        are ignored in this minimal baseline implementation.

    Returns
    -------
    pandas.DataFrame
        Minimal flat DataFrame produced by results_to_dataframe.
    """
    return results_to_dataframe(results)


# ---------------------------------------------------------------------------
# Extended legacy-style export
# ---------------------------------------------------------------------------

def _extract_first_pt_station_from_path(path: Dict[str, Any]) -> Optional[str]:
    """Return the first PT departure_location (station name) in a path, or None."""
    for leg in path.get("legs", []) or []:
        if (leg.get("type") or "").lower() == "pt":
            return leg.get("departure_location")
    return None


def _collect_per_path_arrays(paths: List[Dict[str, Any]], slice_departure_time: str) -> Dict[str, Any]:
    """Collect list-valued columns for all options.

    Rules (per user clarification):
    - first_stations: first PT station name per path (None if no PT leg)
    - transfers: raw GraphHopper transfers value (keep -1 for walk-only)
    - departure_times_list: departure time of first PT leg; if no PT -> slice departure time
    - normal_all_total_times: path total travel time in seconds (path['time']/1000)
    """
    first_stations: List[Optional[str]] = []
    transfers_list: List[Optional[int]] = []
    dep_times: List[Optional[str]] = []
    total_times: List[Optional[float]] = []
    for p in paths:
        first_stations.append(_extract_first_pt_station_from_path(p))
        transfers_list.append(p.get("transfers"))
        # Determine first PT leg departure time
        first_dep = None
        for leg in p.get("legs", []) or []:
            if (leg.get("type") or "").lower() == "pt":
                first_dep = leg.get("departure_time")
                break
        dep_times.append(first_dep or slice_departure_time)
        try:
            total_times.append(float(p.get("time", 0)) / 1000.0)
        except Exception:
            total_times.append(None)
    return {
        "first_stations": first_stations,
        "transfers": transfers_list,
        "departure_times_list": dep_times,
        "normal_all_total_times": total_times,
    }


def _compute_walking_time_and_wait(path: Dict[str, Any]) -> Dict[str, Any]:
    """Compute walking_time (sum of walk leg travel_time) and total wait between PT legs.

    Assumptions (per user instruction):
    - Traveler arrives just-in-time for first PT leg -> initial wait excluded.
    - Wait time only sums gaps between arrival_time of PT leg i and departure_time of PT leg i+1 (if positive).
    - Leg times in GraphHopper: 'travel_time' in ms.
    """
    walking_time_ms = 0.0
    pt_events = []  # List of (dep_iso, arr_iso)
    for leg in path.get("legs", []) or []:
        ltype = (leg.get("type") or "").lower()
        if ltype == "walk":
            try:
                walking_time_ms += float(leg.get("travel_time", 0) or 0)
            except Exception:
                pass
        elif ltype == "pt":
            pt_events.append((leg.get("departure_time"), leg.get("arrival_time")))
    # Compute waits between consecutive PT legs (exclude initial)
    import datetime as _dt
    def _parse_iso(ts: Optional[str]):
        if not ts:
            return None
        try:
            return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None
    wait_seconds = 0.0
    if len(pt_events) > 1:
        for i in range(len(pt_events) - 1):
            arr = _parse_iso(pt_events[i][1])
            dep_next = _parse_iso(pt_events[i + 1][0])
            if arr and dep_next:
                delta = (dep_next - arr).total_seconds()
                if delta > 0:
                    wait_seconds += delta
    return {
        "normal_walking_time": walking_time_ms / 1000.0,
        "normal_total_wait_time_between_pt_legs": wait_seconds,
    }


def results_to_legacy_dataframe(
    results: List[Optional[Dict[str, Any]]],
    departure_time: str,
    od_meta: Optional[pd.DataFrame] = None,
    relation_id_col: str = "relation_id",
    sample_id_col: str = "sample_id",
) -> pd.DataFrame:
    """Produce a DataFrame matching the legacy column structure (as far as currently supported).

    Parameters
    ----------
    results : list of raw GraphHopper responses (one per OD)
    departure_time : str
        Slice departure time (ISO) applied to all rows.
    od_meta : DataFrame, optional
        Metadata aligned by index with ``results``. Expected columns (if available):
        - relation_id, sample_id
        - origin_name, destination_name
        - origin_point_wkt, destination_point_wkt (or similar; we attempt flexible matching)
    relation_id_col, sample_id_col : str
        Column names in od_meta for relation and sample identifiers.

    Returns
    -------
    DataFrame
        Columns in the legacy order. Missing enrichment (line names, modes) left as None.
    """
    legacy_rows: List[Dict[str, Any]] = []
    # Flexible metadata extraction helpers
    def _get(meta_row, *candidates):
        for c in candidates:
            if c in meta_row and pd.notna(meta_row[c]):
                return meta_row[c]
        return None

    # Auto-detect relation/sample id columns if not explicitly present
    if od_meta is not None:
        cols = set(od_meta.columns)
        if relation_id_col not in cols:
            for cand in ["relation_id", "origin_id", "origin", "origin_relation"]:
                if cand in cols:
                    relation_id_col = cand
                    break
        if sample_id_col not in cols:
            for cand in ["sample_id", "destination_id", "dest_id", "destination"]:
                if cand in cols:
                    sample_id_col = cand
                    break

    for idx, res in enumerate(results):
        paths = res.get("paths") if res else None
        paths_list: List[Dict[str, Any]] = paths or []
        n_options = len(paths_list)
        per_path_arrays = _collect_per_path_arrays(paths_list, departure_time) if paths_list else {
            "first_stations": [],
            "transfers": [],
            "departure_times_list": [],
            "normal_all_total_times": [],
        }
        best_path = select_best_path(paths_list) if paths_list else None
        parsed_best = parse_graphhopper_response(best_path) if best_path else {}
        extra_best = _compute_walking_time_and_wait(best_path) if best_path else {
            "normal_walking_time": None,
            "normal_total_wait_time_between_pt_legs": None,
        }

        # First / last PT station info (extend parsed_best)
        first_station_name = parsed_best.get("first_station_name")
        first_station_coord = parsed_best.get("first_station_coord") or [None, None]
        # Derive last station name/coord by scanning legs reverse
        last_station_name = None
        last_station_coord = [None, None]
        if best_path:
            for leg in reversed(best_path.get("legs", []) or []):
                if (leg.get("type") or "").lower() == "pt":
                    last_station_name = leg.get("arrival_location")
                    coords = (leg.get("geometry") or {}).get("coordinates")
                    if coords:
                        last_station_coord = coords[-1]
                    break

        meta_row = od_meta.iloc[idx] if (od_meta is not None and idx < len(od_meta)) else {}
        relation_id_val = meta_row.get(relation_id_col) if isinstance(meta_row, dict) else getattr(meta_row, relation_id_col, None)
        sample_id_val = meta_row.get(sample_id_col) if isinstance(meta_row, dict) else getattr(meta_row, sample_id_col, None)

        origin_name = _get(meta_row, "origin_name") if isinstance(meta_row, (dict, pd.Series)) else None
        destination_name = _get(meta_row, "destination_name") if isinstance(meta_row, (dict, pd.Series)) else None
        # Extend candidate lists to include connector_far_endpoints_*_wkt columns
        origin_point = _get(meta_row, "origin_point_wkt", "origin_point", "origin_wkt", "connector_far_endpoints_o_wkt") if isinstance(meta_row, (dict, pd.Series)) else None
        destination_point = _get(meta_row, "destination_point_wkt", "destination_point", "destination_wkt", "connector_far_endpoints_d_wkt") if isinstance(meta_row, (dict, pd.Series)) else None

        # Fallback names if missing
        if origin_name is None:
            origin_name = relation_id_val
        if destination_name is None:
            destination_name = sample_id_val

        row = {
            # Slice level
            "departure_time": departure_time,
            "relation_id": relation_id_val,
            "sample_id": sample_id_val,
            "n_options": n_options,
            "first_stations": per_path_arrays["first_stations"],
            "transfers": per_path_arrays["transfers"],
            "departure_times_list": per_path_arrays["departure_times_list"],
            "normal_all_total_times": per_path_arrays["normal_all_total_times"],
            # Chosen path core metrics
            "normal_total_time": parsed_best.get("gh_time"),
            "normal_walking_time": extra_best.get("normal_walking_time"),
            "normal_first_station_name": first_station_name,
            "normal_last_station_name": last_station_name,
            "normal_first_station_x": first_station_coord[0] if isinstance(first_station_coord, (list, tuple)) else None,
            "normal_first_station_y": first_station_coord[1] if isinstance(first_station_coord, (list, tuple)) else None,
            "normal_last_station_x": last_station_coord[0] if isinstance(last_station_coord, (list, tuple)) else None,
            "normal_last_station_y": last_station_coord[1] if isinstance(last_station_coord, (list, tuple)) else None,
            "normal_gh_time": parsed_best.get("gh_time"),
            "normal_pure_pt_travel_time": parsed_best.get("pure_pt_travel_time"),
            "normal_gh_distance": parsed_best.get("gh_distance"),
            "normal_transfers": parsed_best.get("transfers"),
            "normal_walking_distance_to_first_station": parsed_best.get("walking_dist_to_first_pt_station"),
            "normal_total_walking_distance": parsed_best.get("total_walking_distance"),
            # PT line IDs
            "normal_pt_line_id_0": parsed_best.get("route_id_0"),
            "normal_pt_line_id_1": parsed_best.get("route_id_1"),
            "normal_pt_line_id_2": parsed_best.get("route_id_2"),
            # Line names & modes left None (future enrichment)
            "normal_pt_line_name_0": None,
            "normal_pt_line_name_1": None,
            "normal_pt_line_name_2": None,
            "normal_pt_mode_0": None,
            "normal_pt_mode_1": None,
            "normal_pt_mode_2": None,
            # Headsigns
            "normal_trip_headsign_0": parsed_best.get("trip_headsign_0"),
            "normal_trip_headsign_1": parsed_best.get("trip_headsign_1"),
            "normal_trip_headsign_2": parsed_best.get("trip_headsign_2"),
            # Timing
            "normal_first_leg_departure_time": parsed_best.get("first_leg_departure_time"),
            "normal_final_arrival_time": parsed_best.get("final_arrival_time"),
            "normal_total_wait_time_between_pt_legs": extra_best.get("normal_total_wait_time_between_pt_legs"),
            # Duplicates / meta
            "normal_relation_id": relation_id_val,
            "normal_origins": relation_id_val,  # Placeholder: without explicit origin list per sample
            "normal_origin_name": origin_name,
            "normal_origin_point": origin_point,
            "normal_destinations": sample_id_val,  # Placeholder symmetry
            "normal_destination_name": destination_name,
            "normal_destination_point": destination_point,
            "normal_sample_id": sample_id_val,
            "normal_departure_time_used": departure_time,
        }
        legacy_rows.append(row)

    # Column order as per legacy header
    col_order = [
        "departure_time", "relation_id", "sample_id", "n_options", "first_stations", "transfers", "departure_times_list",
        "normal_all_total_times", "normal_total_time", "normal_walking_time", "normal_first_station_name",
        "normal_last_station_name", "normal_first_station_x", "normal_first_station_y", "normal_last_station_x", "normal_last_station_y",
        "normal_gh_time", "normal_pure_pt_travel_time", "normal_gh_distance", "normal_transfers",
        "normal_walking_distance_to_first_station", "normal_total_walking_distance", "normal_pt_line_id_0", "normal_pt_line_id_1",
        "normal_pt_line_id_2", "normal_pt_line_name_0", "normal_pt_line_name_1", "normal_pt_line_name_2", "normal_pt_mode_0",
        "normal_pt_mode_1", "normal_pt_mode_2", "normal_trip_headsign_0", "normal_trip_headsign_1", "normal_trip_headsign_2",
        "normal_first_leg_departure_time", "normal_final_arrival_time", "normal_total_wait_time_between_pt_legs", "normal_relation_id",
        "normal_origins", "normal_origin_name", "normal_origin_point", "normal_destinations", "normal_destination_name",
        "normal_destination_point", "normal_sample_id", "normal_departure_time_used"
    ]
    df = pd.DataFrame(legacy_rows)
    # Ensure all expected columns exist (if some metadata missing)
    for c in col_order:
        if c not in df.columns:
            df[c] = None
    return df[col_order]
