import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path


class GTFSModifier:
    def __init__(self, gtfs_zip_path: str = "../data/input/connect_low_level_stops.zip", temp_dir: str = "../data/temp/gtfs_edit"):
        self.gtfs_zip_path = Path(gtfs_zip_path)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.gtfs_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)

        # Load required files
        self.stops = pd.read_csv(self.temp_dir / "stops.txt")
        self.stop_times = pd.read_csv(self.temp_dir / "stop_times.txt")

    def move_stop(self, stop_id: str, new_lat: float, new_lon: float):
        self.stops.loc[self.stops["stop_id"] == stop_id, ["stop_lat", "stop_lon"]] = [new_lat, new_lon]

    def add_virtual_stop(self, new_stop_id: str, lat: float, lon: float, stop_name="Virtual Stop", copy_trip_id=None):
        # Add to stops.txt
        new_stop = pd.DataFrame([{
            "stop_id": new_stop_id,
            "stop_name": stop_name,
            "stop_lat": lat,
            "stop_lon": lon
        }])
        self.stops = pd.concat([self.stops, new_stop], ignore_index=True)

        if copy_trip_id:
            # Get max sequence to append new stop
            existing_times = self.stop_times[self.stop_times["trip_id"] == copy_trip_id]
            max_seq = existing_times["stop_sequence"].max() if not existing_times.empty else 0

            new_time = pd.DataFrame([{
                "trip_id": copy_trip_id,
                "arrival_time": "08:00:00",
                "departure_time": "08:00:00",
                "stop_id": new_stop_id,
                "stop_sequence": max_seq + 1
            }])
            self.stop_times = pd.concat([self.stop_times, new_time], ignore_index=True)

    def remove_stop(self, stop_id: str):
        self.stops = self.stops[self.stops["stop_id"] != stop_id]
        self.stop_times = self.stop_times[self.stop_times["stop_id"] != stop_id]

    def save(self, output_zip_path: str = "data/output/gtfs_modified.zip"):
        # Save edited files
        self.stops.to_csv(self.temp_dir / "stops.txt", index=False)
        self.stop_times.to_csv(self.temp_dir / "stop_times.txt", index=False)

        # Repackage zip
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in self.temp_dir.glob("*.txt"):
                zipf.write(file, arcname=file.name)

    def cleanup(self):
        shutil.rmtree(self.temp_dir)
