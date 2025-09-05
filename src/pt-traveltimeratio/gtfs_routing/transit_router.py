import requests
import aiohttp
import asyncio
from datetime import datetime, timedelta


class TransitRouter:
    def __init__(self, port=8989, max_parallel_requests=10):
        self.base_url = f"http://localhost:{port}"
        self.max_parallel_requests = max_parallel_requests

    def get_car_route(self, start_lat, start_lon, end_lat, end_lon):
        url = f"{self.base_url}/route"
        params = {
            "point": [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
            "profile": "car",
            "points_encoded": "false"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if "paths" in data and len(data["paths"]) > 0:
                path = data["paths"][0]
                return {
                    "time": path["time"] / 60000,
                    "distance": path["distance"],
                    "segments": path.get("legs", [])
                }
            else:
                return None
        except requests.RequestException as e:
            print(f"Routing error (car): {e}")
            return None

    def get_pt_route(self, start_lat, start_lon, end_lat, end_lon, departure_time="2025-04-28T08:00:00Z"):
        url = f"{self.base_url}/route"
        params = {
            "point": [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
            "profile": "pt",
            "pt.earliest_departure_time": departure_time,
            "pt.limit_solutions": 1,
            "points_encoded": "false"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
            # data = response.json()
            # if "paths" in data and len(data["paths"]) > 0:
            #     return self._parse_pt_response(data["paths"][0])
            # else:
            #     return None
        except requests.RequestException as e:
            print(f"Routing error (pt): {e}")
            return None

    async def get_pt_route_async(self, session, start_lat, start_lon, end_lat, end_lon,
                                departure_time="2025-04-28T08:00:00Z"):
        # departure_time in datetime umwandeln
        dep_time_dt = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
        dep_time_dt = dep_time_dt - timedelta(minutes=15)
        
        # wenn latest_departure_time nicht gesetzt ist, automatisch +15 Minuten
        
        latest_dep_time_dt = dep_time_dt + timedelta(minutes=30)
        latest_departure_time = latest_dep_time_dt.isoformat().replace("+00:00", "Z")

        url = f"{self.base_url}/route"
        params = {
            "point": [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
            "profile": "pt",
            "pt.earliest_departure_time": departure_time,
            "pt.latest_departure_time": latest_departure_time,
            "pt.limit_solutions": 3,
            "pt.profile": "true",
            "points_encoded": "false"
        }

        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data
                # if "paths" in data and len(data["paths"]) > 0:
                #     return self._parse_pt_response(data["paths"][0])
                # else:
                #     return None
        except Exception as e:
            print(f"Async routing error: {e}")
            return None

    async def batch_pt_routes(self, coords_list, departure_time="2025-04-28T08:00:00Z"):
        """
        coords_list: list of (start_lat, start_lon, end_lat, end_lon)
        """
        connector = aiohttp.TCPConnector(limit=self.max_parallel_requests)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.get_pt_route_async(session, *coords, departure_time=departure_time) for coords in coords_list]
            results = await asyncio.gather(*tasks)
        return results
    
    async def _run_batch(self, coords_list, departure_time):
        connector = aiohttp.TCPConnector(limit=self.max_parallel_requests)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.wait_for(
                    self.get_pt_route_async(session, *coords, departure_time=departure_time),
                    timeout=200,
                )
                for coords in coords_list
            ]
            return await asyncio.gather(*tasks)
        

    async def batch_pt_routes_safe(self, coords_list, departure_time="2025-04-28T08:00:00Z", batch_size=100,
                                    max_retries=20):
            all_results = []
            total = len(coords_list)

            for i in range(0, total, batch_size):
                batch = coords_list[i:i + batch_size]
                print(f"Routing batch {i // batch_size + 1} ({i} to {i + len(batch) - 1})...")

                for attempt in range(1, max_retries + 1):
                    try:
                        results = await self._run_batch(batch, departure_time)                    
                        if results is None or all(r is None for r in results):
                            raise RuntimeError("Batch returned only None results")
                        all_results.extend(results)
                        break  # success
                    except Exception as e:
                        print(f"Batch failed on attempt {attempt}: {e}")
                        if attempt == max_retries:
                            raise RuntimeError(f"Max retries reached for batch {i // batch_size + 1}") from e
                            #print("Giving up on this batch.")
                            #all_results.extend([None] * len(batch))  # placeholder
                        else:
                            await asyncio.sleep(2 * attempt)  # backoff

            return all_results
    
