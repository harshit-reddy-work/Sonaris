import json
from ..core.interfaces import GPSCoordinate

def read_gps_metadata(metadata_file: str) -> GPSCoordinate | None:
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        if latitude is not None and longitude is not None:
            return GPSCoordinate(latitude=float(latitude), longitude=float(longitude), source="metadata")
    except Exception:
        pass
    return None
