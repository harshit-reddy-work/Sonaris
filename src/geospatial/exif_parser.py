from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from ..core.interfaces import GPSCoordinate
import os

def _convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def extract_gps_from_exif(image_path_or_pil_image) -> GPSCoordinate | None:
    try:
        if isinstance(image_path_or_pil_image, str):
            if not os.path.exists(image_path_or_pil_image):
                return None
            image = Image.open(image_path_or_pil_image)
        else:
            image = image_path_or_pil_image
            
        exif = image._getexif()
        if not exif:
            return None
            
        gps_info = {}
        for key, val in exif.items():
            tag = TAGS.get(key, key)
            if tag == "GPSInfo":
                for t in val:
                    sub_tag = GPSTAGS.get(t, t)
                    gps_info[sub_tag] = val[t]
                    
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = _convert_to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") != "N":
                lat = -lat
            lon = _convert_to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") != "E":
                lon = -lon
            return GPSCoordinate(latitude=lat, longitude=lon, source="exif")
    except Exception:
        pass
    return None
