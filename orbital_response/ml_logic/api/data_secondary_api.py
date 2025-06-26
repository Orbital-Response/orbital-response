from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import math

#load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))

tile_size_m = 300  # Approx size in meters
resolution = 512   # Image size (pixels)
zoom = 17
grid_radius = 4    # (8x8 grid)

def google_api(lat, lon, tile_name):
    size= "512x512"
    scale= 2
    api_key = os.getenv("GOOGLE_API_KEY")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}&scale={scale}&maptype=satellite&key={api_key}"
    )

    output_dir = Path("../data_secondary")
    output_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(url)
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save(f"{output_dir}/{tile_name}")
        print(f"Saved {output_dir.name}")
    else:
        print(f"Failed to download tile at {lat}, {lon}")

locations = {
    "gaza_city_shejaiya": (31.49944, 34.458220),
    "gaza_city_jabalya": (31.530831, 34.496223),
    "khan_yunis": (31.353261, 34.292597),
    "nuseirat_camp": (31.459875, 34.391665),
    "al_arara": (31.375398, 34.333428),
    "beit_hanoun": (31.536525, 34.540841)
}

def meters_per_pixel(lat, zoom):
    """Approximate meters per pixel at a given latitude and zoom."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

for city, (center_lat, center_lon) in locations.items():
    mpp = meters_per_pixel(center_lat, zoom)
    offset_deg = (tile_size_m / 2) / (111_320)  # Approx lat offset in degrees

    for i in range(-grid_radius, grid_radius + 1):
        for j in range(-grid_radius, grid_radius + 1):
            lat_offset = i * tile_size_m * mpp / 111_320  # latitude degrees
            lon_offset = j * tile_size_m * mpp / (111_320 * math.cos(math.radians(center_lat)))  # longitude degrees

            tile_lat = center_lat + lat_offset
            tile_lon = center_lon + lon_offset

            tile_name = f"{city}_{i+grid_radius}_{j+grid_radius}"
            google_api(tile_lat, tile_lon, tile_name)
