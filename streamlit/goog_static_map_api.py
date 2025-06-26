#from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

#load_dotenv(dotenv_path=".env")
#load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))


def google_api(lat, lon):
    zoom = 17
    size= "512x512"
    scale= 2
    api_key = os.environ.get("GOOGLE_API_KEY")
    #api_key = 'AIzaSyAz4skeLv37RPY2flqyUbnk6WU384yJvUA'

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}&scale={scale}&maptype=satellite&key={api_key}"
    )

    r = requests.get(url)
    if r.ok:
        img = Image.open(BytesIO(r.content))
        #os.makedirs("streamlit", exist_ok=True)
        save_path = os.path.join(os.path.dirname(__file__), 'images_masks', 'satellite_images', "post_disaster.png")
        #save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        print("Saved post image .png")
    else:
        print("Error when downloading pre image:", r.status_code)

if __name__ == '__main__':
    google_api(30.03, 31.36)
