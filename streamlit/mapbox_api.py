#from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

#load_dotenv(dotenv_path=".env")
#load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))

def mapbox_api(lat, lon):
    zoom = 17
    size = (1024, 1024)
    mapbox_token = "pk.eyJ1IjoiY2hyaXN0aWFubWlyc2UiLCJhIjoiY21iaHo4b3dmMDA3YjJrcW5obTBwMDJjOCJ9.TUODbtq5ae1xvG1Q_h7rHQ"
    #mapbox_token = os.environ.get("MAPBOX_API_KEY")
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{size[0]}x{size[1]}?access_token={mapbox_token}"
    )

    save_path = os.path.join(os.path.dirname(__file__), 'images_masks', 'satellite_images', "pre_disaster.png")

    r = requests.get(url)
    if r.ok:
        img = Image.open(BytesIO(r.content))
        # os.makedirs("streamlit", exist_ok=True)

        # save_path = (
        #     Path(__file__)
        #     .resolve()
        #     .parents[2]
        #     / "images_masks" / "satellite_images" / "pre_disaster.png"
        #     )
        #save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        print("Saved pre image .png")
    else:
        print("Error when downloading pre image:", r.status_code)

    return save_path

if __name__ == '__main__':
    mapbox_api(31, 33)
