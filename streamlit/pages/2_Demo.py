import streamlit as st
import numpy as np
import sys
import os
from pathlib import Path
import folium
import requests
from dotenv import load_dotenv
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw
from streamlit_folium import st_folium
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
from ml_logic.model.model import get_model
from ml_logic.model.model import get_model_destruction
# Importing Google API
from goog_static_map_api import google_api
# Importing MapBox API
from mapbox_api import mapbox_api
#Importing img_transform and model module
#from ml_logic.model.model import get_model, get_model_destruction
#load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
# Dictionary of Warzones
warzone = {
    "-": ["-"],
    "Europe": ["-", "Spain", "Ukraine", "Turkey"],
    "Middle East": ["-", "Lebanon", "Syria", "Israel", "Iran", "Yemen"],
    "Africa": ["-", "Libya", "Sudan"]
}
region_centers = {
    "-": (0, 0),
    "Middle East": (29.0, 45.0),
    "Europe": (54.5, 15.3),
    "Asia": (34.0, 100.0),
    "Africa": (1.0, 20.0),
    "South America": (-15.6, -60.0)
}
# -----------------------------------------------------------------------------
# STREAMLIT CONFIG & GLOBAL STYLE
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        /* Reduce left/right padding */
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        /* Images & widgets take full width of their parent */
        .element-container img { width: 100% !important; height: auto !important; }
        /* Tighten vertical spacing between elements */
        .element-container { margin-bottom: 0rem !important; }
    </style>
    """,
    unsafe_allow_html=True)
# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("Choose Region")
region = st.sidebar.selectbox("List of Regions", list(region_centers.keys()))
st.sidebar.markdown("**Or use the search bar below:**")
search_query = st.sidebar.text_input("Search Location")
search_coords = None
if search_query:
    try:
        params = {"address": search_query, "key": google_api_key}
        response = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params)
        data = response.json()
        if data["status"] == "OK":
            result = data["results"][0]
            lat = result["geometry"]["location"]["lat"]
            lon = result["geometry"]["location"]["lng"]
            formatted_address = result["formatted_address"]
            search_coords = (lat, lon)
            st.sidebar.success(f"Found: {formatted_address}")
        else:
            st.sidebar.error(f"Geocoding failed: {data['status']}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
# -----------------------------------------------------------------------------
# MAP SECTION
# -----------------------------------------------------------------------------
# Determine map center
st.markdown("<h1 style='text-align: center;'>Orbital Response AI</h1>", unsafe_allow_html=True)
if search_coords:
    lat, lon = search_coords
    map_title = f"Map of: {search_query}"
elif region != "-":
    lat, lon = region_centers[region]
    map_title = f"Map of {region}"
else:
    st.warning("Please select a region or search for a location.")
    st.stop()
# Wrap map into centred column layout
left_spacer, map_col, right_spacer = st.columns([1, 2, 1])
with map_col:
    st.subheader(map_title)
    m = folium.Map(location=[lat, lon], zoom_start=6 if search_coords else 4)
    folium.LatLngPopup().add_to(m)
    map_data = st_folium(m, width=700, height=500)
# -----------------------------------------------------------------------------
# HANDLE MAP CLICK & SHOW SATELLITE IMAGES
# -----------------------------------------------------------------------------
with right_spacer:
    if map_data.get("last_clicked"):
        st.subheader(" ")
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        st.success(f"Clicked Location: Latitude: {clicked_lat:.5f}, Longitude: {clicked_lon:.5f}")
        st.session_state["lat"] = clicked_lat
        st.session_state["lon"] = clicked_lon
        if st.button("Fetch Satellite Images"):
            mapbox_api(clicked_lat, clicked_lon)
            google_api(clicked_lat, clicked_lon)
            st.session_state["images_ready"] = True
            st.session_state["predict_ready"] = False  # reset
        if st.session_state.get("images_ready"):
            pre_image_path = Path("images_masks/satellite_images/pre_disaster.png")
            post_image_path = Path("images_masks/satellite_images/post_disaster.png")
            if st.button("Predict"):
                st.session_state["predict_ready"] = True
        # Fetch satellite imagery from APIs
        image_dir = Path.cwd() / "images_masks" / "satellite_images"
        pre_image_path = image_dir / "pre_disaster.png"
        post_image_path = image_dir / "post_disaster.png"
        # Two columns right underneath the map with a SMALL gap to minimise whitespace
#-----------------------------------------------------------------------------
#___________________________YOLO Model____________________________#
#-----------------------------------------------------------------------------
if st.session_state.get("predict_ready"):
    model = YOLO("models/yolo_150ep_secondary.pt")
    results = model.predict(source=post_image_path, save=False, conf=0.25, verbose=False)
    post_img = Image.open(post_image_path).convert("RGB").resize((1024, 1024))
    base_rgba = post_img.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls)
            xyxy = box.xyxy[0].tolist()
            color = (255, 0, 0, 100) if cls == 0 else (0, 255, 0, 80)
            draw.rectangle(xyxy, fill=color)
    final_image = Image.alpha_composite(base_rgba, overlay)
    output_path = Path("images_masks/masks/yolo_output.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path)
    yolo_mask_path = "images_masks/masks/yolo_output.png"
    col1, col2 = st.columns([1, 1], gap="small")
    with col1:
        st.subheader("Pre vs Post Disaster")
        image_comparison(
            img1=str(pre_image_path),
            img2=str(post_image_path),
            label1="Pre-Disaster",
            label2="Post-Disaster",
            width=700)
    with col2:
        st.subheader("Post-Disaster Focus")
        image_comparison(
            img1=str(post_image_path),
            img2=str(yolo_mask_path),
            label1="Post-Disaster",
            label2="Mask",
            width=700)
    st.markdown("---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def load_models():
        model = get_model()
        model_destruction = get_model_destruction()
        model.load_state_dict(torch.load("models/Gaza_2nd_BestModel_70.pth", map_location=torch.device("cpu")))
        model_destruction.load_state_dict(torch.load("models/GazaData_v2_Destruction_Model_71.pth", map_location=torch.device("cpu")))
        return model.to(device), model_destruction.to(device)
    model, model_destruction = load_models()
    # Load and resize images
    pre_img = Image.open(pre_image_path).convert("RGB")
    post_img = Image.open(post_image_path).convert("RGB")
    resize = T.Resize((1024, 1024))
    pre_tensor = T.ToTensor()(resize(pre_img))
    post_tensor = T.ToTensor()(resize(post_img))
#-----------------------------------------------------------------------------
#___________________BUILDING DETECTION MODEL____________________________#
#-----------------------------------------------------------------------------
    input_tensor = torch.cat([pre_tensor, post_tensor], dim=0)
    normalize_build = T.Normalize(mean=[0.485]*6, std=[0.229]*6)
    input_tensor = normalize_build(input_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        pred = torch.sigmoid(model(input_tensor))
        bin_mask = (pred > 0.7).float().squeeze(0).squeeze(0)  # [H, W]
        mask_img = to_pil_image(bin_mask.cpu())
        os.makedirs("images_masks/masks", exist_ok=True)
        mask_img.save("images_masks/masks/building_mask.png")
    building_mask_path = "images_masks/masks/building_mask.png"
#----------------------------------------------------------------------------
#_____________________DESTRUCTION MODEL____________________________#
#-----------------------------------------------------------------------------

    building_img = resize(Image.open(building_mask_path))
    building_tensor = T.ToTensor()(building_img)
    building_tensor = (building_tensor < 0.5).float()
    input_tensor = torch.cat([pre_tensor, post_tensor, building_tensor], dim=0)
    normalize_destruction = T.Normalize(
        mean=[0.485, 0.456, 0.406]*2 + [0.5],
        std=[0.229, 0.224, 0.225]*2 + [0.25]
    )
    input_tensor = normalize_destruction(input_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        model_destruction.eval()
        output = model_destruction(input_tensor)  # [1, 1, H, W]
        probs = torch.sigmoid(output)[0, 0]  # [H, W]
        pred_mask = (probs < 0.95).float()
        destruction_mask_img = to_pil_image(pred_mask.cpu())
        destruction_mask_img.save("images_masks/masks/Destruction_mask.png")
    destruction_mask_path = "images_masks/masks/Destruction_mask.png"
    post_img = Image.open(post_image_path).convert("RGB").resize((1024, 1024))
    building_mask = Image.open(building_mask_path).convert("L").resize((1024, 1024))
    destruction_mask = Image.open(destruction_mask_path).convert("L").resize((1024, 1024))
    building_np = np.array(building_mask) > 127  # bool mask
    destruction_np = np.array(destruction_mask) > 127  # bool mask
    overlay = Image.new("RGBA", post_img.size, (0, 0, 0, 0))
    overlay_np = np.array(overlay)
    overlay_np[building_np] = [0, 255, 0, 100]  # verde con alpha
    overlay_np[(destruction_np) & (building_np)] = [255, 0, 0, 150]  # rojo con alpha mayor
    overlay = Image.fromarray(overlay_np, mode="RGBA")
    post_img_rgba = post_img.convert("RGBA")

    final_img = Image.alpha_composite(post_img_rgba, overlay)
    left_spacer, Final_image, right_spacer = st.columns([1, 3, 1])
    with Final_image:
        st.subheader("Our Model Prediction")
        st.image(final_img)
