import streamlit as st
import base64
from pathlib import Path

# Page config
st.set_page_config(page_title="Orbital Response - Further Development", layout="wide")

# Load and encode banner image
banner_path = Path("presentation_images/gaza_banner.png")
if banner_path.exists():
    with open(banner_path, "rb") as img_file:
        b64_encoded = base64.b64encode(img_file.read()).decode()
    banner_url = f"data:image/png;base64,{b64_encoded}"
else:
    st.warning("🚫 Gaza banner image not found.")
    banner_url = ""

# Add thin banner
st.markdown(f"""
    <style>
    .thin-banner {{
        position: relative;
        width: 100vw;
        left: 50%;
        margin-left: -50vw;
        background-image: url('{banner_url}');
        background-size: cover;
        background-position: center;
        height: 160px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }}
    .thin-banner::before {{
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.4);
        z-index: 0;
    }}
    .thin-banner h1 {{
        z-index: 1;
        color: white !important;
        font-size: 3.5rem;
        margin: 0;
        position: relative;
    }}
    .custom-section h3 {{
        font-size: 1.8rem;
        color: #004080;
        margin-bottom: 1rem;
    }}
    .custom-section ul {{
        font-size: 1.2rem;
        line-height: 2rem;
    }}
    </style>

    <div class="thin-banner">
        <h1>Orbital Response AI</h1>
    </div>
""", unsafe_allow_html=True)

# Title and content
st.title("Further Development")
col1, col2 = st.columns([1,1.4])

with col1:
    st.markdown("""
    <div class="custom-section">
        <h3>Limitations</h3>
        <ul>
            <li><b>Data:</b> 150 labelled images insufficient to achieve desired model accuracy</li>
            <li><b>Computational Capacity:</b> ~6 hours of model training for 150 images (with GPU)</li>
            <li><b>Project Time:</b> Limited fine-tune opportunity for U-Net model</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-section">
        <h3>Next Steps</h3>
        <ul>
            <li><b>Manual Labelling:</b> Min. 1000 manually labelled images, focused on Gaza (1/4 of land mass)</li>
            <li><b>Generalisation:</b> Train the model on images from additional conflict zones, starting with Ukraine, to broaden the scope of use</li>
            <li><b>Translation into Aid:</b> Use building damage as a proxy to estimate affected population (via density mapping), then quantify needed humanitarian response: <b>shelter</b>, <b>food & water</b>, <b>medical supplies</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
