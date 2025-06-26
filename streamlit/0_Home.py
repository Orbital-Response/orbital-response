import streamlit as st
import base64
from PIL import Image
import os
from pathlib import Path


st.set_page_config(
    page_title="Orbital Response - Home",
    page_icon="🌍",
    layout="wide"
)

banner_path = Path("presentation_images/gaza_banner.png")  # update if in a different folder

if banner_path.exists():
    with open(banner_path, "rb") as img_file:
        b64_encoded = base64.b64encode(img_file.read()).decode()
    banner_url = f"data:image/jpeg;base64,{b64_encoded}"
else:
    st.warning("🚫 Gaza banner image not found.")
    banner_url = ""

st.markdown(f"""
    <style>
    .full-width-banner {{
        position: relative;
        width: 100vw;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        background-image: url('{banner_url}');
        background-size: cover;
        background-position: center;
        height: 360px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: white;
    }}

    .full-width-banner::before {{
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.4);
        z-index: 0;
    }}

    .banner-text {{
        z-index: 1;
        padding: 0 2rem;
        max-width: 900px;
    }}

    .banner-text h1 {{
        font-size: 4.2rem;
        margin-bottom: 0.3rem;
    }}

    .banner-text h3 {{
        font-size: 1.4rem;
        font-weight: normal;
        margin-top: 0;
    }}
    </style>

    <div class="full-width-banner">
        <div class="banner-text">
            <h1>Orbital Response</h1>
            <h3>Optimising Humanitarian Aid Distribution with AI</h3>
        </div>
    </div>
""", unsafe_allow_html=True)





# Add mission statement content
st.markdown("---")
st.markdown("### **Our Mission**")
st.markdown(
"""
**Harness the power of CNNs and high-frequency satellite imagery to optimise humanitarian aid distribution.**

High-resolution, high-frequency satellite imagery is an emerging and transformative resource. Today, we have access to imagery with resolutions below 0.5 meters per pixel and revisit intervals of less than five days. When combined with advances in artificial intelligence — particularly Convolutional Neural Networks (CNNs) — this unlocks new capabilities.

Given today’s global socio-political climate, we chose to focus on humanitarian aid in conflict zones. Satellite imagery is playing an increasingly vital role in these contexts. Modern offensives often involve drone strikes, and their consequences — especially building destruction — are visible from space.

By mapping the distribution and severity of this destruction, we can support a more optimized, data-driven approach to humanitarian aid delivery in affected regions.
"""
)
st.markdown("---")
st.markdown("### Project Repository")

st.markdown(
    """
    You can explore the full codebase, datasets, and model architecture on our [https://github.com/benshaw0/orbital-response](#).
    """,
    unsafe_allow_html=True
)
st.write("")

st.markdown("---")
st.markdown("<h3 style='text-align:center; color:#004080;'>Meet The Team</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

team_profiles = [
    {"name": "Ben Shaw", "url": "https://www.linkedin.com/in/bencshaw/", "img": "presentation_images/Ben.png"},
    {"name": "Felix Pless", "url": "https://www.linkedin.com/in/felixpless/", "img": "presentation_images/Felix.jpg"},
    {"name": "Christian Miro", "url": "https://www.linkedin.com/in/christianmiro/", "img": "presentation_images/Christian.jpeg"},
]

def make_avatar_tag(path: Path, width: int = 120) -> str:
    """Return a base64-embedded circular avatar suitable for HTML."""
    img_format = "png" if path.suffix.lower() == ".png" else "jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<img src="data:image/{img_format};base64,{b64}" '
        f'style="width:{width}px;height:{width}px;border-radius:50%;display:block;margin:auto;" />'
    )

cols = st.columns(len(team_profiles))
for col, member in zip(cols, team_profiles):
    p = Path(member["img"])
    with col:
        if p.exists():
            avatar_html = make_avatar_tag(p, width=140)
            st.markdown(
                f"""
                <a href="{member["url"]}" target="_blank">
                    {avatar_html}
                    <p style="text-align:center; margin-top:0.5rem;">{member["name"]}</p>
                </a>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<p style="text-align:center;"><a href="{member["url"]}">{member["name"]}</a></p>', unsafe_allow_html=True)
