FROM python:3.10.6-buster
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY models models
COPY ml_logic ml_logic
COPY images_masks images_masks
COPY goog_static_map_api.py goog_static_map_api.py
COPY mapbox_api.py mapbox_api.py
COPY 0_Home.py 0_Home.py
COPY pages pages
COPY presentation_images presentation_images

RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    libgdal-dev \
    gdal-bin \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
#CMD streamlit run app_ui_ben.py --port $PORT
CMD ["sh", "-c", "streamlit run 0_Home.py --server.address 0.0.0.0 --server.port $PORT"]
#CMD streamlit run app_ui_ben.py --port $PORT
#CMD ["sh", "-c", "streamlit run app_ui_ben.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]
