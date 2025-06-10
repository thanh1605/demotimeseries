import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import io
from PIL import Image
from datetime import datetime, timedelta
from meteostat import Stations, Daily
import h5py
import os

# --- Cấu hình ---
st.set_page_config("Live Landfall Predictor", layout="wide")
st.title("🌪️ Dự đoán Landfall với Dữ liệu Thời Gian Thực")

# Định nghĩa mapping name → file path
MODEL_DIR = "model"
model_files = {
    "LSTM+CNN": "lstm_cnn_landfall_model.h5",
    "RNN":      "rnn_landfall_model.h5",
    "TFT":      "tft_landfall_model.h5",
}


def load_models():
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        st.write(f"🔍 Checking {name!r} at {path!r}: exists? {os.path.exists(path)}")
        try:
            models[name] = tf.keras.models.load_model(path)
            st.write(f"✅ Loaded {name!r}")
        except Exception as e:
            st.error(f"❌ Failed to load {name!r}: {type(e).__name__}: {e}")
            # Nếu model quan trọng, có thể st.stop() để dừng app
    return models

models = load_models()

# Sidebar chọn model
st.sidebar.header("Chọn Model")
model_name = st.sidebar.selectbox("Bạn muốn dùng model nào?", list(models.keys()))

# Lấy đúng model
model = models[model_name]


# --- Sidebar: chọn station & thời điểm ---
st.sidebar.header("Cấu hình nguồn dữ liệu")
station_id = st.sidebar.text_input("Station ID (Meteostat)", value="10637")  
# ví dụ 10637 = Tokyo
date = st.sidebar.date_input("Chọn ngày", datetime.utcnow().date() - timedelta(days=1))
fetch = st.sidebar.button("🔄 Lấy dữ liệu và Dự đoán")

def fetch_station_data(station_id, date):
    """Trả về DataFrame daily cho station và ngày đã chọn."""
    stations = Stations()
    station = stations.id(station_id).fetch(1)
    df = Daily(station, start=date, end=date).fetch()
    return df

def fetch_goes_image(band="M3", dt=None):
    """
    Lấy ảnh GOES từ AWS Public Bucket.
    Band M3 ~ hồng ngoại gần, dt: datetime UTC.
    """
    # Ví dụ URL template (AWS): 
    # https://noaa-goes18.s3.amazonaws.com/ABI/FDCC/2025/175/00/OR_ABI-L2-FDCC-M3_G18_s20251750000_e20251751500_c20251751503.nc
    # Ở đây ta sử dụng service demo lấy PNG nhỏ (GOES-16 GeoColor)
    url = (
        "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/2025/{}{:02d}{:02d}/"
        "GOES16-FD-GEOCOLOR-5000x3000-{}.jpg"
    ).format(dt.year, dt.timetuple().tm_yday, dt.hour, dt.strftime("%Y%j%H%M"))
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError("Không tải được ảnh GOES")
    img = Image.open(io.BytesIO(resp.content))
    return img

if fetch:
    # 1. Lấy số liệu trạm
    try:
        df_station = fetch_station_data(station_id, date)
        st.subheader("📊 Dữ liệu Mặt đất:")
        st.dataframe(df_station)
        # Giả sử model cần 4 biến: t2m, prcp, wdir, wspd
        features = df_station[["tavg","prcp","wdir","wspd"]].iloc[0].fillna(0).to_numpy()
    except Exception as e:
        st.error(f"Không lấy được dữ liệu trạm: {e}")
        st.stop()

    # 2. Lấy ảnh vệ tinh/radar
    try:
        dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        img = fetch_goes_image(dt=dt)
        st.subheader("🛰️ Ảnh GOES-16 GeoColor (UTC {})".format(dt.strftime("%Y-%m-%d %H:%M")))
        st.image(img, use_column_width=True)
        # tiền xử lý giống app trước: resize, normalize
        arr = np.array(img.resize((224,224))) / 255.0
        img_input = arr[np.newaxis,...]
    except Exception as e:
        st.error(f"Không tải được ảnh vệ tinh: {e}")
        st.stop()

    # 3. Chạy model
    num_input = features.reshape(1,-1).astype(np.float32)
    pred = model.predict([img_input, num_input])[0,0]
    st.success(f"🎯 Xác suất đi vào đất liền ~ **{pred*100:.1f}%**")
