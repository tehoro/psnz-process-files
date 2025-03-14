# Here's the complete, optimized main.py file for your Streamlit application.

# Created by Neil Gordon January 2025
# Modified to include EXIF data extraction
# Refactored and Optimized March 2025 for Streamlit Cloud

import streamlit as st
import pandas as pd
import requests
from PIL import Image, ExifTags
from io import BytesIO
import os
import zipfile
import tempfile
import shutil
import re
import gc
import csv
from pathlib import Path
import threading
import warnings

# Disable PIL DecompressionBombWarning
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Safer decompression limit (~100 MP)
Image.MAX_IMAGE_PIXELS = 100_000_000

# App configuration
APP_CONFIG = {
    "debug": False,
    "thumbnail_size": (810, 810),
    "fullsize_limit": (3840, 2160),
    "absolute_max_size": 7680,
    "jpeg_quality": 100,
    "page_title": "PSNZ Image Entries Processor",
    "batch_size": 5,
}

st.set_page_config(page_title=APP_CONFIG["page_title"], layout="wide")

@st.cache_resource
def get_resource_limiter():
    return threading.Semaphore(1)

def get_exif_data(img):
    exif_data = {'Width': img.width, 'Height': img.height}
    try:
        if hasattr(img, '_getexif'):
            exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in img._getexif().items()}
            exif_data['DateTimeOriginal'] = exif.get('DateTimeOriginal', None)
    except:
        pass
    return exif_data

def pad_id_with_sequence(filename, _, sequence_dict):
    return filename

def setup_directories(temp_dir, limit_size, remove_exif):
    fullsize_dir = temp_dir / ("4K-size" if limit_size else "submitted-size")
    thumbnail_dir = temp_dir / "thumbnails"
    fullsize_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    return fullsize_dir, thumbnail_dir, temp_dir / "metadata.csv"

def validate_csv(df):
    errors = []
    for col in ['File Name', 'Image: URL']:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    return errors

def fetch_and_process_image(row, fullsize_dir, thumbnail_dir, limit_size, remove_exif, sequence_dict=None):
    filename = re.sub(r'[\\/:*?"<>|]', '_', row['File Name'])
    filepath = fullsize_dir / filename
    filepath_small = thumbnail_dir / filename

    try:
        response = requests.get(row['Image: URL'], timeout=30, stream=True)
        image_data = BytesIO(response.content)
        response.close()

        with Image.open(image_data) as img:
            exif_data = get_exif_data(img)
            processed_img = img.convert('RGB') if img.mode in ('RGBA', 'P') else img.copy()

            if max(processed_img.size) > APP_CONFIG["absolute_max_size"]:
                processed_img.thumbnail((APP_CONFIG["absolute_max_size"], APP_CONFIG["absolute_max_size"]))
            elif limit_size:
                processed_img.thumbnail(APP_CONFIG["fullsize_limit"])

            processed_img.save(filepath, "JPEG", quality=APP_CONFIG["jpeg_quality"])

            thumbnail = processed_img.copy()
            thumbnail.thumbnail(APP_CONFIG["thumbnail_size"])
            thumbnail.save(filepath_small, "JPEG")

            processed_img.close()
            thumbnail.close()
            del image_data

        gc.collect()
        return {'FileName': filename, **exif_data}

    except Exception as e:
        st.error(f"Error processing image {filename}: {e}")
        return None

def write_exif_data(exif_csv_path, exif_data_list):
    with open(exif_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=exif_data_list[0].keys())
        writer.writeheader()
        writer.writerows(exif_data_list)

def create_zip(directory, csv_filename):
    zip_filename = f"{Path(csv_filename).stem}_images.zip"
    temp_zip_path = Path(tempfile.gettempdir()) / zip_filename

    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(directory))

    return temp_zip_path

def main():
    st.title(APP_CONFIG["page_title"])

    limit_size = st.checkbox("Limit image size to 3840x2160 pixels", True)
    remove_exif = st.checkbox("Remove EXIF metadata from images", True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    resource_limiter = get_resource_limiter()

    if uploaded_file and st.button("Process Images"):
        if not resource_limiter.acquire(blocking=False):
            st.error("Another process is running. Please wait.")
            return

        try:
            with st.spinner("Processing images..."):
                temp_dir = Path(tempfile.mkdtemp())
                fullsize_dir, thumbnail_dir, exif_csv_path = setup_directories(temp_dir, limit_size, remove_exif)

                df = pd.read_csv(uploaded_file)
                errors = validate_csv(df)
                if errors:
                    for error in errors:
                        st.error(error)
                    return

                exif_data_list = []

                for start_idx in range(0, len(df), APP_CONFIG["batch_size"]):
                    batch_df = df.iloc[start_idx:start_idx+APP_CONFIG["batch_size"]]
                    for _, row in batch_df.iterrows():
                        result = fetch_and_process_image(row, fullsize_dir, thumbnail_dir, limit_size, remove_exif)
                        if result:
                            exif_data_list.append(result)
                    gc.collect()

                write_exif_data(exif_csv_path, exif_data_list)

                zip_path = create_zip(temp_dir, uploaded_file.name)
                with open(zip_path, 'rb') as zip_file:
                    st.download_button("Download ZIP", data=zip_file, file_name=zip_path.name, mime="application/zip")

                shutil.rmtree(temp_dir)
                zip_path.unlink(missing_ok=True)
                gc.collect()

        finally:
            resource_limiter.release()

if __name__ == "__main__":
    main()
