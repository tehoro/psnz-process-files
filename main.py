# Here's the complete, optimized main.py file for your Streamlit application.

# Created by Neil Gordon January 2025
# Modified to include EXIF data extraction
# Refactored and Optimized March 2025 for Streamlit Cloud
# Complete, optimized main.py with progress bar, detailed status, and sequence numbering.

# Complete, optimized main.py with correct metadata columns in EXIF CSV.

# Complete, optimized main.py with EXIF date handling fixes.

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

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 100_000_000

APP_CONFIG = {
    "debug": True,
    "thumbnail_size": (810, 810),
    "fullsize_limit": (3840, 2160),
    "absolute_max_size": 7680,
    "jpeg_quality": 100,
    "page_title": "PSNZ Image Entries Processor",
    "batch_size": 3,
    "zip_batch_size": 150,
}

st.set_page_config(page_title=APP_CONFIG["page_title"], layout="wide")

@st.cache_resource
def get_resource_limiter():
    return threading.Semaphore(1)

def log_debug(message):
    if APP_CONFIG["debug"]:
        st.write(message)

def get_exif_data(img):
    exif_data = {'Width': img.width, 'Height': img.height, 'DateTimeCreated': None, 'DateTimeOriginal': None}
    try:
        if hasattr(img, '_getexif') and img._getexif():
            exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in img._getexif().items()}
            exif_data['DateTimeCreated'] = exif.get('DateTime', None)
            exif_data['DateTimeOriginal'] = exif.get('DateTimeOriginal', None)
    except Exception as e:
        log_debug(f"EXIF extraction error: {e}")
    return exif_data

def pad_id_with_sequence(filename, sequence_dict):
    match = re.match(r"(\d+)(.*)", filename)
    if match:
        id_num, rest = match.groups()
        seq_num = sequence_dict.get(id_num, 0) + 1
        sequence_dict[id_num] = seq_num
        return f"{id_num}-{seq_num}{rest}"
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
    original_filename = row['File Name']
    filename = re.sub(r'[\\/:*?"<>|]', '_', original_filename)
    if sequence_dict is not None:
        filename = pad_id_with_sequence(filename, sequence_dict)

    filepath = fullsize_dir / filename
    filepath_small = thumbnail_dir / filename

    try:
        log_debug(f"Fetching image: {row['Image: URL']}")
        response = requests.get(row['Image: URL'], timeout=30, stream=True)
        image_data = BytesIO(response.content)
        response.close()

        with Image.open(image_data) as img:
            exif_data = get_exif_data(img)
            original_size = f"{img.width}x{img.height}"
            processed_img = img.convert('RGB') if img.mode in ('RGBA', 'P') else img.copy()

            resized = False
            if max(processed_img.size) > APP_CONFIG["absolute_max_size"]:
                processed_img.thumbnail((APP_CONFIG["absolute_max_size"], APP_CONFIG["absolute_max_size"]))
                resized = True
            elif limit_size:
                processed_img.thumbnail(APP_CONFIG["fullsize_limit"])
                resized = True

            processed_img.save(filepath, "JPEG", quality=APP_CONFIG["jpeg_quality"])

            thumbnail = processed_img.copy()
            thumbnail.thumbnail(APP_CONFIG["thumbnail_size"])
            thumbnail.save(filepath_small, "JPEG")

            processed_img.close()
            thumbnail.close()
            del image_data

        gc.collect()
        return {
            'FileName': filename,
            'OriginalFileName': original_filename,
            'OriginalSize': original_size,
            'Resized': "Yes" if resized else "No",
            'DateTimeCreated': exif_data.get('DateTimeCreated', 'N/A'),
            'DateTimeOriginal': exif_data.get('DateTimeOriginal', 'N/A'),
            **exif_data
        }
    except Exception as e:
        st.error(f"Error processing image {filename}: {e}")
        return None

def write_exif_data(exif_csv_path, exif_data_list):
    with open(exif_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['FileName', 'OriginalFileName', 'OriginalSize', 'Resized', 'DateTimeCreated', 'DateTimeOriginal', 'Width', 'Height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exif_data_list)

def main():
    st.title(APP_CONFIG["page_title"])

    limit_size = st.checkbox("Limit image size to 3840x2160 pixels", True)
    remove_exif = st.checkbox("Remove EXIF metadata from images", True)
    add_sequence = st.checkbox("Add sequence # for multiple images per ID", False)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    resource_limiter = get_resource_limiter()

    if uploaded_file and st.button("Process Images"):
        if not resource_limiter.acquire(blocking=False):
            st.error("Another process is running. Please wait.")
            return

        try:
            with st.spinner("Processing images..."):
                df = pd.read_csv(uploaded_file)
                errors = validate_csv(df)
                if errors:
                    for error in errors:
                        st.error(error)
                    return

                sequence_dict = {} if add_sequence else None
                total_images = len(df)
                progress_bar = st.progress(0)
                status_area = st.empty()
                status_messages = []

                temp_dir = Path(tempfile.mkdtemp())
                fullsize_dir, thumbnail_dir, exif_csv_path = setup_directories(temp_dir, limit_size, remove_exif)
                exif_data_list = []

                for idx, (_, row) in enumerate(df.iterrows(), 1):
                    result = fetch_and_process_image(row, fullsize_dir, thumbnail_dir, limit_size, remove_exif, sequence_dict)
                    if result:
                        exif_data_list.append(result)
                        status_messages.append(f"Processed {idx}/{total_images}: {result['FileName']} (Original: {result['OriginalSize']}, Resized: {result['Resized']})")
                        status_area.text("\n".join(status_messages[-10:]))
                    progress_bar.progress(idx / total_images)

                write_exif_data(exif_csv_path, exif_data_list)
                st.success("Processing complete! EXIF metadata saved.")
        finally:
            resource_limiter.release()

if __name__ == "__main__":
    main()
