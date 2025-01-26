# Created by Neil Gordon January 2025

import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import zipfile
import tempfile
import shutil
import re
import math
import psutil
import gc

st.set_page_config(page_title="PSNZ Image Entries Processor")

DEBUG = False  # Set to True to enable detailed debug output

def pad_id_with_sequence(filename, id_length, sequence_dict=None):
    match = re.match(r'^(\d+)(.*)$', filename)
    if match:
        id_num, rest = match.groups()

        if sequence_dict is None:
            return f"{id_num}{rest}"

        if id_num not in sequence_dict:
            sequence_dict[id_num] = 1
        sequence_num = sequence_dict[id_num]
        sequence_dict[id_num] += 1

        name_parts = rest.rsplit('.', 1)
        if len(name_parts) == 2:
            clean_title = name_parts[0].lstrip('- ')
            return f"{id_num}-{sequence_num} {clean_title}.{name_parts[1]}"
        clean_rest = rest.lstrip('- ')
        return f"{id_num}-{sequence_num} {clean_rest}"
    return filename

def log_memory_usage(message):
    if DEBUG:
        process = psutil.Process()
        memory_info = process.memory_info()
        st.write(f"Memory usage at {message}: {memory_info.rss / 1024 / 1024:.2f} MB")

def process_images(csv_file, limit_size=True, remove_exif=True, add_sequence=False):
    temp_dir = tempfile.mkdtemp()

    fullsize_folder_name = "4K-size" if limit_size else "submitted-size"
    if remove_exif:
        fullsize_folder_name += "-exifremoved"
        thumbnail_folder_name = "thumbnails-exifremoved"
    else:
        thumbnail_folder_name = "thumbnails"

    fullsize_dir = os.path.join(temp_dir, fullsize_folder_name)
    thumbnail_dir = os.path.join(temp_dir, thumbnail_folder_name)
    os.makedirs(fullsize_dir, exist_ok=True)
    os.makedirs(thumbnail_dir, exist_ok=True)

    try:
        if DEBUG:
            st.write("Reading CSV file...")
        df = pd.read_csv(csv_file)
        log_memory_usage("after reading CSV")

        required_columns = ['File Name', 'Image: URL']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None

        sequence_dict = {} if add_sequence else None

        total_images = len(df)
        st.write(f"Total images to process: {total_images}")
        progress_bar = st.progress(0)

        for index, row in df.iterrows():
            # Replace invalid Windows filename characters with underscore
            filename = re.sub(r'[\\/:*?"<>|]', '_', pad_id_with_sequence(row['File Name'], None, sequence_dict))
            filepath = os.path.join(fullsize_dir, filename)
            filepath_small = os.path.join(thumbnail_dir, filename)
            try:
                response = requests.get(row['Image: URL'], timeout=10)
                img = Image.open(BytesIO(response.content))
                log_memory_usage(f"after opening {filename}")

                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                original_size = f"{img.width}x{img.height}"
                resized = False
                if limit_size and (img.width > 3840 or img.height > 2160):
                    img.thumbnail((3840, 2160))
                    resized = True

                if remove_exif:
                    new_img = Image.new('RGB', img.size)
                    new_img.paste(img)
                    img = new_img
                    log_memory_usage(f"after EXIF removal for {filename}")

                img.save(filepath, "JPEG", quality=100)

                img_small = img.copy()
                img_small.thumbnail((810, 810))
                img_small.save(filepath_small, "JPEG")

                del img_small
                del img
                gc.collect()
                log_memory_usage(f"after cleanup for {filename}")

                status = "resized" if resized else "original size"
                st.write(f"Processed {index + 1}/{total_images}: {filename} ({original_size}, {status})")
                progress_bar.progress((index + 1) / total_images)

            except Exception as e:
                st.error(f"Error processing image for file {filename}: {str(e)}")
                if DEBUG:
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")

    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        if DEBUG:
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        return None

    return temp_dir

def create_zip(directory, csv_filename):
    zip_filename = f"{os.path.splitext(csv_filename)[0]}_images.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory)
                zipf.write(file_path, arcname)
    return zip_path

def main():
    st.title("PSNZ Image Entries Processor")

    # Clean up previous zip files
    for file in os.listdir(tempfile.gettempdir()):
        if file.endswith('_images.zip'):
            os.remove(os.path.join(tempfile.gettempdir(), file))

    st.header("CSV File Upload")
    st.write(
        "Please upload the CSV file with columns 'File Name' and 'Image: URL'. "
        "All images will be fetched and padded with leading zeros. "
        "We'll create thumbnails and provide a zip for download."
    )

    limit_size = st.checkbox("Limit image size to 3840x2160 pixels", value=True)
    remove_exif = st.checkbox("Remove EXIF metadata from images", value=True)
    add_sequence = st.checkbox("Add sequence # after ID for multiple images from the same ID", value=False)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        if st.button("Process Images"):
            with st.spinner("Processing images..."):
                temp_dir = process_images(uploaded_file, limit_size, remove_exif, add_sequence)
                if temp_dir:
                    zip_path = create_zip(temp_dir, uploaded_file.name)
                    shutil.rmtree(temp_dir)  # Clean up the temp directory

                    # Read the final zip file into memory before streaming it to download_button
                    with open(zip_path, "rb") as f:
                        zip_bytes = f.read()

                    zip_filename = os.path.basename(zip_path)
                    st.download_button(
                        label="Download Processed Images (Full-size and Thumbnails)",
                        data=zip_bytes,
                        file_name=zip_filename,
                        mime="application/zip"
                    )
                else:
                    st.error("Failed to process images. Please check the CSV file format.")

if __name__ == "__main__":
    main()
