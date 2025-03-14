
# Created by Neil Gordon January 2025
# Modified to include EXIF data extraction
# Refactored and Optimized March 2025 for Streamlit Cloud
# Complete, optimized main.py with progress bar, detailed status, and sequence numbering.



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

from typing import Dict, List, Optional, Tuple, Any
import math
import warnings

# Ignore PIL warnings from large image files - should not be a problem
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)



APP_CONFIG = {
    "debug": True,
    "thumbnail_size": (810, 810),
    "fullsize_limit": (3840, 2160),
    "absolute_max_size": 7680,
    "jpeg_quality": 100,
    "page_title": "PSNZ Image Entries Processor",
    "batch_size": 150  # Default batch size to avoid memory issues

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

        for data in exif_data_list:
            writer.writerow(data)

def create_zip(directory: Path, csv_filename: str, batch_num: int = None) -> Tuple[str, bytes]:
    """
    Create a zip file with all processed files and return its content as bytes.
    
    Args:
        directory: Directory containing files to zip
        csv_filename: Original CSV filename
        batch_num: Batch number (optional)
        
    Returns:
        Tuple of (zip_filename, zip_content_bytes)
    """
    # Create a zip filename based on the input CSV and batch number
    base_name = Path(csv_filename).stem
    zip_filename = f"{base_name}_images"
    if batch_num is not None:
        zip_filename += f"_batch{batch_num}"
    zip_filename += ".zip"
    
    # Use BytesIO to create the zip in memory instead of writing to disk
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                arcname = str(file_path.relative_to(directory))
                zipf.write(file_path, arcname)
    
    # Reset buffer position to the beginning
    zip_buffer.seek(0)
    
    # Return filename and content as bytes
    return zip_filename, zip_buffer.getvalue()

def process_images_in_batches(csv_file, limit_size=True, remove_exif=True, add_sequence=False, batch_size=150) -> List[Dict]:
    """
    Process images from CSV file in batches to manage memory usage.
    
    Args:
        csv_file: CSV file with image data
        limit_size: Whether to limit image size
        remove_exif: Whether to remove EXIF data
        add_sequence: Whether to add sequence numbers
        batch_size: Number of images to process in each batch
        
    Returns:
        List of dictionaries with batch information
    """
    # Check if we already have results in session state
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        return st.session_state.batch_results
    
    # Read and validate CSV file
    df = pd.read_csv(csv_file)
    errors = validate_csv(df)
    if errors:
        for error in errors:
            st.error(error)
        return []
    
    # Initialize sequence dictionary if needed
    sequence_dict = {} if add_sequence else None

    # Calculate number of batches
    total_images = len(df)
    num_batches = math.ceil(total_images / batch_size)
    st.write(f"Processing {total_images} images in {num_batches} batches (max {batch_size} images per batch)")
    
    batch_results = []
    
    # Process each batch
    for batch_index in range(num_batches):
        # Create a new temporary directory for this batch
        temp_dir = Path(tempfile.mkdtemp())
        
        # Set up directory structure
        fullsize_dir, thumbnail_dir, exif_csv_path = setup_directories(temp_dir, limit_size, remove_exif)
        
        # Calculate start and end indices for this batch
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_df = df.iloc[start_idx:end_idx]
        
        batch_exif_data = []
        
        # Set up progress tracking for this batch
        st.write(f"\nProcessing Batch {batch_index + 1}/{num_batches} (images {start_idx + 1} to {end_idx})")
        progress_bar = st.progress(0)

        # Process each image in this batch
        for i, (index, row) in enumerate(batch_df.iterrows()):
            result = fetch_and_process_image(
                row, fullsize_dir, thumbnail_dir, limit_size, remove_exif, sequence_dict
            )
            
            if result:
                batch_exif_data.append(result['exif_info'])
                st.write(f"Processed {start_idx + i + 1}/{total_images}: {result['exif_info']['FileName']} "
                         f"({result['original_size']}, {result['status']})")
            
            # Update progress
            progress_bar.progress((i + 1) / len(batch_df))

        # Write EXIF data to CSV for this batch
        write_exif_data(exif_csv_path, batch_exif_data)
        
        # Create zip file for this batch
        try:
            batch_zip_filename, batch_zip_bytes = create_zip(temp_dir, csv_file.name, batch_index + 1)
            
            # Add batch result
            batch_results.append({
                'batch_num': batch_index + 1,
                'zip_filename': batch_zip_filename,
                'zip_bytes': batch_zip_bytes,
                'num_images': len(batch_df),
                'start_idx': start_idx,
                'end_idx': end_idx - 1
            })
            
            # Clean up the temp directory
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            st.error(f"Error creating zip for batch {batch_index + 1}: {str(e)}")
            if APP_CONFIG["debug"]:
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
        
        # Force garbage collection
        gc.collect()
    
    # Store results in session state
    st.session_state.batch_results = batch_results
    return batch_results

def main() -> None:
    """Main application function."""
    st.title(APP_CONFIG["page_title"])

    # Initialize session state for batch results if it doesn't exist
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    
    # Initialize session state for processing flag
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    st.header("CSV File Upload")
    st.write(
        "Please upload the CSV file with columns 'File Name' and 'Image: URL'. "
        "All images will be fetched and padded with leading zeros. "
        "We'll create thumbnails and provide zip files for download. "
        "EXIF data (dimensions, creation date, original capture date) will be extracted "
        "and included in a separate CSV file in each zip."
    )

    # User options
    limit_size = st.checkbox("Limit image size to 3840x2160 pixels", value=True)
    remove_exif = st.checkbox("Remove EXIF metadata from images", value=True)
    add_sequence = st.checkbox("Add sequence # after ID for multiple images from the same ID", value=False)
    
    # Batch size option
    col1, col2 = st.columns([3, 1])
    with col1:
        batch_size = st.slider("Batch size (max images per download)", 
                             min_value=10, max_value=500, value=APP_CONFIG["batch_size"], step=10)
    with col2:
        st.write("")
        st.write("")
        if st.button("Reset to Default"):
            batch_size = APP_CONFIG["batch_size"]
            st.rerun()

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Process button
        process_button = st.button("Process Images")
        
        # Check if we should start processing
        start_processing = process_button or st.session_state.processing_complete
        
        if start_processing:
            # Set the processing flag
            st.session_state.processing_complete = True
            
            # Show processing spinner only if we don't have results yet
            if not st.session_state.batch_results:
                with st.spinner("Processing images in batches..."):
                    batch_results = process_images_in_batches(
                        uploaded_file, limit_size, remove_exif, add_sequence, batch_size
                    )
            else:
                batch_results = st.session_state.batch_results
            
            if batch_results:
                st.success(f"✅ Processing complete! {len(batch_results)} batch(es) ready for download.")
                
                # Display all batches
                st.subheader("Download Processed Batches")
                st.write("Each batch can be downloaded independently.")
                
                # Create columns for download buttons (2 buttons per row)
                cols_per_row = 2
                for i in range(0, len(batch_results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    # Add buttons to columns
                    for j in range(cols_per_row):
                        if i + j < len(batch_results):
                            batch = batch_results[i + j]
                            with cols[j]:
                                # Create download button with unique key that includes batch number
                                st.download_button(
                                    label=(f"Batch {batch['batch_num']} "
                                          f"(Images {batch['start_idx'] + 1}-{batch['end_idx'] + 1})"),
                                    data=batch['zip_bytes'],
                                    file_name=batch['zip_filename'],
                                    mime="application/zip",
                                    key=f"download_batch_{batch['batch_num']}",
                                    use_container_width=True
                                )
                
                # Add a note about downloading all batches
                st.info("📋 Remember to download all batches to get your complete processed dataset.")
                
                # Add reset button to clear results and start over
                if st.button("Reset Processing"):
                    st.session_state.batch_results = []
                    st.session_state.processing_complete = False
                    st.rerun()
            else:
                st.error("Failed to process images. Please check the CSV file format and try again.")


if __name__ == "__main__":
    main()
