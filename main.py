# Created by Neil Gordon January 2025
# Modified to include EXIF data extraction
# Refactored March 2025
# Fix problem with very large images

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
import psutil
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Disable PIL DecompressionBombWarning
# This is safe in our context since we're processing trusted images
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Set a higher decompression bomb limit if needed
Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely, use with caution

# App configuration
APP_CONFIG = {
    "debug": False,
    "thumbnail_size": (810, 810),
    "fullsize_limit": (3840, 2160),
    "jpeg_quality": 100,
    "page_title": "PSNZ Image Entries Processor"
}

# Setup page configuration
st.set_page_config(page_title=APP_CONFIG["page_title"])

def log_memory_usage(message: str) -> None:
    """Log current memory usage if debug is enabled."""
    if APP_CONFIG["debug"]:
        process = psutil.Process()
        memory_info = process.memory_info()
        st.write(f"Memory usage at {message}: {memory_info.rss / 1024 / 1024:.2f} MB")

def pad_id_with_sequence(filename: str, id_length: Optional[int] = None, 
                         sequence_dict: Optional[Dict[str, int]] = None) -> str:
    """
    Pad ID with a sequence number for multiple entries from the same ID.
    
    Args:
        filename: Original filename
        id_length: Length to pad the ID to
        sequence_dict: Dictionary to track sequence numbers by ID
        
    Returns:
        Modified filename with sequence number if appropriate
    """
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

def get_exif_data(img: Image.Image) -> Dict[str, Any]:
    """
    Extract EXIF data from image.
    
    Args:
        img: PIL Image object
        
    Returns:
        Dictionary containing extracted EXIF data
    """
    exif_data = {
        'DateTimeCreated': None,
        'DateTimeOriginal': None,
        'Width': img.width,
        'Height': img.height
    }
    
    try:
        # Get EXIF data if available
        exif = {ExifTags.TAGS.get(tag, tag): value 
                for tag, value in img._getexif().items()} if hasattr(img, '_getexif') and img._getexif() else {}
        
        # Date/Time when the image was created/modified
        if 'DateTime' in exif:
            exif_data['DateTimeCreated'] = exif['DateTime']
        
        # Original Date/Time when the photo was taken
        if 'DateTimeOriginal' in exif:
            exif_data['DateTimeOriginal'] = exif['DateTimeOriginal']
        elif 'DateTimeDigitized' in exif:
            exif_data['DateTimeOriginal'] = exif['DateTimeDigitized']
    except Exception as e:
        if APP_CONFIG["debug"]:
            st.write(f"Error extracting EXIF data: {str(e)}")
    
    return exif_data

def setup_directories(temp_dir: Path, limit_size: bool, remove_exif: bool) -> Tuple[Path, Path, Path]:
    """
    Create necessary directory structure for image processing.
    
    Args:
        temp_dir: Base temporary directory
        limit_size: Whether to limit image size
        remove_exif: Whether to remove EXIF data
        
    Returns:
        Tuple of (fullsize_dir, thumbnail_dir, exif_csv_path)
    """
    # Determine folder names based on options
    fullsize_folder_name = "4K-size" if limit_size else "submitted-size"
    if remove_exif:
        fullsize_folder_name += "-exifremoved"
        thumbnail_folder_name = "thumbnails-exifremoved"
    else:
        thumbnail_folder_name = "thumbnails"

    # Create directories
    fullsize_dir = temp_dir / fullsize_folder_name
    thumbnail_dir = temp_dir / thumbnail_folder_name
    fullsize_dir.mkdir(exist_ok=True)
    thumbnail_dir.mkdir(exist_ok=True)
    
    # Path for EXIF data CSV
    exif_csv_path = temp_dir / "image_metadata.csv"
    
    return fullsize_dir, thumbnail_dir, exif_csv_path

def validate_csv(df: pd.DataFrame) -> List[str]:
    """
    Validate that the CSV has the required columns and format.
    
    Args:
        df: Pandas DataFrame containing CSV data
        
    Returns:
        List of error messages, empty if no errors
    """
    errors = []
    required_columns = ['File Name', 'Image: URL']
    
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check for empty values in required columns
    for col in required_columns:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Column '{col}' contains empty values")
            
    return errors

def fetch_and_process_image(
    row: pd.Series, 
    fullsize_dir: Path, 
    thumbnail_dir: Path, 
    limit_size: bool, 
    remove_exif: bool, 
    sequence_dict: Optional[Dict[str, int]] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch and process a single image.
    
    Args:
        row: Pandas Series containing image data
        fullsize_dir: Directory for full-size images
        thumbnail_dir: Directory for thumbnails
        limit_size: Whether to limit image size
        remove_exif: Whether to remove EXIF data
        sequence_dict: Dictionary to track sequence numbers
        
    Returns:
        Dictionary with EXIF data or None if processing failed
    """
    # Replace invalid Windows filename characters with underscore
    original_filename = row['File Name']
    filename = re.sub(r'[\\/:*?"<>|]', '_', pad_id_with_sequence(original_filename, None, sequence_dict))
    filepath = fullsize_dir / filename
    filepath_small = thumbnail_dir / filename
    
    try:
        # Fetch the image
        response = requests.get(row['Image: URL'], timeout=10)
        
        # Use context manager for better resource management
        with Image.open(BytesIO(response.content)) as img:
            log_memory_usage(f"after opening {filename}")

            # Get EXIF data before any modifications
            exif_data = get_exif_data(img)
            
            # Add filename and original filename to exif data
            exif_info = {
                'FileName': filename,
                'OriginalFileName': original_filename,
                'Width': exif_data['Width'],
                'Height': exif_data['Height'],
                'DateTimeCreated': exif_data['DateTimeCreated'],
                'DateTimeOriginal': exif_data['DateTimeOriginal']
            }

            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Record original size and resize if needed
            original_size = f"{img.width}x{img.height}"
            resized = False
            
            # Create a copy for processing
            processed_img = img.copy()
            
            if limit_size and (processed_img.width > APP_CONFIG["fullsize_limit"][0] or 
                              processed_img.height > APP_CONFIG["fullsize_limit"][1]):
                processed_img.thumbnail(APP_CONFIG["fullsize_limit"])
                resized = True

            # Remove EXIF if requested
            if remove_exif:
                new_img = Image.new('RGB', processed_img.size)
                new_img.paste(processed_img)
                processed_img = new_img
                log_memory_usage(f"after EXIF removal for {filename}")

            # Save full-size image
            processed_img.save(filepath, "JPEG", quality=APP_CONFIG["jpeg_quality"])

            # Create and save thumbnail
            thumbnail = processed_img.copy()
            thumbnail.thumbnail(APP_CONFIG["thumbnail_size"])
            thumbnail.save(filepath_small, "JPEG")

            # Clean up
            del thumbnail
            del processed_img
            
            # Return status information
            return {
                'exif_info': exif_info,
                'status': "resized" if resized else "original size",
                'original_size': original_size
            }

    except Exception as e:
        st.error(f"Error processing image for file {filename}: {str(e)}")
        if APP_CONFIG["debug"]:
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        # Force garbage collection
        gc.collect()
        log_memory_usage(f"after cleanup for {filename}")

def write_exif_data(exif_csv_path: Path, exif_data_list: List[Dict[str, Any]]) -> None:
    """
    Write EXIF data to CSV file.
    
    Args:
        exif_csv_path: Path to CSV file
        exif_data_list: List of dictionaries containing EXIF data
    """
    with open(exif_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['FileName', 'OriginalFileName', 'Width', 'Height', 
                     'DateTimeCreated', 'DateTimeOriginal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in exif_data_list:
            writer.writerow(data)

def process_images(csv_file, limit_size=True, remove_exif=True, add_sequence=False) -> Optional[Path]:
    """
    Process all images from CSV file.
    
    Args:
        csv_file: CSV file with image data
        limit_size: Whether to limit image size
        remove_exif: Whether to remove EXIF data
        add_sequence: Whether to add sequence numbers
        
    Returns:
        Path to temporary directory with processed images or None if processing failed
    """
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Set up directory structure
    fullsize_dir, thumbnail_dir, exif_csv_path = setup_directories(temp_dir, limit_size, remove_exif)
    
    # Initialize EXIF data list
    exif_data_list = []

    try:
        if APP_CONFIG["debug"]:
            st.write("Reading CSV file...")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        log_memory_usage("after reading CSV")

        # Validate CSV
        errors = validate_csv(df)
        if errors:
            for error in errors:
                st.error(error)
            return None

        # Initialize sequence dictionary if needed
        sequence_dict = {} if add_sequence else None

        # Set up progress tracking
        total_images = len(df)
        st.write(f"Total images to process: {total_images}")
        progress_bar = st.progress(0)

        # Process each image
        for index, row in df.iterrows():
            result = fetch_and_process_image(
                row, fullsize_dir, thumbnail_dir, limit_size, remove_exif, sequence_dict
            )
            
            if result:
                exif_data_list.append(result['exif_info'])
                st.write(f"Processed {index + 1}/{total_images}: {result['exif_info']['FileName']} "
                         f"({result['original_size']}, {result['status']})")
            
            # Update progress
            progress_bar.progress((index + 1) / total_images)

        # Write EXIF data to CSV
        write_exif_data(exif_csv_path, exif_data_list)

    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        if APP_CONFIG["debug"]:
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        return None

    return temp_dir

def create_zip(directory: Path, csv_filename: str) -> Tuple[str, bytes]:
    """
    Create a zip file with all processed files and return its content as bytes.
    
    Args:
        directory: Directory containing files to zip
        csv_filename: Original CSV filename
        
    Returns:
        Tuple of (zip_filename, zip_content_bytes)
    """
    # Create a zip filename based on the input CSV
    zip_filename = f"{Path(csv_filename).stem}_images.zip"
    
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

# This function is no longer needed since we're not saving zip files to disk
# def cleanup_old_zips() -> None:
#     """Remove previous zip files from temp directory."""
#     temp_dir = Path(tempfile.gettempdir())
#     for file in temp_dir.glob('*_images.zip'):
#         file.unlink()

def main() -> None:
    """Main application function."""
    st.title(APP_CONFIG["page_title"])

    # We no longer need to clean up zip files since we're not saving them to disk
    # cleanup_old_zips()

    st.header("CSV File Upload")
    st.write(
        "Please upload the CSV file with columns 'File Name' and 'Image: URL'. "
        "All images will be fetched and padded with leading zeros. "
        "We'll create thumbnails and provide a zip for download. "
        "EXIF data (dimensions, creation date, original capture date) will be extracted "
        "and included in a separate CSV file in the zip."
    )

    # User options
    limit_size = st.checkbox("Limit image size to 3840x2160 pixels", value=True)
    remove_exif = st.checkbox("Remove EXIF metadata from images", value=True)
    add_sequence = st.checkbox("Add sequence # after ID for multiple images from the same ID", value=False)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("Process Images"):
            with st.spinner("Processing images..."):
                temp_dir = process_images(uploaded_file, limit_size, remove_exif, add_sequence)
                
                if temp_dir:
                    # Create zip file in memory
                    zip_filename, zip_bytes = create_zip(temp_dir, uploaded_file.name)
                    
                    # Clean up the temp directory
                    shutil.rmtree(temp_dir)

                    # Provide download button
                    st.download_button(
                        label="Download Processed Images (Full-size, Thumbnails, and EXIF Data CSV)",
                        data=zip_bytes,
                        file_name=zip_filename,
                        mime="application/zip"
                    )
                else:
                    st.error("Failed to process images. Please check the CSV file format.")

if __name__ == "__main__":
    main()
