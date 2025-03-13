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
import time

# Disable PIL DecompressionBombWarning
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Set a higher decompression bomb limit if needed
Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely, use with caution

# App configuration
APP_CONFIG = {
    "debug": False,
    "thumbnail_size": (810, 810),
    "fullsize_limit": (3840, 2160),
    "absolute_max_size": 7680,  # Maximum dimension for any image, regardless of settings
    "jpeg_quality": 100,  # Could reduce from 100 to save memory
    "page_title": "PSNZ Image Entries Processor",
    "batch_size": 10,    # Process images in batches to reduce memory usage
    "memory_threshold": 0.8  # Trigger garbage collection when memory usage exceeds 80%
}

# Setup page configuration
st.set_page_config(page_title=APP_CONFIG["page_title"], layout="wide")

# Create a resource usage limiter for Streamlit
@st.cache_resource
def get_resource_limiter():
    """Create a semaphore to limit concurrent resource-intensive operations"""
    import threading
    return threading.Semaphore(1)

# Add this to the main function before processing:
    # Get resource limiter
    resource_limiter = get_resource_limiter()
    
    if uploaded_file is not None:
        st.write("CSV file uploaded. Click 'Process Images' to begin.")
        if st.button("Process Images"):
            # Try to acquire the resource lock to prevent multiple concurrent processes
            if not resource_limiter.acquire(blocking=False):
                st.error("Another process is already running. Please wait for it to complete.")
                return
            
            try:
                with st.spinner("Processing images..."):
                    # Log initial memory usage
                    log_memory_usage("before processing")
                    
                    # Process images
                    temp_dir = process_images(uploaded_file, limit_size, remove_exif, add_sequence)
                    
                    # Continue with zip creation and download...
                    # (rest of the code)
            finally:
                # Release the resource lock
                resource_limiter.release()

def log_memory_usage(message: str) -> float:
    """Log current memory usage if debug is enabled and return usage percentage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = memory_info.rss / psutil.virtual_memory().total
    
    if APP_CONFIG["debug"]:
        st.write(f"Memory usage at {message}: {memory_info.rss / 1024 / 1024:.2f} MB ({memory_percent:.2%})")
    
    return memory_percent

def force_garbage_collection():
    """Force garbage collection and wait for it to complete."""
    collected = gc.collect(generation=2)  # Full collection
    if APP_CONFIG["debug"]:
        st.write(f"Garbage collection: collected {collected} objects")
    time.sleep(0.1)  # Give the system a moment to actually free memory

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
        if hasattr(img, '_getexif') and img._getexif():
            exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in img._getexif().items()}
            
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
        # Fetch the image with a longer timeout for large images
        response = requests.get(row['Image: URL'], timeout=30)
        
        # Check if response is valid
        if response.status_code != 200:
            st.error(f"Failed to download image (HTTP {response.status_code}): {filename}")
            return None
            
        # Load image content
        image_data = BytesIO(response.content)
        
        # Open image and process
        with Image.open(image_data) as img:
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

            # Record original size
            original_size = f"{img.width}x{img.height}"
            resized = False
            
            # Create a modified copy for processing
            if img.mode in ('RGBA', 'P'):
                processed_img = img.convert('RGB')
            else:
                processed_img = img.copy()
            
            # First check against absolute maximum size
            if processed_img.width > APP_CONFIG["absolute_max_size"] or processed_img.height > APP_CONFIG["absolute_max_size"]:
                # Calculate aspect ratio
                aspect_ratio = processed_img.width / processed_img.height
                
                if processed_img.width > processed_img.height:
                    new_width = APP_CONFIG["absolute_max_size"]
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = APP_CONFIG["absolute_max_size"]
                    new_width = int(new_height * aspect_ratio)
                
                processed_img = processed_img.resize((new_width, new_height), Image.LANCZOS)
                resized = True
                
                if APP_CONFIG["debug"]:
                    st.write(f"Image {filename} exceeded maximum allowed dimension, resized to {new_width}x{new_height}")
            
            # Then check against the normal fullsize limit if that option is selected
            elif limit_size and (processed_img.width > APP_CONFIG["fullsize_limit"][0] or 
                              processed_img.height > APP_CONFIG["fullsize_limit"][1]):
                processed_img.thumbnail(APP_CONFIG["fullsize_limit"])
                resized = True

            # Remove EXIF if requested
            if remove_exif:
                new_img = Image.new('RGB', processed_img.size)
                new_img.paste(processed_img)
                processed_img = new_img

            # Save full-size image
            processed_img.save(filepath, "JPEG", quality=APP_CONFIG["jpeg_quality"])

            # Create and save thumbnail
            thumbnail = processed_img.copy()
            thumbnail.thumbnail(APP_CONFIG["thumbnail_size"])
            thumbnail.save(filepath_small, "JPEG")

            # Clean up to reduce memory usage
            del thumbnail
            del processed_img
        
        # Force cleanup of image data
        del image_data
        del response
        
        # Memory check and garbage collection if needed
        if log_memory_usage(f"after processing {filename}") > APP_CONFIG["memory_threshold"]:
            force_garbage_collection()
            
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
        status_container = st.empty()

        # Create a container for status messages to avoid filling the screen
        with st.expander("Processing Details", expanded=True):
            status_area = st.empty()
            status_messages = []

        # Process each image in batches to manage memory
        batch_size = min(APP_CONFIG["batch_size"], total_images)
        
        for start_idx in range(0, total_images, batch_size):
            end_idx = min(start_idx + batch_size, total_images)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Status update for batch
            status_container.write(f"Processing images {start_idx+1} to {end_idx} of {total_images}...")
            
            # Process each image in the batch
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                current_idx = start_idx + idx
                
                # Process the image
                result = fetch_and_process_image(
                    row, fullsize_dir, thumbnail_dir, limit_size, remove_exif, sequence_dict
                )
                
                if result:
                    exif_data_list.append(result['exif_info'])
                    status_message = f"Processed {current_idx + 1}/{total_images}: {result['exif_info']['FileName']} " \
                                    f"({result['original_size']}, {result['status']})"
                else:
                    status_message = f"Failed to process image {current_idx + 1}/{total_images}"
                
                # Update status list and display
                status_messages.append(status_message)
                if len(status_messages) > 10:  # Keep only the 10 most recent messages
                    status_messages = status_messages[-10:]
                status_area.text("\n".join(status_messages))
                
                # Update progress
                progress_bar.progress((current_idx + 1) / total_images)
            
            # Force garbage collection between batches
            force_garbage_collection()
        
        # Final status update
        status_container.write(f"Completed processing {total_images} images. Writing metadata...")
        
        # Write EXIF data to CSV
        write_exif_data(exif_csv_path, exif_data_list)
        
        # Final cleanup
        force_garbage_collection()
        status_container.write(f"Processing complete. Preparing download...")

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
    Uses chunking to handle large file sets without excessive memory usage.
    
    Args:
        directory: Directory containing files to zip
        csv_filename: Original CSV filename
        
    Returns:
        Tuple of (zip_filename, zip_content_bytes)
    """
    # Create a zip filename based on the input CSV
    zip_filename = f"{Path(csv_filename).stem}_images.zip"
    
    # Create a temporary file for the zip instead of using memory
    temp_zip_path = Path(tempfile.gettempdir()) / f"temp_{int(time.time())}_{zip_filename}"
    
    status_message = st.empty()
    status_message.write("Creating zip file...")
    
    # Count total files for progress reporting
    total_files = 0
    for _, _, files in os.walk(directory):
        total_files += len(files)
    
    try:
        # Create the zip file on disk first
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            file_count = 0
            for root, _, files in os.walk(directory):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file
                    arcname = str(file_path.relative_to(directory))
                    zipf.write(file_path, arcname)
                    
                    file_count += 1
                    if file_count % 10 == 0:
                        status_message.write(f"Adding files to zip: {file_count}/{total_files}")
                        # Check memory and collect garbage if needed
                        if log_memory_usage("during zip creation") > APP_CONFIG["memory_threshold"]:
                            force_garbage_collection()
        
        status_message.write(f"Zip file created successfully! Reading zip file in chunks...")
        
        # Now read the file in chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1MB chunks
        file_size = temp_zip_path.stat().st_size
        chunks = []
        bytes_read = 0
        
        with open(temp_zip_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                    
                chunks.append(chunk)
                bytes_read += len(chunk)
                
                # Update status occasionally
                if len(chunks) % 10 == 0:
                    status_message.write(f"Reading zip file: {bytes_read / file_size:.1%} complete")
                    force_garbage_collection()
        
        # Combine chunks
        status_message.write("Preparing zip file for download...")
        zip_bytes = b''.join(chunks)
        
        # Clean up temp file
        temp_zip_path.unlink()
        
        # Final memory cleanup
        force_garbage_collection()
        status_message.write("Zip file ready for download!")
        
        return zip_filename, zip_bytes
        
    except Exception as e:
        # Make sure we clean up the temp file if there's an error
        if temp_zip_path.exists():
            temp_zip_path.unlink()
        raise e

def main() -> None:
    """Main application function."""
    st.title(APP_CONFIG["page_title"])

    st.header("CSV File Upload")
    st.write(
        "Please upload the CSV file with columns 'File Name' and 'Image: URL'. "
        "All images will be fetched and padded with leading zeros. "
        "We'll create thumbnails and provide a zip for download. "
        "EXIF data (dimensions, creation date, original capture date) will be extracted "
        "and included in a separate CSV file in the zip."
    )

    # Option to enable debug mode
    debug_mode = st.sidebar.checkbox("Enable debug mode", value=False)
    APP_CONFIG["debug"] = debug_mode

    # User options
    col1, col2 = st.columns(2)
    with col1:
        limit_size = st.checkbox("Limit image size to 3840x2160 pixels", value=True)
        remove_exif = st.checkbox("Remove EXIF metadata from images", value=True)
    with col2:
        add_sequence = st.checkbox("Add sequence # after ID for multiple images from the same ID", value=False)
        jpeg_quality = st.slider("JPEG Quality", min_value=70, max_value=100, value=95, 
                                help="Lower quality saves memory but reduces image quality")
        APP_CONFIG["jpeg_quality"] = jpeg_quality

    # Advanced settings in sidebar
    st.sidebar.header("Advanced Settings")
    APP_CONFIG["batch_size"] = st.sidebar.slider("Batch Size", min_value=1, max_value=30, value=10, 
                                              help="Number of images to process at once")
    memory_threshold = st.sidebar.slider("Memory Threshold (%)", min_value=50, max_value=95, value=80, 
                                      help="Trigger garbage collection when memory usage exceeds this percentage")
    APP_CONFIG["memory_threshold"] = memory_threshold / 100.0
    
    # Maximum size setting
    max_size = st.sidebar.slider("Absolute Maximum Size (pixels)", min_value=4000, max_value=12000, value=7680, 
                               help="Maximum dimension for any image, regardless of other settings")
    APP_CONFIG["absolute_max_size"] = max_size

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.write("CSV file uploaded. Click 'Process Images' to begin.")
        if st.button("Process Images"):
            with st.spinner("Processing images..."):
                # Log initial memory usage
                log_memory_usage("before processing")
                
                # Process images
                temp_dir = process_images(uploaded_file, limit_size, remove_exif, add_sequence)
                
                if temp_dir:
                    try:
                        # Create zip file in memory
                        zip_filename, zip_bytes = create_zip(temp_dir, uploaded_file.name)
                        
                        # Clean up the temp directory
                        shutil.rmtree(temp_dir)
                        force_garbage_collection()

                        # Provide download button
                        st.success("Processing complete! Click the button below to download your files:")
                        st.download_button(
                            label="Download Processed Images (Full-size, Thumbnails, and EXIF Data CSV)",
                            data=zip_bytes,
                            file_name=zip_filename,
                            mime="application/zip"
                        )
                    except Exception as e:
                        st.error(f"Error creating zip file: {str(e)}")
                        if APP_CONFIG["debug"]:
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                else:
                    st.error("Failed to process images. Please check the CSV file format.")# Replace the download section in main() with this:

                if temp_dir:
                    try:
                        # Show a message about downloading process
                        download_note = st.info(
                            "Please wait while we prepare your download. " +
                            "This may take a few minutes for large batches of images. " +
                            "Do not close this page."
                        )
                        
                        # Create zip file in memory
                        zip_filename, zip_bytes = create_zip(temp_dir, uploaded_file.name)
                        
                        # Clean up the temp directory
                        shutil.rmtree(temp_dir)
                        force_garbage_collection()

                        # Provide download button
                        download_note.success("Processing complete! Click the button below to download your files:")
                        
                        # Split the download into chunks if it's very large (over 200MB)
                        zip_size_mb = len(zip_bytes) / (1024 * 1024)
                        if zip_size_mb > 200:
                            st.warning(
                                f"The generated zip file is quite large ({zip_size_mb:.1f} MB). " +
                                "If you encounter issues downloading it, try processing fewer images at once."
                            )
                        
                        # Use the download button
                        st.download_button(
                            label=f"Download Processed Images ({zip_size_mb:.1f} MB)",
                            data=zip_bytes,
                            file_name=zip_filename,
                            mime="application/zip"
                        )
                        
                        # Force garbage collection again
                        del zip_bytes
                        force_garbage_collection()
                        
                    except Exception as e:
                        st.error(f"Error creating zip file: {str(e)}")
                        if APP_CONFIG["debug"]:
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                else:
                    st.error("Failed to process images. Please check the CSV file format.")
                    
if __name__ == "__main__":
    main()
