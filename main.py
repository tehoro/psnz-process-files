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
import math
import warnings

# Ignore PIL warnings from large image files - should not be a problem
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# App configuration
APP_CONFIG = {
    "debug": True,  # Enable debugging by default to help troubleshoot memory issues
    "thumbnail_size": (810, 810),
    "fullsize_limit": (3840, 2160),
    "jpeg_quality": 100,
    "page_title": "PSNZ Image Entries Processor",
    "batch_size": 50,  # Further reduced default batch size to avoid memory issues
    "file_read_timeout": 15,  # Timeout in seconds for file download
    "memory_limit_mb": 2048  # Approximate memory limit in MB before warning
}

# Setup page configuration
st.set_page_config(page_title=APP_CONFIG["page_title"])

# Initialize session state
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = 0
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'total_batches' not in st.session_state:
    st.session_state.total_batches = 0
if 'sequence_dict' not in st.session_state:
    st.session_state.sequence_dict = {}

def log_memory_usage(message: str) -> None:
    """Log current memory usage if debug is enabled and warn if approaching limits."""
    if APP_CONFIG["debug"]:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Format log message
        log_message = f"Memory usage at {message}: {memory_mb:.2f} MB"
        
        # Check if memory usage is getting high
        if memory_mb > APP_CONFIG["memory_limit_mb"] * 0.8:
            st.warning(f"⚠️ {log_message} - approaching memory limit!")
        else:
            st.write(log_message)
        
        # Return memory so we can react to it in code
        return memory_mb

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
        # Fetch the image with increased timeout
        response = requests.get(row['Image: URL'], timeout=APP_CONFIG["file_read_timeout"])
        
        if response.status_code != 200:
            st.error(f"Failed to download image for {filename}: HTTP status {response.status_code}")
            return None
        
        # Check if the response is too large
        content_size_mb = len(response.content) / (1024 * 1024)
        if content_size_mb > 50:  # If image is larger than 50MB
            st.warning(f"Very large image detected for {filename}: {content_size_mb:.1f}MB")
        
        # Use context manager and try-except for better error handling
        try:
            # Open image with a try-except block to catch decompression errors
            with Image.open(BytesIO(response.content)) as img:
                memory_usage = log_memory_usage(f"after opening {filename}")

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
                
                # For very large images, force resize to prevent memory issues
                if img.width * img.height > 40000000:  # More than 40 megapixels
                    st.warning(f"Very large image detected for {filename}: {img.width}x{img.height} pixels")
                    # Force limit size for these large images
                    limit_size = True
                
                # Create a copy for processing
                processed_img = img.copy()
                del img  # Release the original image memory
                
                # Check if memory is getting high after loading large image
                memory_usage = log_memory_usage(f"after copying {filename}")
                if memory_usage > APP_CONFIG["memory_limit_mb"] * 0.9:
                    # Try emergency garbage collection
                    gc.collect()
                
                if limit_size and (processed_img.width > APP_CONFIG["fullsize_limit"][0] or 
                                processed_img.height > APP_CONFIG["fullsize_limit"][1]):
                    processed_img.thumbnail(APP_CONFIG["fullsize_limit"])
                    resized = True
                    # After resizing, force garbage collection
                    gc.collect()

                # Remove EXIF if requested
                if remove_exif:
                    new_img = Image.new('RGB', processed_img.size)
                    new_img.paste(processed_img)
                    del processed_img  # Release memory from previous image
                    processed_img = new_img
                    log_memory_usage(f"after EXIF removal for {filename}")

                # Save full-size image
                processed_img.save(filepath, "JPEG", quality=APP_CONFIG["jpeg_quality"])

                # Create and save thumbnail
                thumbnail = processed_img.copy()
                thumbnail.thumbnail(APP_CONFIG["thumbnail_size"])
                thumbnail.save(filepath_small, "JPEG")

                # Clean up immediately to free memory
                del thumbnail
                del processed_img
                
                # Return status information
                return {
                    'exif_info': exif_info,
                    'status': "resized" if resized else "original size",
                    'original_size': original_size
                }
        except Image.DecompressionBombError:
            st.error(f"Image {filename} is too large to process safely")
            return None
        except Exception as img_error:
            st.error(f"Error processing image content for {filename}: {str(img_error)}")
            if APP_CONFIG["debug"]:
                import traceback
                st.error(f"Image processing traceback: {traceback.format_exc()}")
            return None

    except requests.RequestException as e:
        st.error(f"Network error downloading image for {filename}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing image for file {filename}: {str(e)}")
        if APP_CONFIG["debug"]:
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        # Force garbage collection after every image
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
    
    # Create a status message and progress bar for zip creation
    zip_status = st.empty()
    zip_status.info("Creating zip file... this may take a while for large batches")
    zip_progress = st.progress(0)
    
    # Use BytesIO to create the zip in memory instead of writing to disk
    zip_buffer = BytesIO()
    
    # Count total files for progress tracking
    total_files = 0
    for root, _, files in os.walk(directory):
        total_files += len(files)
    
    # Create the zip file with progress updates
    files_processed = 0
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                arcname = str(file_path.relative_to(directory))
                
                # Update status periodically
                if files_processed % 10 == 0 or files_processed == total_files - 1:
                    zip_status.info(f"Creating zip file... adding file {files_processed + 1}/{total_files}")
                    zip_progress.progress(files_processed / total_files)
                
                # Try to write file with error handling
                try:
                    zipf.write(file_path, arcname)
                except Exception as e:
                    zip_status.error(f"Error adding {file} to zip: {str(e)}")
                    # Continue with other files
                
                files_processed += 1
    
    # Reset buffer position to the beginning
    zip_buffer.seek(0)
    
    # Complete progress and update status
    zip_progress.progress(1.0)
    zip_status.success(f"Zip file created successfully with {files_processed} files")
    
    # Return filename and content as bytes
    return zip_filename, zip_buffer.getvalue()

def process_single_batch(batch_index, df, csv_filename, limit_size, remove_exif, add_sequence):
    """
    Process a single batch of images.
    
    Args:
        batch_index: Index of the batch to process
        df: DataFrame containing CSV data
        csv_filename: Name of the CSV file
        limit_size: Whether to limit image size
        remove_exif: Whether to remove EXIF data
        add_sequence: Whether to add sequence numbers
        
    Returns:
        Dictionary with batch information
    """
    # Status area for progress feedback
    status_area = st.empty()
    status_area.info(f"Setting up batch {batch_index + 1}...")
    
    # Get sequence dictionary from session state
    sequence_dict = st.session_state.sequence_dict if add_sequence else None
    
    # Calculate total images and batch parameters
    total_images = len(df)
    batch_size = st.session_state.batch_size
    
    # Calculate start and end indices for this batch
    start_idx = batch_index * batch_size
    end_idx = min(start_idx + batch_size, total_images)
    batch_df = df.iloc[start_idx:end_idx]
    
    try:
        # Create a new temporary directory for this batch
        temp_dir = Path(tempfile.mkdtemp())
        status_area.info(f"Created temporary directory for batch {batch_index + 1}")
        
        # Set up directory structure
        fullsize_dir, thumbnail_dir, exif_csv_path = setup_directories(temp_dir, limit_size, remove_exif)
        status_area.info(f"Set up directory structure for batch {batch_index + 1}")
        
        batch_exif_data = []
        
        # Set up progress tracking for this batch
        batch_header = st.empty()
        batch_header.write(f"\nProcessing Batch {batch_index + 1}/{st.session_state.total_batches} (images {start_idx + 1} to {end_idx})")
        progress_bar = st.progress(0)
        status_messages = st.empty()

        # Log memory
        log_memory_usage(f"before processing batch {batch_index + 1}")

        # Process each image in this batch
        for i, (index, row) in enumerate(batch_df.iterrows()):
            current_image_idx = start_idx + i + 1
            status_area.info(f"Processing image {current_image_idx}/{total_images} in batch {batch_index + 1}")
            
            result = fetch_and_process_image(
                row, fullsize_dir, thumbnail_dir, limit_size, remove_exif, sequence_dict
            )
            
            if result:
                batch_exif_data.append(result['exif_info'])
                status_messages.write(f"Processed {current_image_idx}/{total_images}: {result['exif_info']['FileName']} "
                        f"({result['original_size']}, {result['status']})")
            
            # Update progress
            progress_bar.progress((i + 1) / len(batch_df))
            
            # Periodically force garbage collection for large batches
            if i > 0 and i % 25 == 0:
                gc.collect()
                log_memory_usage(f"after {i} images in batch {batch_index + 1}")

        status_area.info(f"Writing EXIF data to CSV for batch {batch_index + 1}")
        # Write EXIF data to CSV for this batch
        write_exif_data(exif_csv_path, batch_exif_data)
        
        status_area.info(f"Creating zip file for batch {batch_index + 1} (this may take a while)...")
        log_memory_usage(f"before creating zip for batch {batch_index + 1}")
        
        # Create zip file for this batch
        batch_zip_filename, batch_zip_bytes = create_zip(temp_dir, csv_filename, batch_index + 1)
        
        # Create batch result
        batch_result = {
            'batch_num': batch_index + 1,
            'zip_filename': batch_zip_filename,
            'zip_bytes': batch_zip_bytes,
            'num_images': len(batch_df),
            'start_idx': start_idx,
            'end_idx': end_idx - 1
        }
        
        status_area.info(f"Cleaning up temporary files for batch {batch_index + 1}")
        # Clean up the temp directory
        shutil.rmtree(temp_dir)
        
        # Force garbage collection after processing a batch
        gc.collect()
        log_memory_usage(f"after batch {batch_index + 1} complete")
        
        status_area.success(f"Batch {batch_index + 1} completed successfully!")
        return batch_result
        
    except Exception as e:
        status_area.error(f"Error processing batch {batch_index + 1}")
        st.error(f"Error processing batch {batch_index + 1}: {str(e)}")
        if APP_CONFIG["debug"]:
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        
        # Try to clean up resources on error
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
        except:
            pass
        
        # Force garbage collection on error
        gc.collect()
        return None

def init_processing(uploaded_file, limit_size, remove_exif, add_sequence, batch_size):
    """Initialize the batch processing state."""
    # Read and validate CSV file
    df = pd.read_csv(uploaded_file)
    errors = validate_csv(df)
    if errors:
        for error in errors:
            st.error(error)
        return False
    
    # Calculate number of batches
    total_images = len(df)
    num_batches = math.ceil(total_images / batch_size)
    
    # Initialize or reset session state
    st.session_state.csv_data = df
    st.session_state.batch_size = batch_size
    st.session_state.total_batches = num_batches
    st.session_state.current_batch = 0
    st.session_state.batch_results = []
    st.session_state.sequence_dict = {} if add_sequence else None
    st.session_state.processing_options = {
        'limit_size': limit_size,
        'remove_exif': remove_exif,
        'add_sequence': add_sequence,
        'csv_filename': uploaded_file.name
    }
    
    return True

def continue_processing():
    """Continue processing the next batch."""
    current_batch = st.session_state.current_batch
    if current_batch < st.session_state.total_batches:
        # Display a header message with batch information
        st.subheader(f"Processing Batch {current_batch + 1} of {st.session_state.total_batches}")
        
        # Create a separate container for memory monitoring
        memory_container = st.container()
        with memory_container:
            if APP_CONFIG["debug"]:
                st.info("Memory monitoring is enabled. You'll see memory usage information during processing.")
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024
                st.write(f"Starting memory usage: {start_memory:.2f}MB")
        
        # Check if memory usage is already high before starting
        process = psutil.Process()
        if process.memory_info().rss / 1024 / 1024 > APP_CONFIG["memory_limit_mb"] * 0.7:
            st.warning("⚠️ Memory usage is already high. Consider restarting the app before processing this batch.")
        
        try:
            # Process the current batch
            with st.spinner(f"Processing batch {current_batch + 1} of {st.session_state.total_batches}..."):
                batch_result = process_single_batch(
                    current_batch,
                    st.session_state.csv_data,
                    st.session_state.processing_options['csv_filename'],
                    st.session_state.processing_options['limit_size'],
                    st.session_state.processing_options['remove_exif'],
                    st.session_state.processing_options['add_sequence']
                )
                
                if batch_result:
                    # Add result to the list
                    st.session_state.batch_results.append(batch_result)
                    
                    # Display success message
                    st.success(f"✅ Batch {current_batch + 1} processed successfully!")
                    
                    # Create a download button for the just-processed batch
                    download_container = st.container()
                    with download_container:
                        st.write("### Download Current Batch")
                        st.download_button(
                            label=f"Download Batch {batch_result['batch_num']} (Images {batch_result['start_idx'] + 1}-{batch_result['end_idx'] + 1})",
                            data=batch_result['zip_bytes'],
                            file_name=batch_result['zip_filename'],
                            mime="application/zip",
                            key=f"download_latest_batch_{batch_result['batch_num']}",
                            use_container_width=True
                        )
                        st.info("👆 Download this batch before continuing to free up memory")
                    
                    # Increment the current batch
                    st.session_state.current_batch += 1
                    
                    # Check if all batches are processed
                    if st.session_state.current_batch >= st.session_state.total_batches:
                        st.session_state.processing_complete = True
                else:
                    # Display error message if batch processing failed
                    st.error(f"❌ Batch {current_batch + 1} processing failed. See error details above.")
        
        except Exception as e:
            # Catch any exceptions that weren't caught in the batch processing
            st.error(f"Unexpected error during batch processing: {str(e)}")
            if APP_CONFIG["debug"]:
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
        
        # Display memory usage after batch processing
        if APP_CONFIG["debug"]:
            with memory_container:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024
                st.write(f"Ending memory usage: {end_memory:.2f}MB")
                st.write(f"Memory change: {end_memory - start_memory:.2f}MB")
                
                # Provide recommendations if memory usage is high
                if end_memory > APP_CONFIG["memory_limit_mb"] * 0.8:
                    st.warning("⚠️ Memory usage is high after batch processing. Consider:")
                    st.write("- Downloading the current batch results and restarting the app")
                    st.write("- Reducing batch size for future batches")
                    st.write("- Further reducing the image resolution")
        
        # Force garbage collection after batch processing
        gc.collect()
        
        # Add a continue button directly in this function
        continue_container = st.container()
        with continue_container:
            st.write("### Continue to next batch?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process Next Batch", key="continue_next_batch", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("Stop Processing", key="stop_processing_button", use_container_width=True):
                    st.session_state.processing_complete = True
                    st.rerun()

def main() -> None:
    """Main application function."""
    st.title(APP_CONFIG["page_title"])

    # Add expandable section for app info and tips
    with st.expander("ℹ️ About this app & Tips for preventing crashes"):
        st.markdown("""
        ### About
        This app processes image entries from a CSV file, resizing them and organizing them for competition judging.
        
        ### Tips to prevent crashes
        1. **Use smaller batch sizes** (30-50 images) for more reliable processing
        2. **Download each batch** as soon as it's ready to free up memory
        3. **Restart the app** if you notice slow performance or warnings about memory usage
        4. **Reduce image resolution** further if dealing with very high-resolution images
        5. **Enable debug mode** to monitor memory usage during processing
        """)

    st.header("CSV File Upload")
    st.write(
        "Please upload the CSV file with columns 'File Name' and 'Image: URL'. "
        "All images will be fetched and processed. "
        "To prevent memory issues, images will be processed in batches, "
        "with each batch requiring manual progression."
    )

    # Debug mode toggle
    debug_mode = st.checkbox("Enable debug mode (shows memory usage)", value=APP_CONFIG["debug"])
    APP_CONFIG["debug"] = debug_mode

    # User options in columns for better space usage
    col1, col2 = st.columns(2)
    with col1:
        limit_size = st.checkbox("Limit image size to 3840x2160 pixels", value=True)
        remove_exif = st.checkbox("Remove EXIF metadata from images", value=True)
    with col2:
        add_sequence = st.checkbox("Add sequence # after ID for multiple images", value=False)
        # Memory limit option for advanced users
        memory_limit = st.number_input("Memory limit (MB)", 
                                     min_value=512, max_value=8192, 
                                     value=APP_CONFIG["memory_limit_mb"], step=512,
                                     help="Set a warning threshold for memory usage")
        APP_CONFIG["memory_limit_mb"] = memory_limit
    
    # Batch size option
    col1, col2 = st.columns([3, 1])
    with col1:
        batch_size = st.slider("Batch size (max images per download)", 
                             min_value=10, max_value=200, value=APP_CONFIG["batch_size"], step=10,
                             help="Lower values (30-50) reduce memory usage and are more stable")
    with col2:
        st.write("")
        st.write("")
        if st.button("Reset to Default"):
            batch_size = APP_CONFIG["batch_size"]
            st.rerun()

    # Display current memory usage
    if APP_CONFIG["debug"]:
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024
        st.write(f"Current memory usage: {current_memory:.2f}MB")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Initialize or continue processing
        if not st.session_state.batch_results and not st.session_state.processing_complete:
            # Show a preview of the CSV data
            try:
                preview_df = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)  # Reset file pointer after reading
                
                st.write(f"CSV contains {len(preview_df)} image entries")
                with st.expander("Preview CSV data"):
                    st.dataframe(preview_df.head(5))
                
                # Calculate estimated zip file sizes
                avg_image_size_mb = 2.5  # Estimated average size after resizing to 4K
                images_per_batch = batch_size
                estimated_batch_size_mb = avg_image_size_mb * images_per_batch
                
                st.info(f"Estimated download size per batch: ~{estimated_batch_size_mb:.1f}MB (varies based on actual images)")
                
                # Calculate total batches
                total_batches = math.ceil(len(preview_df) / batch_size)
                st.write(f"Will process in {total_batches} batches of up to {batch_size} images each")
                
                # Check for potential memory issues
                if estimated_batch_size_mb > APP_CONFIG["memory_limit_mb"] * 0.7:
                    st.warning("⚠️ Batch size may be too large for available memory. Consider reducing batch size.")
            except Exception as e:
                st.error(f"Error previewing CSV: {str(e)}")
            
            # Start processing button with clear instructions
            st.write("#### Ready to begin?")
            start_col1, start_col2 = st.columns([2, 1])
            with start_col1:
                start_button = st.button("Start Processing", use_container_width=True)
            with start_col2:
                # Add a visual indicator of what to expect
                st.info("Process one batch at a time")
            
            if start_button:
                if init_processing(uploaded_file, limit_size, remove_exif, add_sequence, batch_size):
                    continue_processing()
                    
        elif not st.session_state.processing_complete:
            # Display progress and continue button
            st.info(f"Processed {st.session_state.current_batch} of {st.session_state.total_batches} batches.")
            
            # Progress bar for overall completion
            overall_progress = st.session_state.current_batch / st.session_state.total_batches
            st.progress(overall_progress)
            
            # Display completed batches for download
            if st.session_state.batch_results:
                st.subheader("Completed Batches")
                for batch in st.session_state.batch_results:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.download_button(
                            label=f"Download Batch {batch['batch_num']} (Images {batch['start_idx'] + 1}-{batch['end_idx'] + 1})",
                            data=batch['zip_bytes'],
                            file_name=batch['zip_filename'],
                            mime="application/zip",
                            key=f"download_batch_{batch['batch_num']}",
                            use_container_width=True
                        )
                    with col2:
                        st.write(f"Contains {batch['num_images']} images")
            
            # Continue or stop buttons
            st.write("#### Continue Processing?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process Next Batch", use_container_width=True):
                    continue_processing()
                    st.rerun()
            with col2:
                if st.button("Stop Processing", use_container_width=True):
                    st.session_state.processing_complete = True
                    st.rerun()
        else:
            # All processing complete
            st.success(f"✅ Processing complete! {len(st.session_state.batch_results)} batch(es) ready for download.")
            
            # Display all batches for download in a more organized way
            st.subheader("Download Processed Batches")
            for i, batch in enumerate(st.session_state.batch_results):
                if i % 2 == 0:
                    cols = st.columns(2)
                
                with cols[i % 2]:
                    st.download_button(
                        label=f"Batch {batch['batch_num']} (Images {batch['start_idx'] + 1}-{batch['end_idx'] + 1})",
                        data=batch['zip_bytes'],
                        file_name=batch['zip_filename'],
                        mime="application/zip",
                        key=f"download_batch_{batch['batch_num']}",
                        use_container_width=True
                    )
            
            # Add summary information
            total_images = sum(batch['num_images'] for batch in st.session_state.batch_results)
            st.write(f"Total images processed: {total_images}")
            
            # Reset button
            if st.button("Process More Images (Start Over)", type="primary"):
                # Reset all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()
