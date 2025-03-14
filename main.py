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

# App configuration
APP_CONFIG = {
    "debug": False,
    "thumbnail_size": (810, 810),
    "fullsize_limit": (3840, 2160),
    "jpeg_quality": 100,
    "page_title": "PSNZ Image Entries Processor",
    "batch_size": 150  # Default batch size to avoid memory issues
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
