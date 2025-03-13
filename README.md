# PSNZ Image Entries Processor

A Streamlit application for processing image entries for PSNZ competitions.

## Features

- Processes images from a CSV file containing image URLs
- Resizes images to a maximum of 3840x2160 pixels (optional)
- Removes EXIF metadata (optional)
- Creates thumbnails for easier viewing
- Adds sequence numbers for multiple entries from the same ID (optional)
- Extracts and saves EXIF data to a CSV file

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run main.py`

## Usage

1. Prepare a CSV file with at least two columns: `File Name` and `Image: URL`
2. Upload the CSV file to the app
3. Configure options:
   - Limit image size to 3840x2160 pixels
   - Remove EXIF metadata
   - Add sequence numbers for multiple entries from the same ID
4. Click "Process Images"
5. Download the zip file containing processed images and metadata

## Deployment

This app is deployed on Streamlit Cloud at: https://psnz-process-files.streamlit.app/

## Development

To contribute to this project:

1. Create a new branch for your changes
2. Make your changes
3. Test your changes by deploying a development version on Streamlit Cloud
4. Create a pull request

## License

This project is copyright © Neil Gordon, 2025.
