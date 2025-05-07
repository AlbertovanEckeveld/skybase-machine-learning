import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(file_path):
    try:
        image = Image.open(file_path)
        exif_data = image._getexif()
        if not exif_data:
            return "No EXIF data found."

        exif_dict = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_dict[tag_name] = value

        # Extract specific EXIF data
        date_time = exif_dict.get("DateTime", "Unknown")
        orientation = exif_dict.get("Orientation", "Unknown")
        gps_info = exif_dict.get("GPSInfo", None)
        camera_make = exif_dict.get("Make", "Unknown")
        camera_model = exif_dict.get("Model", "Unknown")

        # Parse GPS data if available
        location = "Unknown"
        if gps_info:
            gps_data = {}
            for key in gps_info.keys():
                name = GPSTAGS.get(key, key)
                gps_data[name] = gps_info[key]
            location = gps_data

        # Get file information
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        resolution = image.size  # (width, height)

        return {
            "File Name": file_name,
            "File Size (bytes)": file_size,
            "Resolution": resolution,
            "Camera Make": camera_make,
            "Camera Model": camera_model,
            "Date and Time": date_time,
            "Orientation": orientation,
            "Location": location,

        }
    except Exception as e:
        return f"Error reading EXIF data: {e}"

# Example usage
file_path = "example2.jpg"
exif = get_exif_data(file_path)
print(exif)