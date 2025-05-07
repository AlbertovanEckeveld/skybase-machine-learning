import torch
import cv2
import numpy as np
import os
import time
from PIL import Image
from get_exif import get_exif_data


def detect_objects(image_path, conf_threshold=0.5, save_output=True, model_size='s'):
    """
    Detect objects in an image with YOLOv5 and show performance metrics

    Args:
        image_path: Path to the image file
        conf_threshold: Minimum confidence score for detections (0-1)
        save_output: Whether to save annotated image with detections
        model_size: YOLOv5 model size ('n', 's', 'm', 'l', 'x')

    Returns:
        objects: Dictionary of detected objects with their properties
        inference_time: Time taken for model inference
    """
    # Track overall function time
    function_start = time.time()
    print(f"Processing image: {os.path.basename(image_path)}")

    # Load the YOLOv5 model
    load_start = time.time()
    model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}', pretrained=True)
    model.conf = conf_threshold  # Set confidence threshold
    load_time = time.time() - load_start
    print(f"Model loading time: {load_time:.2f} seconds")

    # Load the image
    img_start = time.time()
    image = Image.open(image_path)
    img_time = time.time() - img_start
    print(f"Image loading time: {img_time:.2f} seconds")

    # Get EXIF data
    exif_start = time.time()
    exif_info = get_exif_data(image_path)
    exif_time = time.time() - exif_start

    # Record inference start time
    inference_start = time.time()

    # Perform object detection
    results = model(image)

    # Calculate inference time
    inference_time = time.time() - inference_start

    # Parse results
    parsing_start = time.time()
    detected_objects = results.pandas().xyxy[0]  # Bounding box predictions
    objects = detected_objects[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']].to_dict(orient='records')
    parsing_time = time.time() - parsing_start

    # Save results if requested
    saving_time = 0
    if save_output:
        saving_start = time.time()
        output_path = f"detected_{os.path.basename(image_path)}"

        # Convert PIL image to OpenCV format for annotations
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Draw boxes and labels
        for obj in objects:
            x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
            label = f"{obj['name']} {obj['confidence']:.2f}"

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1 - 25), (x1 + text_size[0], y1), (0, 255, 0), -1)

            # Add text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imwrite(output_path, img)
        saving_time = time.time() - saving_start
        print(f"Detection results saved to {output_path}")

    # Calculate total execution time
    total_time = time.time() - function_start

    # Display timing results
    print(f"\nTIMING REPORT:")
    print(f"Model loading: {load_time:.2f} seconds")
    print(f"Image loading: {img_time:.2f} seconds")
    print(f"EXIF extraction: {exif_time:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Results parsing: {parsing_time:.2f} seconds")
    if save_output:
        print(f"Results saving: {saving_time:.2f} seconds")
    print(f"Total execution: {total_time:.2f} seconds")

    # Display EXIF data
    print(f"\nIMAGE METADATA:")
    if isinstance(exif_info, dict):
        for key, value in exif_info.items():
            print(f"{key}: {value}")
    else:
        print(exif_info)

    # Display detection results
    print(f"\nDETECTION RESULTS (confidence threshold: {conf_threshold}):")
    print(f"Found {len(objects)} objects:")
    for i, obj in enumerate(objects, 1):
        print(f"{i}. {obj['name']} (confidence: {obj['confidence']:.2f})")

    # Visualize results with PyTorch
    results.show()

    return objects, inference_time


if __name__ == "__main__":
    image_path = "example2.jpg"
    detected_objects, detection_time = detect_objects(
        image_path,
        conf_threshold=0.4,  # Adjust confidence threshold as needed
        save_output=True,  # Save annotated image
        model_size='x'  # Model size (options: n, s, m, l, x)
    )