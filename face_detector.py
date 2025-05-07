import face_recognition
import cv2
import numpy as np
import os
import time
from PIL import Image
from get_exif import get_exif_data

def face_recognition_process(image_path, known_faces_dir=None, save_output=True, tolerance=0.6):
    """
    Detect and recognize faces in an image

    Args:
        image_path: Path to the image file
        known_faces_dir: Directory containing known face images (optional)
        save_output: Whether to save annotated image with face detections
        tolerance: How strict the face matching should be (lower = stricter)

    Returns:
        faces: List of detected faces and their information
        process_time: Time taken for face detection and recognition
    """
    # Track overall function time
    function_start = time.time()
    print(f"Processing image: {os.path.basename(image_path)}")

    # Load image
    img_start = time.time()
    image = face_recognition.load_image_file(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_time = time.time() - img_start
    print(f"Image loading time: {img_time:.2f} seconds")

    # Get EXIF data
    exif_start = time.time()
    exif_info = get_exif_data(image_path)
    exif_time = time.time() - exif_start

    # Known face encoding and names
    known_face_encodings = []
    known_face_names = []

    # If a known faces directory is provided, load and encode those faces
    if known_faces_dir and os.path.isdir(known_faces_dir):
        load_known_start = time.time()

        print("Loading known faces...")
        for filename in os.listdir(known_faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                # Extract name from filename (without extension)
                name = os.path.splitext(filename)[0]

                # Load and encode face
                face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
                try:
                    face_encoding = face_recognition.face_encodings(face_image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    print(f"  - Loaded known face: {name}")
                except IndexError:
                    print(f"  - No face found in {filename}, skipping")

        load_known_time = time.time() - load_known_start
        print(f"Loaded {len(known_face_encodings)} known faces in {load_known_time:.2f} seconds")

    # Detect faces in the image
    detect_start = time.time()
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    detection_time = time.time() - detect_start
    print(f"Face detection time: {detection_time:.2f} seconds")

    # Process the results
    faces = []

    # Match detected faces against known faces if available
    match_start = time.time()
    for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
        # Default name if no match is found
        name = f"Person {i + 1}"
        confidence = None

        # Try to recognize the face if we have known faces
        if known_face_encodings:
            # Compare face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)

            # Use the known face with the smallest distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

        # Extract face location
        top, right, bottom, left = face_location

        # Add to results
        faces.append({
            "name": name,
            "confidence": confidence,
            "location": (left, top, right, bottom)
        })

    match_time = time.time() - match_start

    # Save results if requested
    saving_time = 0
    img_draw = rgb_image.copy()

    if save_output:
        saving_start = time.time()
        output_path = f"faces_{os.path.basename(image_path)}"

        # Draw boxes and labels for each face
        for face in faces:
            left, top, right, bottom = face["location"]

            # Format the label
            if face["confidence"] is not None:
                label = f"{face['name']} ({face['confidence']:.2f})"
            else:
                label = face['name']

            # Draw rectangle around face
            cv2.rectangle(img_draw, (left, top), (right, bottom), (0, 255, 0), 2)

            # Create text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_draw, (left, top - 25), (left + text_size[0], top), (0, 255, 0), -1)

            # Add text
            cv2.putText(img_draw, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Save the image
        cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        saving_time = time.time() - saving_start
        print(f"Detection results saved to {output_path}")

    # Calculate total execution time
    total_time = time.time() - function_start

    # Display timing results
    print(f"\nTIMING REPORT:")
    print(f"Image loading: {img_time:.2f} seconds")
    print(f"EXIF extraction: {exif_time:.2f} seconds")
    if known_face_encodings:
        print(f"Known faces loading: {load_known_time:.2f} seconds")
    print(f"Face detection time: {detection_time:.2f} seconds")
    print(f"Face matching time: {match_time:.2f} seconds")
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
    print(f"\nFACE RECOGNITION RESULTS:")
    print(f"Found {len(faces)} faces:")
    for i, face in enumerate(faces, 1):
        if face["confidence"] is not None:
            print(f"{i}. {face['name']} (confidence: {face['confidence']:.2f})")
        else:
            print(f"{i}. {face['name']} (unknown)")

    # Display the image with OpenCV
    cv2.imshow("Face Recognition", cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return faces, total_time


if __name__ == "__main__":
    # Example usage
    image_path = "example2.jpg"

    # Optional: specify directory with known faces for face recognition
    known_faces_dir = "known_faces"

    detected_faces, processing_time = face_recognition_process(
        image_path,
        known_faces_dir=known_faces_dir if os.path.exists(known_faces_dir) else None,
        save_output=True,
        tolerance=0.6
    )