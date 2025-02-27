from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
from rembg import remove  # Ensure rembg is installed

# Set temporary storage to D:\temp
os.environ["TMP"] = "D:\\temp"
os.environ["TEMP"] = "D:\\temp"

if not os.path.exists("D:\\temp"):
    os.makedirs("D:\\temp")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

PROCESSED_FOLDER = "D:\\temp"
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def resize_image(image, max_dim=1000):
    """Resize image while maintaining aspect ratio (No Quality Loss)"""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

@app.route("/remove_bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_path = os.path.join(PROCESSED_FOLDER, "uploaded.png")
    file.save(image_path)

    processed_path = remove_background(image_path)

    return send_file(processed_path, mimetype="image/png")

def remove_background(image_path):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Resize image before processing (reduce RAM usage)
    image = resize_image(image, max_dim=1000)

    # Remove background with improved settings
    result = remove(image, 
                    alpha_matting=True, 
                    alpha_matting_foreground_threshold=250, 
                    alpha_matting_background_threshold=10, 
                    alpha_matting_erode_size=1, 
                    alpha_matting_base_size=1000)

    # Convert to grayscale for better edge detection
    gray = cv2.cvtColor(result, cv2.COLOR_BGRA2GRAY)

    # Apply Laplacian sharpening for improved edge clarity
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(gray - 0.5 * laplacian)

    # Merge sharpened edges with transparent background
    b, g, r, a = cv2.split(result)
    final_image = cv2.merge((b, g, r, a))

    # Save output
    output_path = os.path.join(PROCESSED_FOLDER, "processed.png")
    cv2.imwrite(output_path, final_image)

    return output_path

if __name__ == "__main__":
    app.run(debug=True)
