import glob
import os
import cv2
import time
import torch
import numpy as np
import streamlit as st

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms


# Function to collect all image paths
def collect_all_images(dir_test):
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images


# Cache model loading to avoid reloading each time
@st.cache_resource
def load_model(weights_path, model_name, device):
    checkpoint = torch.load(weights_path, map_location=device)
    NUM_CLASSES = checkpoint['config']['NC']
    CLASSES = checkpoint['config']['CLASSES']

    try:
        build_model = create_model[str(model_name)]
    except:
        build_model = create_model[checkpoint['model_name']]

    model = build_model(num_classes=NUM_CLASSES, coco_model=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device).eval()
    return model, CLASSES


# --------------------------
# Streamlit App
# --------------------------
st.title("Faster R-CNN Object Detection")

# FIXED model path and name
FIXED_WEIGHTS = "best_model.pth"   # put your actual weights file path here
FIXED_MODEL = "fasterrcnn_resnet50_fpn_v2"

# Only let user adjust threshold
threshold = st.sidebar.slider("Detection threshold", 0.1, 1.0, 0.3, 0.05)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
if os.path.exists(FIXED_WEIGHTS):
    model, CLASSES = load_model(FIXED_WEIGHTS, FIXED_MODEL, device)
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
else:
    st.error(f"Model weights not found at {FIXED_WEIGHTS}")
    st.stop()

# File uploader
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    OUT_DIR = set_infer_dir()

    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        orig_image = image.copy()

        # Preprocess
        rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        rgb = infer_transforms(rgb)
        rgb = torch.unsqueeze(rgb, 0)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(rgb.to(device))
        fps = 1 / (time.time() - start_time)

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:
            annotated = inference_annotations(outputs, threshold, CLASSES, COLORS, orig_image)
        else:
            annotated = orig_image

        # Save + show result
        save_path = os.path.join(OUT_DIR, f"{uploaded_file.name}.jpg")
        cv2.imwrite(save_path, annotated)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Processed {uploaded_file.name}", use_column_width=True)
        st.write(f"FPS: {fps:.2f}")

import os
import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Fixed weights file
WEIGHTS_PATH = "model/yolo/best.pt"  # make sure this file exists in your project folder

# Cache model loading
@st.cache_resource
def load_model():
    model = YOLO(WEIGHTS_PATH)
    return model

# Streamlit App
st.title("YOLO Object Detection")

# Sidebar controls
threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

# Load YOLO model
if os.path.exists(WEIGHTS_PATH):
    model = load_model()
else:
    st.error(f"Model weights not found at {WEIGHTS_PATH}. Please add the file.")
    st.stop()

# File uploader
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Inference
        start_time = time.time()
        results = model.predict(image, conf=threshold)
        fps = 1 / (time.time() - start_time)

        # Annotated image
        annotated = results[0].plot()  # numpy array (BGR)

        # Display result
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption=f"Processed {uploaded_file.name}", use_column_width=True)
        st.write(f"FPS: {fps:.2f}")

