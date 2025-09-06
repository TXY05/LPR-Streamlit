import os
import glob
import time
import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms
# ---------------------------
# Cascade Classifier
# ---------------------------
CASCADE_PATH = "model/haar/cascade.xml"  # update path
cascade = cv2.CascadeClassifier(CASCADE_PATH)
if cascade.empty():
    st.error(f"Failed to load cascade from {CASCADE_PATH}")
    st.stop()

# ---------------------------
# Faster R-CNN helpers
# ---------------------------
def collect_all_images(dir_test):
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images

@st.cache_resource
def load_fasterrcnn(weights_path, model_name, device):
    checkpoint = torch.load(weights_path, map_location=device,weights_only=False)
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

# ---------------------------
# YOLO helpers
# ---------------------------
YOLO_WEIGHTS = "model/yolo/best.pt"

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_WEIGHTS)

# ---------------------------
# Streamlit App
# ---------------------------
st.title("License Plate Detection")

# Sidebar controls
with st.sidebar.expander("Cascade Classifier"):
    scaleFactor = st.slider("Scale Factor", 1.1, 2.0, 1.1, 0.1)
    minNeighbors = st.slider("Min Neighbors", 1, 10, 5, 1)
    minSize_w = st.slider("Min Width", 10, 200, 50, 10)
    minSize_h = st.slider("Min Height", 10, 200, 20, 10)

with st.sidebar.expander("Faster R-CNN"):
    frcnn_thresh = st.slider("Detection threshold", 0.1, 1.0, 0.3, 0.05)

with st.sidebar.expander("YOLO"):
    yolo_thresh = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

# Load models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FASTER_PATH = "model/faster-rcnn/best_model.pth"
FASTER_MODEL = "fasterrcnn_resnet50_fpn_v2"

if os.path.exists(FASTER_PATH):
    faster_model, CLASSES = load_fasterrcnn(FASTER_PATH, FASTER_MODEL, device)
    FASTER_COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
else:
    st.error(f"Faster R-CNN weights not found at {FASTER_PATH}")
    st.stop()

if os.path.exists(YOLO_WEIGHTS):
    yolo_model = load_yolo()
else:
    st.error(f"YOLO weights not found at {YOLO_WEIGHTS}")
    st.stop()

# File uploader
uploaded_files = st.file_uploader(
    "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    # Preload all images into memory once
    images_data = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            images_data.append((uploaded_file.name, img))
        else:
            st.warning(f"Could not decode {uploaded_file.name}")

    # =======================
    # Cascade Section
    # =======================
    st.header("Cascade Classifier Results")
    for name, image in images_data:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        detections = cascade.detectMultiScale(
            gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
            minSize=(minSize_w, minSize_h)
        )
        fps = 1 / (time.time() - start_time)

        img_copy = image.copy()
        for (x, y, w, h) in detections:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        st.image(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB),
                 caption=f"{name} | {len(detections)} detections | FPS: {fps:.2f}",
                 use_container_width=True)

    # =======================
    # Faster R-CNN Section
    # =======================
    st.header("Faster R-CNN Results")
    for name, image in images_data:
        img_copy = image.copy()
        rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        rgb = infer_transforms(rgb)
        rgb = torch.unsqueeze(rgb, 0)

        start_time = time.time()
        with torch.no_grad():
            outputs = faster_model(rgb.to(device))
        fps = 1 / (time.time() - start_time)

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) != 0:
            img_copy = inference_annotations(outputs, frcnn_thresh, CLASSES, FASTER_COLORS, img_copy)

        st.image(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB),
                 caption=f"{name} | FPS: {fps:.2f}",
                 use_container_width=True)

    # =======================
    # YOLO Section
    # =======================
    st.header("YOLO Results")
    for name, image in images_data:
        start_time = time.time()
        results = yolo_model.predict(image, conf=yolo_thresh)
        fps = 1 / (time.time() - start_time)

        yolo_img = results[0].plot()

        st.image(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB),
                 caption=f"{name} | FPS: {fps:.2f}",
                 use_container_width=True)

