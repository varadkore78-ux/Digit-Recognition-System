import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# UI
st.title("Handwritten Digit Recognition (Robust Preprocessing)")

canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=20,
    stroke_color="#FFFFFF",  # White ink
    background_color="#000000",  # Black canvas
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

@st.cache_resource
def get_model(path="digit_model.keras"):
    return load_model(path)

def is_canvas_empty(img_rgba, threshold=10):
    """Return True if canvas is empty (no drawing).
       Uses max of RGB channels and alpha presence to decide.
    """
    if img_rgba is None:
        return True
    # img_rgba dtype might be float; convert range to 0-255
    arr = (img_rgba * 255).astype(np.uint8) if img_rgba.max() <= 1.0 else img_rgba.astype(np.uint8)
    # If alpha exists and alpha == 0 everywhere -> empty
    if arr.shape[2] == 4:
        if np.all(arr[:, :, 3] == 0):
            return True
    # else check max intensity (white strokes) over RGB channels
    rgb_max = arr[:, :, :3].max()
    return rgb_max < threshold

def preprocess_canvas(img_rgba):
    """Return a 28x28 single-channel float32 image normalized 0..1 with digit centered.
       Steps:
        - Convert RGBA to grayscale
        - Normalize 0..255
        - Threshold / binary invert heuristics
        - Crop to bounding box of non-zero pixels
        - Resize keeping aspect ratio so largest side -> 20 px
        - Pad to 28x28 and center
    """
    # Convert to uint8 0..255
    arr = (img_rgba * 255).astype(np.uint8) if img_rgba.max() <= 1.0 else img_rgba.astype(np.uint8)
    # If RGBA -> drop alpha but keep composite: convert to RGB first
    if arr.shape[2] == 4:
        # composite on black background using alpha
        alpha = arr[:, :, 3] / 255.0
        for c in range(3):
            arr[:, :, c] = (arr[:, :, c] * alpha + 0 * (1 - alpha)).astype(np.uint8)
        rgb = arr[:, :, :3]
    else:
        rgb = arr[:, :, :3]

    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)  # 0..255 (0 black, 255 white)

    # Heuristic: if background is light (mean > 127), invert so foreground=white strokes -> white pixels
    mean_val = gray.mean()
    if mean_val > 127:
        gray = 255 - gray

    # Binary threshold (make drawing strong)
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box of the drawn region
    coords = cv2.findNonZero(th)
    if coords is None:
        # nothing drawn
        h, w = gray.shape
        blank28 = np.zeros((28, 28), dtype=np.float32)
        return blank28

    x, y, w_box, h_box = cv2.boundingRect(coords)

    # Crop the region with small padding
    padding = int(0.15 * max(w_box, h_box))
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w_box + padding, gray.shape[1])
    y2 = min(y + h_box + padding, gray.shape[0])
    cropped = th[y1:y2, x1:x2]

    # Resize keeping aspect ratio: scale so largest side -> 20
    h_c, w_c = cropped.shape
    if h_c > w_c:
        new_h = 20
        new_w = int(round((w_c / h_c) * 20))
    else:
        new_w = 20
        new_h = int(round((h_c / w_c) * 20))
    if new_w == 0: new_w = 1
    if new_h == 0: new_h = 1
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Make a 28x28 canvas and center the resized image
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Optional: apply a final gaussian blur to smooth, then normalize to 0..1 (float)
    canvas = cv2.GaussianBlur(canvas, (3,3), 0)
    canvas = canvas.astype(np.float32) / 255.0

    return canvas  # shape (28,28) float32, foreground ~1.0

def prepare_input_for_model(img28, model):
    """Given 28x28 float image and a keras model, reshape to expected input."""
    input_shape = model.input_shape  # e.g. (None, 784) or (None, 28,28,1) etc.
    if len(input_shape) == 2:
        # Flattened MLP expecting (None, 784)
        x = img28.reshape(1, 28*28).astype(np.float32)
    elif len(input_shape) == 3:
        # (None, 28, 28)
        x = img28.reshape(1, 28, 28).astype(np.float32)
    else:
        # (None, 28, 28, 1) typical CNN
        x = img28.reshape(1, 28, 28, 1).astype(np.float32)
    return x

if st.button("Predict"):
    img = canvas_result.image_data
    if is_canvas_empty(img):
        st.warning("Please draw a digit before predicting.")
    else:
        model = get_model("digit_model.keras")  # cached load
        # Preprocess
        proc = preprocess_canvas(img)
        # Show processed image for debugging
        st.subheader("Processed image (28x28)")
        st.image((proc * 255).astype(np.uint8), width=140)

        # Prepare for model input
        x = prepare_input_for_model(proc, model)

        preds = model.predict(x)
        probs = preds.ravel()
        top3_idx = probs.argsort()[::-1][:3]

        st.subheader("Top predictions")
        for i in top3_idx:
            st.write(f"{i} — {probs[i]:.4f}")

        predicted_class = int(top3_idx[0])
        st.success(f"🎯 Predicted Digit: **{predicted_class}** (p={probs[predicted_class]:.3f})")