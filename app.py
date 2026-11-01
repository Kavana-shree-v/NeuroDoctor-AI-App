import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------------
# 1. Page Config & CSS Theme
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI - AI Brain Tumor Visualizer",
    page_icon="🧠",
    layout="wide"
)

# Dark Blue CSS Injection
st.markdown("""
<style>
    .stApp {
        background-color: #0b132b !important;
        color: #e0e6ed !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1c2541 !important;
        border-right: 1px solid #3a506b !important;
    }
    h1, h2, h3 {
        color: #4cc9f0 !important;
    }
    p, label {
        color: #e0e6ed !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. Model & Pipeline Logic
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_densenet121_mri.h5')

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image):
    img = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    resized_img = cv2.resize(enhanced_img, (224, 224), interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    normalized_img = rgb_img.astype(np.float32) / 255.0
    return np.expand_dims(normalized_img, axis=0), normalized_img

def generate_gradcam(model, img_array):
    base_model = None
    for layer in model.layers:
        if 'densenet' in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        base_model = model

    last_conv_layer_name = base_model.layers[-1].name
    grad_model = tf.keras.models.Model(
        inputs=[base_model.inputs],
        outputs=[base_model.get_layer(last_conv_layer_name).output]
    )

    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_array)
        tape.watch(conv_outputs)
        x = conv_outputs
        for layer in model.layers[model.layers.index(base_model) + 1:]:
            x = layer(x)
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), pred_index.numpy(), predictions[0].numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_colors = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_colors = cv2.cvtColor(jet_colors, cv2.COLOR_BGR2RGB)
    orig_uint8 = np.uint8(255 * original_img)
    return cv2.addWeighted(orig_uint8, 1 - alpha, jet_colors, alpha, 0)

# ------------------------------------------------------------------
# 3. Web Interface Layout
# ------------------------------------------------------------------
st.title("🧠 NeuroScan AI: AI Brain Tumor Detection")
st.write("Deep Learning MRI Analysis powered by Fine-Tuned DenseNet121 & Grad-CAM Explainability")
st.divider()

st.sidebar.header("📁 MRI Input Panel")
uploaded_file = st.sidebar.file_uploader("Upload Brain MRI Scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_batch, orig_norm_img = preprocess_image(image)
    
    heatmap, pred_idx, probabilities = generate_gradcam(model, img_batch)
    gradcam_overlay = overlay_gradcam(orig_norm_img, heatmap)

    predicted_class = class_names[pred_idx].upper()
    confidence = probabilities[pred_idx] * 100

    st.success(f"### Predicted Diagnosis: {predicted_class} | Confidence: {confidence:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Uploaded MRI Scan")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔥 Grad-CAM XAI Heatmap")
        st.image(gradcam_overlay, use_container_width=True)

    st.divider()
    st.subheader("📊 Confidence Breakdown")
    for idx, name in enumerate(class_names):
        prob = float(probabilities[idx])
        st.write(f"**{name.capitalize()}**: `{prob * 100:.2f}%`")
        st.progress(prob)
else:
    st.info("👈 Please upload a brain MRI image file from the sidebar panel to start analysis.")
