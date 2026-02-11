import streamlit as st
import numpy as np
import pandas as pd
import pickle
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, regularizers

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="HerHealth AI | Ovarian Cancer Diagnosis",
    layout="wide"
)

# =====================================================
# UI THEME
# =====================================================
st.markdown("""
<style>
.stApp { background-color: #f5f9ff; }
.header-box {
    background: linear-gradient(90deg, #0f4c81, #1f77b4);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
}
.step-card, .input-card {
    background: white;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
}
.step-card { height: 140px; }
.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 12px;
    font-size: 17px;
    height: 3.2em;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header-box">
    <h1>HerHealth AI</h1>
    <h4>Multimodal Explainable Ovarian Cancer Diagnosis</h4>
    <p>Clinician & Researcher Decision-Support Platform</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# CONSTANTS
# =====================================================
image_classes = ["Clear Cell", "Endometri", "Mucinous", "Non-Cancerous", "Serous"]
NON_CANCER_INDEX = 3
IMG_SIZE = (227, 227)

# =====================================================
# MODEL DEFINITIONS
# =====================================================
def create_kk_net(input_shape=(227,227,3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(32,(3,3),activation='elu',padding='same')(inputs)
    x = layers.Conv2D(32,(3,3),activation='elu',padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64,(3,3),activation='elu',padding='same')(x)
    x = layers.Conv2D(64,(3,3),activation='elu',padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128,(3,3),activation='elu',padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    last_conv = layers.Conv2D(
        256,(3,3),activation='elu',
        padding='same',name="last_conv_layer"
    )(x)

    x = layers.MaxPooling2D()(last_conv)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32,activation='elu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(16,activation='elu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes,activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

@st.cache_resource
def load_image_model():
    model = create_kk_net()
    model.load_weights("kk_net_best.h5")
    return model

@st.cache_resource
def load_text_model():
    with open("ovarian_cancer_model.pkl","rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"], obj["features"]

image_model = load_image_model()
text_model, text_scaler, all_features = load_text_model()

# =====================================================
# UTILITIES
# =====================================================
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def weighted_fusion(img_p, clin_p):
    return 0.6 * img_p + 0.4 * clin_p

def make_gradcam_heatmap(img_array, model):
    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer("last_conv_layer").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    return heatmap / heatmap.max()

def overlay_heatmap(img, heatmap):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# =====================================================
# USER MODE
# =====================================================
user_mode = st.radio(
    "Select User Mode",
    ["👩‍⚕️ Clinician", "🔬 Researcher"],
    horizontal=True
)

# =====================================================
# CLINICIAN MODE
# =====================================================
if user_mode == "👩‍⚕️ Clinician":

    st.markdown("## 🧪 Clinician Mode")
    left, right = st.columns(2)

    with left:
        uploaded_img = st.file_uploader("Upload Histopathology Image", type=["jpg","png","jpeg"])

    with right:
        HE4 = st.number_input("HE4", 0.0)
        CA125 = st.number_input("CA-125", 0.0)
        CEA = st.number_input("CEA", 0.0)
        NEU = st.number_input("NEU", 0.0)
        PLT = st.number_input("PLT", 0.0)

    run = st.button("🔍 Run Diagnosis", use_container_width=True)

    if run and uploaded_img:
        img_pil = Image.open(uploaded_img).convert("RGB")
        img_array = preprocess_image(img_pil)
        preds = image_model(img_array).numpy()[0]

        image_prob = 1 - preds[NON_CANCER_INDEX]
        image_class = image_classes[int(np.argmax(preds))]

        full_input = {f:0 for f in all_features}
        full_input.update({"HE4":HE4,"CA125":CA125,"CEA":CEA,"NEU":NEU,"PLT":PLT})

        X = text_scaler.transform(pd.DataFrame([full_input]))
        clinical_prob = text_model.predict_proba(X)[0][1]

        # Clinical-only prediction
        clinical_diagnosis = "Cancerous" if clinical_prob >= 0.5 else "Non-Cancerous"
        clinical_risk = (
            "Low" if clinical_prob < 0.33 else
            "Medium" if clinical_prob < 0.66 else
            "High"
        )

        final_prob = weighted_fusion(image_prob, clinical_prob)

        risk = (
            "Low" if final_prob < 0.33 else
            "Medium" if final_prob < 0.66 else
            "High"
        )

        st.markdown("### 📊 Diagnosis Results")
        a,b,c,d = st.columns(4)

        a.metric("Final Cancer Probability", f"{final_prob:.2%}")
        b.metric("Final Risk Level", risk)
        c.metric("Image Model Probability", f"{image_prob:.2%}")
        d.metric("Clinical Model Probability", f"{clinical_prob:.2%}")

        st.markdown("### 🧪 Individual Model Diagnosis")
        m1, m2 = st.columns(2)
        m1.info(f"Image Model: {'Cancerous' if image_prob >= 0.5 else 'Non-Cancerous'}")
        m2.info(f"Clinical Model: {clinical_diagnosis} ({clinical_risk} Risk)")

        if final_prob >= 0.5:
            st.success("Final Diagnosis: Cancerous")
            st.success(f"Detected Cancer Type: {image_class}")
        else:
            st.success("Final Diagnosis: Non-Cancerous")

        st.markdown("### 🧠 Grad-CAM Explainability")
        heatmap = make_gradcam_heatmap(img_array, image_model)
        overlay = overlay_heatmap(img_pil, heatmap)
        x,y = st.columns(2)
        x.image(img_pil, caption="Original Image", use_container_width=True)
        y.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
# =====================================================
# RESEARCHER MODE (YOUR ORIGINAL VERSION)
# =====================================================
if user_mode == "🔬 Researcher":

    st.markdown("## 🔬 Researcher Mode: Batch Uncertainty Analysis")

    csv_file = st.file_uploader("Upload Clinical CSV", type=["csv"])
    img_files = st.file_uploader(
        "Upload Histopathology Images (multiple)",
        type=["jpg","png","jpeg"],
        accept_multiple_files=True
    )

    if st.button("📊 Run Research Analysis") and csv_file and img_files:

        df = pd.read_csv(csv_file)
        results = []

        for i, img_file in enumerate(img_files):

            img = Image.open(img_file).convert("RGB")
            img_array = preprocess_image(img)
            preds = image_model(img_array).numpy()[0]

            image_prob = 1 - preds[NON_CANCER_INDEX]
            image_class = image_classes[int(np.argmax(preds))]

            row = df.iloc[min(i, len(df)-1)].to_dict()
            full_input = {f: row.get(f, 0) for f in all_features}

            X = text_scaler.transform(pd.DataFrame([full_input]))
            clinical_prob = text_model.predict_proba(X)[0][1]

            final_prob = weighted_fusion(image_prob, clinical_prob)

            diagnosis = "Cancerous" if final_prob >= 0.5 else "Non-Cancerous"
            risk_level = (
                "Low" if final_prob < 0.33 else
                "Medium" if final_prob < 0.66 else
                "High"
            )

            agreement = 1 - abs(image_prob - clinical_prob)

            results.append({
                "Image_Name": img_file.name,
                "Diagnosis": diagnosis,
                "Risk_Level": risk_level,
                "Predicted_Class": image_class,
                "Image_Prob": image_prob,
                "Clinical_Prob": clinical_prob,
                "Final_Prob": final_prob,
                "Agreement_Score": agreement
            })

        result_df = pd.DataFrame(results)

        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            "Download Results CSV",
            result_df.to_csv(index=False),
            "research_results.csv"
        )