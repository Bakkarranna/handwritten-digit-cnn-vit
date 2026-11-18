"""Streamlit app: upload a photo of one handwritten digit, see the prepared
8x8 input and the predictions of both saved models (CNN and ViT).

Run:  streamlit run app.py
No training happens here. Models are loaded from models/*.pt.
"""
import numpy as np
import streamlit as st
import torch
from PIL import Image

from common import load_checkpoint
from preprocess import ImageProblem, prepare_image

st.set_page_config(page_title="Handwritten Digit Prediction", page_icon="🔢", layout="centered")


@st.cache_resource
def get_models():
    cnn, _ = load_checkpoint("cnn")
    vit, _ = load_checkpoint("vit")
    return cnn, vit


def predict(model, arr):
    x = torch.from_numpy(arr).float().reshape(1, 1, 8, 8)    # [1,1,8,8]
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    return int(probs.argmax()), probs


st.title("Handwritten Digit Prediction")
st.caption("CNN vs Vision Transformer, trained on the sklearn 8x8 digits dataset.")

try:
    cnn, vit = get_models()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()
except Exception as e:                                       # corrupt checkpoint etc.
    st.error(f"The saved models could not be loaded: {e}")
    st.stop()

st.write("Upload a clear PNG or JPEG photo with **one dark digit on light paper** "
         "(or a light digit on a dark background).")
up = st.file_uploader("Digit image", type=["png", "jpg", "jpeg"])

if up is None:
    st.info("No image yet. Choose a file to begin.")
    st.stop()

try:
    arr, steps = prepare_image(up, return_steps=True)
except ImageProblem as e:
    st.error(str(e))
    st.stop()
except Exception:
    st.error("Something went wrong while reading this image. Please try another file.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Your photo")
    st.image(steps["original"])
with col2:
    st.subheader("Prepared 8x8 input")
    big = Image.fromarray((arr * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)
    st.image(big, caption="What the models actually see (enlarged, pixel by pixel)")

p_cnn, pr_cnn = predict(cnn, arr)
p_vit, pr_vit = predict(vit, arr)

st.subheader("Predictions")
c1, c2 = st.columns(2)
c1.metric("CNN predicts", p_cnn, f"{pr_cnn[p_cnn] * 100:.1f}% confident", delta_color="off")
c2.metric("ViT predicts", p_vit, f"{pr_vit[p_vit] * 100:.1f}% confident", delta_color="off")
if p_cnn == p_vit:
    st.success(f"Both models agree: {p_cnn}")
else:
    st.warning("The models disagree. Real photos look different from the small training digits, so this can happen.")

with st.expander("Probabilities for each digit"):
    st.bar_chart({"CNN": pr_cnn, "ViT": pr_vit})
