import streamlit as st
from fastai.vision.all import *
import gdown

st.set_page_config(page_title="Food Classifier", page_icon="🥟", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Mongolian Food Classifier🥟")
st.subheader("""This model simply returns whether the picture you've uploaded is a: 
- Buuz, 
- Tsuivan 
- Potato salad 
- Khuushuur""")
image_file = st.file_uploader("Insert Image File Here", type=['png', 'jpg', 'jpeg'])

model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model..."):
        url = 'https://drive.google.com/u/0/uc?export=download&confirm=b-VY&id=1SVhM6s5PYY4caiNbCUjLR5sKN_6L5tww'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

col1, col2 = st.columns(2)
if image_file is not None:
    img = PILImage.create(image_file)
    pred, pred_idx, probs = learn_inf.predict(img)
    

    with col1:
        st.markdown(f"""#### PREDICTION: 
        - {pred.capitalize()}""")
        st.markdown(f"""#### CONFIDENCE: 
        - {round(max(probs.tolist()), 3) * 100}%""")
        st.balloons()
    with col2:
        st.image(img, width=250)