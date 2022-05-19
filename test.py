from streamlit_drawable_canvas import st_canvas
import streamlit as st
from PIL import Image
import numpy as np

from keras.datasets import mnist

def load_random_mnist(nb_img=1):
    (X_train,_), (_,_) = mnist.load_data()
    return X_train[np.random.randint(X_train.shape[0],size=nb_img)].squeeze()

def set_global_param(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

set_global_param('mnist', load_random_mnist())

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)
    
with st.container():
    canvas_result = st_canvas(
        fill_color='rgba(255, 0, 0, 0.3)',
        stroke_width=2,
        stroke_color='#000',
        background_image=None,
        update_streamlit=True,
        height=300,
        drawing_mode=drawing_mode,
        key="cv",
    )