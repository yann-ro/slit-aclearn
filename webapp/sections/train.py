from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import datatools as dtt
import streamlit as st
from PIL import Image
import pandas as pd
import time


def train_window():

    st.header('Training parameters')
    oracle = training_parameters_section()
    st.markdown(f'---')
    
    if oracle == 'user':
        st.markdown(f"## Labelling: <font color='gray'>{st.session_state.task}", unsafe_allow_html=True)
        labelling_section()
        st.markdown(f'---')


    big_center = st.columns([3,1,4])
    center = st.columns([2,1,2])
    with big_center[1]:
        if oracle == 'computer':
            nb_epochs = st.number_input('number of epochs', min_value=1, max_value=500, step=1)
    with center[1]:
        retrain = st.button('Retrain Model')

    

    if retrain:
        progress_bar = st.progress(0)
        for i in range(1,101):
            time.sleep(0.1)
            progress_bar.progress(i)
        st.success('Model sucessfully retrained !')



def training_parameters_section():
    modify = st.checkbox('modify parameters')

    cols = st.columns([1,1,1,1,1,2])
    with cols[0]:
        st.markdown('Task')
        st.markdown(f"<font color='gray'>{st.session_state.task}", unsafe_allow_html=True)
    with cols[1]:
        st.markdown('Dataset')
        st.markdown(f"<font color='gray'>{st.session_state.dataset_data_path}", unsafe_allow_html=True)
    with cols[2]:
        oracle = st.selectbox('Oracle', ['computer', 'user'])
    with cols[3]:
        query_size = st.slider('Query size',1,100)
    with cols[4]:
        device = st.selectbox('Device', ['cpu', 'gpu'])

    return oracle



def labelling_section():
    if st.session_state.task == 'classification':
        classification_task()
    elif st.session_state.task == 'object_detection':
        object_detection_task()



def classification_task():
        cols4 = st.columns([2,1,1,1])

        with cols4[0]:
            fig = plt.figure(figsize=(1,1))
            
            plt.imshow(st.session_state.mnist, cmap='gray')
            plt.axis('off')
            st.pyplot(fig)

        labels = st.session_state.labels

        with cols4[1]: 
            st.markdown('#\n'*5)
            st.radio('label', labels)
        
        with cols4[2]:
            st.markdown('#\n'*10)
            validate = st.button('validate')
        
        with cols4[3]:
            st.markdown('#\n'*10)
            next_img = st.button('next')

        if validate: 
            st.success('Label sucessfully saved !')
            st.session_state.mnist = dtt.load_dataset.load_random_mnist()
        else:
            st.markdown('#')


def object_detection_task():
    st.sidebar.title('Labelling tools')

    drawing_mode = st.sidebar.selectbox(
        'Drawing tool', ('rect', 'point', 'freedraw', 'transform')
    )

    class_selected = st.sidebar.selectbox('Class', ('class1','class2', 'class3'))

    stroke_width = 2
    point_display_radius = 2
    bg_image = None
    stroke_color = '#000'
    bg_color = '#eee'
    
    if class_selected =='class1':
        fill_color = 'rgba(255, 0, 0, 0.3)'
    if class_selected =='class2':
        fill_color = 'rgba(0, 255, 0, 0.3)'
    if class_selected =='class3':
        fill_color = 'rgba(0, 0, 255, 0.3)'

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=True,
        height=600,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    #st_canvas(initial_drawing=canvas_result.json_data)

    # if canvas_result.image_data is not None:
    #     st.image(canvas_result.image_data)
    # if canvas_result.json_data is not None:
    #     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    #     for col in objects.select_dtypes(include=['object']).columns:
    #         objects[col] = objects[col].astype("str")
    #     st.dataframe(objects)