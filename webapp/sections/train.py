import matplotlib.pyplot as plt
import datatools as dtt
import streamlit as st
import time


def train_window():
    st.title('Train')

    cols4 = st.columns([3,1,1,1])
    with cols4[0]:
        fig = plt.figure(figsize=(2,2))
        
        plt.imshow(st.session_state.mnist, cmap='gray')
        plt.axis('off')
        st.write(fig)

    labels = st.session_state.labels

    with cols4[1]: 
        st.radio('label', labels)
    
    with cols4[2]:
        st.markdown('#\n'*4)
        validate = st.button('validate')
    
    with cols4[3]:
        st.markdown('#\n'*4)
        next_img = st.button('next')

    if validate: 
        st.success('Label sucessfully saved !')
        st.session_state.mnist = dtt.load_dataset.load_random_mnist()
    else:
        st.markdown('#')

    cols3 = st.columns([1,1,1])
    with cols3[1]:
        retrain = st.button('Retrain Model')
    
    if retrain:
        progress_bar = st.progress(0)
        for i in range(1,101):
            time.sleep(0.1)
            progress_bar.progress(i)
        st.success('Model sucessfully retrained !')