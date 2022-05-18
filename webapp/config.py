import matplotlib.pyplot as plt
import datatools as dtt
import streamlit as st
from PIL import Image

def setup_color_plot(theme):
    
    if theme == 'dark_theme':
        color1 = 'None'
        color2 = 'white'

    elif theme == 'light_theme':
        color1 = 'None'
        color2 = 'black'
    
    
    plt.rcParams['figure.facecolor'] = color1
    plt.rcParams['axes.facecolor'] = color1
    plt.rcParams['axes.edgecolor'] = color2
    plt.rcParams['axes.labelcolor'] = color2
    plt.rcParams['axes.titlecolor'] = color2
    plt.rcParams['ytick.labelcolor'] = color2
    plt.rcParams['xtick.labelcolor'] = color2

    return color1,color2

def set_global_param(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def init_global_parameters():
    parameters = ['dataset_data_path','dataset_labels_path', 
                  'dataset_data_unlabeled_path']
    for param in parameters:
        set_global_param(param, None)

    set_global_param('task', 'object_detection')
    set_global_param('oracle', 'user')
    set_global_param('device', 'cpu')
    set_global_param('query_size', 1)

    set_global_param('n_models', 0)
    set_global_param('setup_finished', False)

    set_global_param('mnist', dtt.load_dataset.load_random_mnist())
    set_global_param('labels', ['0','1','2','3','4','5','6','7','8','9'])

    bg_image = 'data/ano_metal_nut.png'
    set_global_param('image', Image.open(bg_image))