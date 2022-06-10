import matplotlib.pyplot as plt
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
    plt.rcParams['legend.facecolor'] = 'gray'

    return color1,color2

def set_global_param(key, value, image=False):
    if key not in st.session_state:
        if image:
            st.session_state[key] = Image.open(value)
        else:
            st.session_state[key] = value


def init_global_parameters():

    set_global_param('dataset_data_unlabeled_path', None)

    set_global_param('task', 'classification')
    set_global_param('oracle', 'computer')
    set_global_param('device', 'cpu')
    set_global_param('query_size', 10)

    set_global_param('n_models', 0)
    set_global_param('setup_finished', False)

    set_global_param('ml_algo_1', None)
    set_global_param('n_epochs', None)
    
    #to debug easily
    set_global_param('dataset_data_path', 'data/mnist_data.npy')
    set_global_param('dataset_labels_path', 'data/mnist_labels.npy')

    set_global_param('image', 'data/ano_metal_nut.png', image=True)

    set_global_param('labels', ['valid','abnormal'])
