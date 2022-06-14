import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import streamlit as st
import numpy as np
import matplotlib

def plot_windows():
    """
    """
    
    st.title(f'Current results')
    left, right = st.columns([3,2])
    
    with left:
        plot_accuracy()
    
    with right:
        if st.session_state.n_models>0:
            plot_confusion(st.session_state['model_1'])

    cols3 = st.columns([6,1,6])
    save_model = cols3[1].button('Save Model')
    if save_model:
        st.success('Model sucessfully saved !')


def plot_accuracy():
    """
    """
    
    fig1, ax1 = plt.subplots()
    for i in range(1, st.session_state.n_models+1):
        ax1.plot(st.session_state[f'model_{i}'].acc_history, label=f"acc model {i} ({st.session_state[f'al_algo_{i}']})")
    
    plt.title('Accuracy on test set')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    st.write(fig1)



def plot_confusion(model):
    """
    """    
    
    fig2, ax2 = plt.subplots()
    ax2.set_title('Confusion matrix on test set')
    c_1 = matplotlib.colors.colorConverter.to_rgba('white', alpha = 1)
    c_2= matplotlib.colors.colorConverter.to_rgba('tab:blue', alpha = 0.4)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap', [c_1, c_2], 512)

    model.plot_confusion(ax=ax2, cmap=cmap)
    st.pyplot(fig2)