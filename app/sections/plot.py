import matplotlib.pyplot as plt
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
    
    if st.session_state.n_models>0:
        plot_confusion()
        

    cols3 = st.columns([6,1,6])
    save_model = cols3[1].button('Save Model')
    if save_model:
        st.success('Model sucessfully saved !')


def plot_accuracy():
    """
    """
    
    fig, ax = plt.subplots()
    for i in range(1, st.session_state.n_models+1):
        ax.plot(st.session_state[f'model_{i}'].acc_history, label=f"acc model {i} ({st.session_state[f'al_algo_{i}']})")
    
    plt.title('Accuracy on test set')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    st.write(fig)



def plot_confusion():
    """
    """    
    n = nrows=st.session_state.n_models
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(5*n, 4))
    
    for i in range(n):

        ax[i].set_title('Confusion matrix on test set')
        c_1 = matplotlib.colors.colorConverter.to_rgba('black', alpha = 1)
        c_2= matplotlib.colors.colorConverter.to_rgba(colors[i], alpha = 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap', [c_1, c_2], 512)

        st.session_state[f'model_{i+1}'].plot_confusion(ax=ax[i], cmap=cmap)
    st.pyplot(fig)