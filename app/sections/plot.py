import matplotlib.pyplot as plt
import streamlit as st

def plot_windows():
    st.title(f'Current results - (Accuracy: {None})')
    
    _, center, _ = st.columns([1,4,1])
    with center:
        fig = plt.figure()
        plt.grid(); plt.title('Results')
        st.write(fig)
    
    cols3 = st.columns([6,1,6])
    save_model = cols3[1].button('Save Model')
    if save_model:
        st.success('Model sucessfully saved !')