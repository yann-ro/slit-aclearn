from cProfile import label
import matplotlib.pyplot as plt
import streamlit as st

def plot_windows():
    st.title(f'Current results')
    
    _, center, _ = st.columns([1,4,1])
    with center:
        fig = plt.figure()
        for i in range(1, st.session_state.n_models+1):
            plt.plot(st.session_state[f'model_{i}'].acc_history, label=f"acc model {i} ({st.session_state[f'al_algo_{i}']})")
        
        plt.title('Accuracy on test set')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend()
        st.write(fig)
    
    cols3 = st.columns([6,1,6])
    save_model = cols3[1].button('Save Model')
    if save_model:
        st.success('Model sucessfully saved !')