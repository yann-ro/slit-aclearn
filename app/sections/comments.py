import streamlit as st

def display_explanation(kind):
    text = '...'

    if kind=='sampling_random':
        text = 'Sampling randomly/uniform correspond to select randomly new samples. It correspond to baseline sampling strategy in active learning.'

    if kind=='sampling_var_ratio':
        text = ''
            
    
    if kind=='sampling_bald':
        text = ''
    
    if kind=='max_entropy':
        text = ''
    
    with st.expander("See explanation"): 
        st.markdown(text, unsafe_allow_html=True)