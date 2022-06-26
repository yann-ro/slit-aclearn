import streamlit as st

def display_explanation(kind):
    kind = kind.lower()
    text = '...'

    #sampling
    if kind=='sampling_random':
        text = 'Sampling randomly/uniform correspond to select randomly new samples. It correspond to baseline sampling strategy in active learning.'

    if kind=='sampling_var_ratio':
        text = ''
            
    if kind=='sampling_bald':
        text = ''
    
    if kind=='sampling_max_entropy':
        text = ''
    
    #algorithms
    if kind=='algo_mc_dropout':
        text = 'Prediction incertitude quantified using dropout as Bayesian estimator of weights. <https://arxiv.org/abs/1506.02142>'


    with st.expander("See explanation"): 
        st.markdown(text, unsafe_allow_html=True)