import streamlit as st


def display_explanation(kind):
    kind = kind.lower()
    text = "..."

    # sampling
    if kind == "sampling_random":
        text = "Sampling randomly/uniform correspond to select randomly new points. It correspond to baseline sampling strategy in active learning."

    if kind == "sampling_var_ratio":
        text = "Sampling points that would minimize output variance, which is one of the components of error."

    if kind == "sampling_bald":
        text = "Sampling points that are expected to maximise the information gained about the model parameters, i.e. maximise the mutual information between predictions and model posterior. <https://arxiv.org/abs/1112.5745>"

    if kind == "sampling_max_entropy":
        text = "Sampling points that maximise the predictive entropy"

    # algorithms
    if kind == "algo_mc_dropout":
        text = "Prediction incertitude quantified using dropout as Bayesian estimator of weights. <https://arxiv.org/abs/1506.02142>"

    with st.expander("details"):
        st.markdown(text, unsafe_allow_html=True)
