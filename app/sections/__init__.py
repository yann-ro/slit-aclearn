from .plot import *
from .setup import *
from .train import *
from .comments import *
import streamlit as st

def nav_bar():
    st.title('Navigation')
    page = st.radio('Slect mode',['setup','train','plot'], disabled=not(st.session_state.setup_finished))
    st.sidebar.markdown('---')
    return page