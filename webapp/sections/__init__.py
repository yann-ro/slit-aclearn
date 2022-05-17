from .plot import *
from .setup import *
from .train import *
import streamlit as st

def nav_bar():
    st.title('Navigation')
    return st.radio('Slect mode',['setup','train','plot'], disabled=not(st.session_state.setup_finished))