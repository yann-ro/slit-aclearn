from .plot import *
from .setup import *
from .train import *
import streamlit as st

def nav_bar():
    st.title('Navigation')
    page = st.radio('Slect mode',['setup','train','plot'], disabled=not(st.session_state.setup_finished))
    #page = st.radio('',['setup','train','plot', 'test'])
    st.sidebar.markdown('---')
    return page