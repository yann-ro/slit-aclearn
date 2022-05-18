import streamlit as st
import webapp as wp

st.set_page_config(layout='wide')
color1,color2 = wp.config.setup_color_plot('dark_theme')

wp.init_global_parameters()

title = 'Active Learning Algorithm Benchmark Interface'
st.markdown(f'# <center>{title}</center>', unsafe_allow_html=True)
st.markdown('---')

with st.sidebar:
    page = wp.sections.nav_bar()


if page == 'setup':
    wp.sections.setup.setup_window()      

elif page == 'train':
    wp.sections.train.train_window()

elif page == 'plot':
    wp.sections.plot.plot_windows()

# if page == 'test':
#   wp.sections.train.object_detection_task()