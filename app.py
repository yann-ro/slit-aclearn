import streamlit as st
import app

st.set_page_config(layout='wide')
color1,color2 = app.config.setup_color_plot('dark_theme')

app.init_global_parameters()

title = 'Active Learning Algorithm Benchmark Interface'
st.markdown(f'# <center>{title}</center>', unsafe_allow_html=True)
st.markdown('---')


with st.sidebar:
    page = app.sections.nav_bar()

if page == 'setup':
    app.sections.setup.setup_window()      

elif page == 'train':
    app.sections.train.train_window()

elif page == 'plot':
    app.sections.plot.plot_windows()