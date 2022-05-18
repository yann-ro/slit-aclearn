import streamlit as st
import os

def setup_window():
    st.header('Dataset')
    dataset_section()
    st.markdown('---')


    st.header('Task')
    task_section()
    st.markdown('---')

    
    st.header('Active Learning Models')
    models_section()
    st.markdown('---')


    _,center,_ = st.columns([5,1,5])
    with center: val_setup = st.button('validate setup')
    if val_setup:
        try:
            if st.session_state['task'] and st.session_state['dataset_data_path'] and st.session_state['ml_algo_1']:
                st.session_state.setup_finished = True
                st.success('setup finished')
        except:
            st.error('missing some elements')



def dataset_section():
    uploaded_data = None
    uploaded_labels = None
    uploaded_unlabeled = None

    if st.checkbox('edit dataset', disabled=st.session_state.setup_finished):
        col, _ = st.columns([1,3])
        with col:
            uploaded_data = st.file_uploader('Choose a file for data', type=['npy'])
            uploaded_labels = st.file_uploader('Choose a file for labels', type=['npy'])

            add_unlabeled = st.checkbox('add unlabeled')
            if add_unlabeled:
                uploaded_unlabeled = st.file_uploader('Choose a file for data unlabeled', type=['npy'])
        
        validate = st.button('validate import dataset')
        if validate:
            if uploaded_data and uploaded_labels is not None:
                st.session_state.dataset_data_path = os.path.join('data', uploaded_data.name)
                save_file(uploaded_data)
                
                st.session_state.dataset_labels_path = os.path.join('data', uploaded_labels.name)
                save_file(uploaded_labels)

                st.success('labeled data imported')
            else:
                st.error('mising data')
            
            if uploaded_unlabeled is not None:
                st.session_state.dataset_data_unlabeled_path = os.path.join('data', uploaded_unlabeled.name)
                save_file(uploaded_unlabeled)
                st.success('unlabeled data imported')
            
            elif add_unlabeled:
                st.error('mising unlabeled data')

    else:
        col1, col2 = st.columns([1,6])
        with st.container():
            with col1:
                st.markdown(f"Dataset :")
                st.markdown(f"Labels :")
                st.markdown(f"Dataset unlabeled :")
            with col2:
                st.markdown(f"<font color='gray'>{st.session_state.dataset_data_path}", unsafe_allow_html=True)
                st.markdown(f"<font color='gray'>{st.session_state.dataset_labels_path}", unsafe_allow_html=True)
                st.markdown(f"<font color='gray'>{st.session_state.dataset_data_unlabeled_path}", unsafe_allow_html=True)

def save_file(file):
    with open(os.path.join('data', file.name), 'wb') as f: 
            f.write(file.getbuffer())


def task_section():
    task_names = ['classification', 'object_detection', 'semantic_segmentation']
    
    if st.checkbox('edit task', disabled=st.session_state.setup_finished):
        st.session_state['task'] = st.selectbox('task', task_names)
            
    else:
        col1, col2 = st.columns([1,6])
        with col1: st.markdown(f"Task :")
        with col2: st.markdown(f"<font color='gray'>{st.session_state.task}", unsafe_allow_html=True)



def models_section():
    
    if st.checkbox('edit models', disabled=st.session_state.setup_finished):
        modify_section_models()
    
    else:
        for i in range(1, st.session_state.n_models+1):
            st.markdown('---')
            cols = st.columns([1,1,1,1])
            with cols[0]: st.markdown(f"**Model {i}** <font color='gray'>(x{st.session_state[f'n_samp_mod_{i}']})", unsafe_allow_html=True)
            with cols[1]: 
                st.markdown(f"**ml algorithm**<br/><font color='gray'>{st.session_state[f'ml_algo_{i}']}", unsafe_allow_html=True)
                with st.expander("See explanation"): st.write("...")
            
            with cols[2]: 
                st.markdown(f"**sampling strategy**<br/><font color='gray'>{st.session_state[f'al_algo_{i}']}", unsafe_allow_html=True)
                with st.expander("See explanation"): st.write("...")
            
            with cols[3]: 
                st.markdown(f"**pre-trained model**<br/><font color='gray'>{st.session_state[f'pre_trained_model_{i}']}", unsafe_allow_html=True)



def modify_section_models():

    left,_ = st.columns([1,3])
    st.markdown('---')
    cols = st.columns([1,1,1,1])

    with left:
        st.session_state.n_models = st.number_input('number of models', min_value=1, max_value=10, value=1, step=1, format='%i')

    if st.session_state.n_models > 0:
        models_cl_names = ['SVC', 'Deep Bayesian Convolutionnal']
        samp_names = ['Random', 'Uncertainity Sampling', 'Bald', 'Var_ratio']
        
        for i in range(1, st.session_state.n_models+1):
            with cols[0]: st.session_state[f'ml_algo_{i}'] = st.selectbox(f'ml algorithm ({i})', models_cl_names)
            with cols[1]: st.session_state[f'al_algo_{i}'] = st.selectbox(f'sampling strategy ({i})', samp_names)
            with cols[2]: st.session_state[f'n_samp_mod_{i}'] = st.slider(f'N samples for variance estimation ({i})', 1, 100)
            with cols[3]: st.session_state[f'pre_trained_model_{i}'] = st.selectbox(f'pre-trained model ({i})', [None])