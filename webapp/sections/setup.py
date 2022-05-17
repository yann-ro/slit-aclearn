import streamlit as st

def setup_window():
    st.title('Dataset')
    dataset_section()
    st.markdown('---')


    st.title('Task')
    task_section()
    st.markdown('---')

    
    st.title('Active Learning Models')
    models_section()
    st.markdown('---')
    
    _,center,_ = st.columns([5,1,5])
    with center: val_setup = st.button('validate setup')
    if val_setup:
        try:
            if st.session_state['task'] and st.session_state['dataset_data'] and st.session_state['ml_algo_1']:
                st.session_state.setup_finished = True
                st.success('setup finished')
        except:
            st.error('missing some elements')



def dataset_section():
    uploaded_data = None
    uploaded_labels = None
    uploaded_unlabeled = None

    if st.checkbox('modify dataset', disabled=st.session_state.setup_finished):
        uploaded_data = st.file_uploader('Choose a file for data')
        uploaded_labels = st.file_uploader('Choose a file for labels')

        add_unlabeled = st.checkbox('add unlabeled')
        if add_unlabeled:
            uploaded_unlabeled = st.file_uploader('Choose a file for data unlabeled')
        
        validate = st.button('validate import dataset')
        if validate:
            if uploaded_data and uploaded_labels is not None:
                st.session_state.dataset_data = uploaded_data.name
                st.session_state.dataset_labels = uploaded_labels.name
                st.success('data imported')
            else:
                st.error('mising data')
            
            if uploaded_unlabeled is not None:
                st.session_state.dataset_data_unlabeled = uploaded_unlabeled.name
                st.success('unlabeled data imported')
            elif add_unlabeled:
                st.error('mising data unlabeled')

    else:
        with st.container():
            st.text(f'Dataset :\t\t\t{st.session_state.dataset_data}')
            st.text(f'Labels :\t\t\t{st.session_state.dataset_labels}')
            st.text(f'Dataset unlabeled :\t\t{st.session_state.dataset_data_unlabeled}')


def task_section():
    task_names = ['classification', 'object_detection', 'semantic_segmentation']
    
    if st.checkbox('modify task', disabled=st.session_state.setup_finished):
        st.session_state['task'] = st.selectbox('task', task_names)
            
    else:
        st.text(f'Task :\t\t\t\t{st.session_state.task}')


def models_section():
    
    if st.checkbox('modify models', disabled=st.session_state.setup_finished):
        modify_section_models()
    
    else:
        for i in range(1, st.session_state.n_models+1):
            st.markdown('---')
            cols = st.columns([1,1,1,1])
            with cols[0]: st.text(f"model {i} (x{st.session_state[f'n_samp_mod_{i}']})")
            with cols[1]: st.text(f"machine learning algorithm:\n{st.session_state[f'ml_algo_{i}']}")
            with cols[2]: st.text(f"sampling strategy:\n{st.session_state[f'al_algo_{i}']}")
            with cols[3]: st.text(f"pre-trained model:\n{st.session_state[f'pre_trained_model_{i}']}")



def modify_section_models():

    cols = st.columns([1,1,1,1])

    st.session_state.n_models = st.number_input('number of models', min_value=1, max_value=10, value=1, step=1, format='%i')

    if st.session_state.n_models > 0:
        models_cl_names = ['SVC', 'Deep Bayesian Convolutionnal']
        samp_names = ['Random', 'Uncertainity Sampling', 'Bald', 'Var_ratio']
        
        for i in range(1, st.session_state.n_models+1):
            with cols[0]: st.session_state[f'ml_algo_{i}'] = st.selectbox(f'machine learning algorithm ({i})', models_cl_names)
            with cols[1]: st.session_state[f'al_algo_{i}'] = st.selectbox(f'Sampling strategy ({i})', samp_names)
            with cols[2]: st.session_state[f'n_samp_mod_{i}'] = st.slider(f'N samples for variance estimation ({i})', 1, 100)
            with cols[3]: st.session_state[f'pre_trained_model_{i}'] = st.selectbox(f'pre-trained model ({i})', [None])