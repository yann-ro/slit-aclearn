from aclearn.dataset import AcLearnDataset
from aclearn.model import AcLearnModel
import streamlit as st
import torch
import time
import copy
import app
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
        init_models()



def dataset_section():
    uploaded_data = None
    uploaded_labels = None
    uploaded_unlabeled = None

    if st.checkbox('edit dataset', disabled=st.session_state.setup_finished):
        left, center, right = st.columns([1,1,1])
        with left: 
            uploaded_data = st.file_uploader('Choose a file for data', type=['npy'], accept_multiple_files=False)
        with center: 
            uploaded_labels = st.file_uploader('Choose a file for labels', type=['npy'], accept_multiple_files=False)
        with right:
            add_unlabeled = st.checkbox('add unlabeled')
            if add_unlabeled:
                uploaded_unlabeled = st.file_uploader('Choose a file for data unlabeled', type=['npy'])
        
        if  st.button('validate import dataset'):
            
            if uploaded_data and uploaded_labels is not None:
                st.session_state.dataset_data_path = os.path.join('data', uploaded_data.name)
                save_file(uploaded_data)
                
                st.session_state.dataset_labels_path = os.path.join('data', uploaded_labels.name)
                save_file(uploaded_labels)

                st.session_state['dataset'] = AcLearnDataset(st.session_state.dataset_data_path,
                                                            st.session_state.dataset_labels_path, 
                                                            size_init_per_class=2)
                
                st.session_state['labels'] = st.session_state['dataset'].classes

                st.success('data & labels imported')
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
    task_names = ['object_detection', 'classification', 'semantic_segmentation']
    edit_task = st.checkbox('edit task', disabled=st.session_state.setup_finished)
    
    if edit_task:
        col1, col2 = st.columns([1,6])
        with col1: st.session_state['task'] = st.selectbox('task', task_names)
            
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
                st.markdown(f"**ml algorithm (only MC_dropout)**<br/><font color='gray'>{st.session_state[f'ml_algo_{i}']}", unsafe_allow_html=True)
                with st.expander("See explanation"): st.write("...")
            
            with cols[2]: 
                st.markdown(f"**sampling strategy**<br/><font color='gray'>{st.session_state[f'al_algo_{i}']}", unsafe_allow_html=True)
                with st.expander("See explanation"): st.write("...")



def modify_section_models():

    left, center, right, _ = st.columns([1,1,1,1])
    st.markdown('---')
    cols = st.columns([1,1,1,1])

    with left:
        st.session_state.n_models = st.number_input('number of models', min_value=0, max_value=10, value=1, step=1, format='%i')
    with center:
        st.session_state['device'] = st.selectbox(f'Device', torch.cuda.is_available()*['cuda']+['cpu'])
    with right:
        st.session_state['fixed_data_init'] = st.checkbox('fixed data init', value=True)

    if st.session_state.n_models > 0:
        models_cl_names = ['MC_dropout']
        samp_names = ['Random', 'Max_entropy', 'Bald', 'Var_ratio']

        for i in range(1, st.session_state.n_models+1):
            with cols[0]: st.markdown(f"**Model {i}**)", unsafe_allow_html=True)
            with cols[1]: st.session_state[f'ml_algo_{i}'] = st.selectbox(f'ml algorithm ({i}) (only MC_dropout)', models_cl_names)
            with cols[2]: st.session_state[f'al_algo_{i}'] = st.selectbox(f'sampling strategy ({i})', samp_names).lower()
            with cols[3]: st.session_state[f'n_samp_mod_{i}'] = st.number_input(f'N samples for variance estimation ({i})', 
                                                                                min_value=1, 
                                                                                max_value=100)



def init_models():
        
        if not st.session_state.setup_finished:
            
            if st.session_state['task'] and st.session_state['dataset_data_path']:
        
                st.session_state.setup_finished = True

                st.session_state['model_0'] = AcLearnModel('random',
                                                            copy.deepcopy(st.session_state['dataset']),
                                                            model_id = 'model_0',
                                                            device = st.session_state['device'])

                print('$(model_0) evaluate max accuracy...')
                st.session_state['model_0'].evaluate_max()
                print('$(model_0) max accuracy calculated')

                for i in range(1, st.session_state.n_models+1):
            
                    if st.session_state[f'n_samp_mod_{i}']>1:
                        for j in range(st.session_state[f'n_samp_mod_{i}']):
                            
                            if st.session_state['fixed_data_init']:
                                st.session_state[f'dataset_{i}.{j}'] = copy.deepcopy(st.session_state['dataset'])
                            
                            else:
                                st.session_state[f'dataset_{i}.{j}'] = AcLearnDataset(st.session_state.dataset_data_path,
                                                                                      st.session_state.dataset_labels_path, 
                                                                                      size_init_per_class=2)
                            
                            st.session_state[f'model_{i}.{j}'] = AcLearnModel(st.session_state[f'al_algo_{i}'],
                                                                              st.session_state[f'dataset_{i}.{j}'],
                                                                              model_id = f"model_{i}.{j}_{st.session_state[f'al_algo_{i}']}",
                                                                              device = st.session_state['device'])
                            
                            st.session_state[f'model_{i}.{j}'].init_training()
                    else:
                        if st.session_state['fixed_data_init']:
                            st.session_state[f'dataset_{i}'] = copy.deepcopy(st.session_state['dataset'])
                        else:
                            st.session_state[f'dataset_{i}'] = AcLearnDataset(st.session_state.dataset_data_path,
                                                                                    st.session_state.dataset_labels_path, 
                                                                                    size_init_per_class=2)

                        st.session_state[f'model_{i}'] = AcLearnModel(st.session_state[f'al_algo_{i}'],
                                                                      st.session_state[f'dataset_{i}'],
                                                                      model_id = f"model_{i}_{st.session_state[f'al_algo_{i}']}",
                                                                      device = st.session_state['device'])
                        
                        st.session_state[f'model_{i}'].init_training()

                st.success('setup finished')
                st.experimental_rerun()
    
            else:
                st.error('missing some elements')
            
        
        
        

