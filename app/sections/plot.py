import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib


def plot_windows():
    """
    """
    
    st.title(f'Current results')
    left, right = st.columns([3,2])
    
    with left:
        plot_accuracy()
        plot_confusion()
    
    with right:
        
        models = [f'model_{i+1}' if st.session_state[f'n_samp_mod_{i+1}']==1 else f'model_{i+1}.0' for i in range(st.session_state.n_models)]
        model_name = st.selectbox('Select model', models)
        
        plot_2D_feature_space(model_name)


    cols3 = st.columns([6,1,6])
    save_model = cols3[1].button('Save Model')
    if save_model:
        st.success('Model sucessfully saved !')


def plot_accuracy():
    """
    """
    
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ax.axhline(y=st.session_state['model_0'].max_accuracy,
               linestyle='--', c='r', label='max accuracy (full dataset)', alpha=0.5)

    for i in range(1, st.session_state.n_models+1):
        if st.session_state[f'n_samp_mod_{i}']>1:
            plot_acc_variance(i)
        else:
            ax.plot(st.session_state[f'model_{i}'].acc_history, color=colors[i], label=f"acc model {i} ({st.session_state[f'al_algo_{i}']})")
    
    plt.title('Accuracy on test set')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    st.write(fig)



def plot_acc_variance(i):
    
    n_samp = st.session_state[f'n_samp_mod_{i}']
    n_epochs = len(st.session_state[f'model_{i}.{0}'].acc_history)

    acc = np.array([st.session_state[f'model_{i}.{j}'].acc_history for j in range(n_samp)])
    df = pd.DataFrame(acc, columns=np.arange(n_epochs)).melt()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    sns.lineplot(data=df, x='variable', y='value', estimator=np.mean, ci='sd', color=colors[i-1], label=f"acc model {i} ({st.session_state[f'al_algo_{i}']}) [x{n_samp}]")
    sns.lineplot(data=df, x='variable', y='value', estimator=np.mean,  ci=50, color=colors[i-1])
    sns.lineplot(data=df, x='variable', y='value', estimator=np.min, linestyle='--', ci=None, color=colors[i-1], alpha=0.5)
    sns.lineplot(data=df, x='variable', y='value', estimator=np.max, linestyle='--', ci=None, color=colors[i-1], alpha=0.5)



def plot_confusion():
    """
    """    
    n = st.session_state.n_models
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(5*n, 4))
    
    if n>1:
        for i in range(n):
            ax[i].set_title('Confusion matrix on test set')

            c_1 = matplotlib.colors.colorConverter.to_rgba('black', alpha = 1)
            c_2 = matplotlib.colors.colorConverter.to_rgba(colors[i], alpha = 1)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap', [c_1, c_2], 512)
            
            if st.session_state[f'n_samp_mod_{i+1}']>1:
                st.session_state[f'model_{i+1}.0'].plot_confusion(ax=ax[i], cmap=cmap)
            else:
                st.session_state[f'model_{i+1}'].plot_confusion(ax=ax[i], cmap=cmap)
    else:
        ax.set_title('Confusion matrix on test set')

        c_1 = matplotlib.colors.colorConverter.to_rgba('black', alpha = 1)
        c_2= matplotlib.colors.colorConverter.to_rgba(colors[0], alpha = 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap', [c_1, c_2], 512)
        
        if st.session_state[f'n_samp_mod_1']>1:
            st.session_state[f'model_1.0'].plot_confusion(ax=ax, cmap=cmap)
        else:
            st.session_state[f'model_1'].plot_confusion(ax=ax, cmap=cmap)

    st.pyplot(fig)



def plot_2D_feature_space(model_name):
    """
    """

    st.write(st.session_state[model_name].emb_fig)