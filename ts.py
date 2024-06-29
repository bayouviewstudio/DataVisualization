## this web application developed by Shuaibing Li.
## Please cite the following reference if you use this tool:
## Li, S., Zhao, Z., Miao, T., Li, X., Yu, S., Zhang, Y., Wei, J., & Zou, K., 2024. Discrimination of Pb-Zn deposit types using the trace element data of galena based on deep learning.
## Ore Geology Reviews, 170, 106133.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import random
import os

def get_desktop_path():
    """Returns the path to the desktop depending on the OS."""
    home = os.path.expanduser("~")
    return os.path.join(home, "Desktop")


# Set the title of the web application
st.title('Data Visualization (PCA, t-SNE and UMAP)')

# dataFile = pd.read_csv(r'C:\Users\lishuaibing\Desktop\test.csv')
dataFile = pd.read_csv('test.csv')
st.subheader('Example')
st.dataframe(dataFile)

# Adding a sidebar with developer and citation information
st.sidebar.header('Version information June 30, 2024')
st.sidebar.write("""
This is a test version 1.0 of the web application developed by Shuaibing Li.
Please cite the following reference if you use this tool:

Li, S., Zhao, Z., Miao, T., Li, X., Yu, S., Zhang, Y., Wei, J., & Zou, K., 2024. Discrimination of Pb-Zn deposit types using the trace element data of galena based on deep learning. Ore Geology Reviews, 170, 106133. [DOI:10.1016/j.oregeorev.2024.106133](https://doi.org/10.1016/j.oregeorev.2024.106133).



The fig. 1, fig. 5, and fig. 6 in the article were implemented using R language. For the development of this web application, the software is based on Python, with PCA and t-SNE implemented using Python's third-party library, sklearn. Due to differences in the computation of methods such as PCA between the R language packages and Python's sklearn, the visualization effects also differ.
""")
# File uploader for data input
data_file = st.file_uploader("Upload a csv file", type=['csv'])
random.seed(40)

# Initialize session state for figure management
if 'fig' not in st.session_state:
    st.session_state.fig = None

if data_file is not None:
    data = pd.read_csv(data_file)
    st.write(data)  # Display the data

    unique_types = data['Type'].unique()
    type_order = st.multiselect('Customize the order of types:', options=unique_types, default=list(unique_types))
    type_colors = {typ: "#" + ''.join(random.choices('0123456789ABCDEF', k=6)) for typ in type_order}
    type_shapes = {typ: 'o' for typ in type_order}
    type_sizes = {typ: 5 for typ in type_order}  # Size modified for better visibility

    customize = st.checkbox('Customize the attributes of each type')
    if customize:
        for typ in type_order:
            type_colors[typ] = st.color_picker(f"select the color of {typ}", type_colors[typ])
            type_shapes[typ] = st.selectbox(f"select the shape of {typ}", ['o', 's', '^', 'P', '*'], index=0, key=f'shape_{typ}')
            type_sizes[typ] = st.slider(f"select the size of {typ}", 1, 10, 5, key=f'size_{typ}')

    log_transform = st.checkbox('Apply logarithmic transformation')
    if log_transform:
        data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: np.log10(x.clip(lower=1)))

    standardize = st.checkbox('Apply standardization')
    if standardize:
        scaler = StandardScaler()
        data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])

    method = st.selectbox('Select a visualization method', ('PCA', 't-SNE', 'UMAP'))

    if method == 't-SNE':
        random_seed = st.number_input('Set the random seed', value=42, step=1)
        pca_init = st.checkbox('PCA')
        perplexity = st.slider('Perplexity', 5, 50, 30)
        learning_rate = st.slider('eta', 10, 1000, 200)
        n_iter = st.slider('max_iter', 250, 5000, 1000)

    if method == 'UMAP':
        random_seed = st.number_input('Set the random seed', value=42, step=1)
        n_neighbors = st.slider('n_neighbors', 5, 50, 15)
        min_dist = st.slider('min_dist', 0.001, 1.0, 0.1, step=0.001)
        metric = st.selectbox('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra', 'braycurtis', 'mahalanobis'])
        n_epochs = st.slider('n_epochs', 200, 2000, 500)

    st.markdown('<p style="font-weight:bold; color:red;">Start visualization</p>', unsafe_allow_html=True)
    if st.button('Start visualization'):
        st.write("Visualization started...")
        if method == 'PCA':
            pca = PCA(n_components=2)
            transformed_data = pca.fit_transform(data.iloc[:, 1:])
            st.session_state.fig, axs = plt.subplots(1, 2, figsize=(15, 6))

            # Scatter plot for PCA
            for typ in type_order:
                idx = data['Type'] == typ
                axs[0].scatter(transformed_data[idx, 0], transformed_data[idx, 1],
                               c=type_colors[typ], marker=type_shapes[typ], s=type_sizes[typ], label=typ)

            # # Centering the origin
            # x_lim = max(abs(transformed_data[:, 0].min()), transformed_data[:, 0].max()) *1.1
            # y_lim = max(abs(transformed_data[:, 1].min()), transformed_data[:, 1].max()) *1.1
            # axs[0].set_xlim(-x_lim, x_lim)
            # axs[0].set_ylim(-y_lim, y_lim)

            axs[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
            axs[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
            axs[0].legend(title="Types", loc='upper left', bbox_to_anchor=(-0.30, 1))
            axs[0].set_aspect('equal', adjustable='datalim')
            axs[0].set_title('PCA Scatter Plot')

            # Loadings plot
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            for i, feature in enumerate(data.columns[1:]):
                axs[1].arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.02, head_length=0.05, fc='black', ec='black')
                axs[1].text(loadings[i, 0], loadings[i, 1], feature, ha='right', va='top')
            # Drawing dashed lines at the origin
            axs[1].axhline(y=0, color='gray', linestyle='--')
            axs[1].axvline(x=0, color='gray', linestyle='--')

            max_loading = np.max(np.abs(loadings)) * 1.1
            axs[1].set_xlim(-max_loading, max_loading)
            axs[1].set_ylim(-max_loading, max_loading)
            axs[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
            axs[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
            axs[1].set_title('PCA Loadings Plot')
            st.pyplot(st.session_state.fig)

        elif method == 't-SNE':
            tsne = TSNE(n_components=2, init='pca' if pca_init else 'random', perplexity=perplexity,
                        learning_rate=learning_rate, n_iter=n_iter, random_state=random_seed)
            transformed_data = tsne.fit_transform(data.iloc[:, 1:])
            st.session_state.fig, ax = plt.subplots(figsize=(8, 8))
            # Create a scatter plot for each unique type, applying the customized or default settings
            for typ in type_order:
                idx = data['Type'] == typ
                ax.scatter(transformed_data[idx, 0], transformed_data[idx, 1], label=typ,
                           color=type_colors[typ], marker=type_shapes[typ], s=type_sizes[typ])

            # ax.set_aspect('equal')
            ax.set_xlabel('t-SNE1')
            ax.set_ylabel('t-SNE2')
            ax.set_title('t-SNE Scatter Plot')
            ax.legend(title="Types", loc='upper left', bbox_to_anchor=(-0.35, 1))
            st.pyplot(st.session_state.fig)
            pass
        elif method == 'UMAP':
            umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_epochs=n_epochs, random_state=random_seed)
            transformed_data = umap.fit_transform(data.iloc[:, 1:])
            st.session_state.fig, ax = plt.subplots()
            scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1],
                                 c=data['Type'].astype('category').cat.codes)
            ax.set_title('UMAP Scatter Plot')
            st.session_state.fig, ax = plt.subplots(figsize=(8, 8))

            for typ in type_order:
                idx = data['Type'] == typ
                ax.scatter(transformed_data[idx, 0], transformed_data[idx, 1], label=typ,
                           color=type_colors[typ], marker=type_shapes[typ], s=type_sizes[typ])

            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.set_title('UMAP Scatter Plot')
            ax.legend(title="Types", loc='upper left', bbox_to_anchor=(-0.35, 1))
            st.pyplot(st.session_state.fig)
            pass


    # Save button and functionality
    filepath = st.text_input('Save file path', get_desktop_path())
    filename = st.text_input('Input filename', "visualization")
    format_choice = st.radio('Select file format', ['PNG', 'PDF'])

    save_button = st.button('Save')
    if save_button and st.session_state.fig:
        full_path = os.path.join(filepath, f"{filename}.{format_choice.lower()}")
        try:
            st.session_state.fig.savefig(full_path, format=format_choice.lower(), bbox_inches='tight')
            st.success(f"The image has been saved as {format_choice.upper()} at {full_path}")
        except Exception as e:
            st.error(f"Save failed: {str(e)}")


# # streamlit run D:/DLearning/myGMT/ts.py --server.enableXsrfProtection=false
