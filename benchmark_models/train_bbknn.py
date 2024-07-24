import sys
sys.path.append('..')
import os
import argparse
import pandas as pd
import scanpy as sc
from utils import safe_make_dir
from data_list import get_scanpy_adata
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")


def run_bbknn(args):
    s_data, t_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    st_data = sc.concat([s_data, t_data])
    sc.pp.pca(st_data)
    sc.pp.neighbors(st_data)
    sc.tl.umap(st_data)
    sc.external.pp.bbknn(st_data, batch_key='batch', pynndescent_random_state=args.seed)
    sc.tl.umap(st_data)
    knn = KNeighborsClassifier(n_neighbors=10)
    s_data = st_data[st_data.obs['domain'] == 'source']
    t_data = st_data[st_data.obs['domain'] == 'target']
    knn.fit(s_data.obsm['X_pca'], s_data.obs['celltype'])
    st_data.obs['pred_celltype'] = st_data.obs['celltype']
    st_data.obs.loc[st_data.obs['domain'] == 'target', 'pred_celltype'] = knn.predict(t_data.obsm['X_pca'])

    z_result = pd.DataFrame(st_data.obsm['X_pca'], index=st_data.obs.index, columns=[f'z{i}' for i in range(st_data.obsm['X_pca'].shape[1])])
    umap_result = pd.DataFrame(st_data.obsm['X_umap'], index=st_data.obs.index, columns=['umap1', 'umap2'])
    st_result = st_data
    st_result.obs.to_csv(os.path.join(args.save_folder, 'st_result.csv'))
    z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))
    umap_result.to_csv(os.path.join(args.save_folder, 'st_umap_result.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train bbknn')
    parser.add_argument('--model', type=str, default='bbknn')
    parser.add_argument('--data_type', type=str, default='pbmc9')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--method', type=str, default='union', choices=['union', 'intersection', 'subtraction'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    args = parser.parse_args()
    args.save_folder = os.path.join('../results', args.data_type, args.model, str(args.seed))

    safe_make_dir(args.save_folder)
    run_bbknn(args)
