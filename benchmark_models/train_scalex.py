import sys
sys.path.append('..')
import pandas as pd
import scanpy as sc
import argparse
import os
from utils import safe_make_dir
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from data_list import get_scanpy_adata
from rich.console import Console
from rich.panel import Panel
import umap
from scalex import SCALEX

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")


def start_scalex(args):
    s_data, t_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    st_data = sc.concat([s_data, t_data])

    # scalex train
    result = SCALEX(st_data, processed=True, seed=args.seed, show=False)

    # caculate t_yhat
    s_res = result[result.obs.domain == 'source']
    t_res = result[result.obs.domain == 'target']

    s_res.obs['pred_celltype'] = s_res.obs['celltype']
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(s_res.obsm['latent'], s_res.obs['pred_celltype'])
    t_yhat = knn.predict(t_res.obsm['latent']).tolist()

    # save result
    t_res.obs['pred_celltype'] = t_yhat
    st_result = pd.concat([s_res.obs, t_res.obs])
    st_result.to_csv(os.path.join(args.save_folder, 'st_result.csv'))

    st_z_result = pd.DataFrame(result.obsm['latent'], index=st_data.obs.index, columns=[f'z{i}' for i in range(result.obsm['latent'].shape[1])])
    st_z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))

    st_umap_result = pd.DataFrame(result.obsm['X_umap'], index=st_data.obs.index, columns=['umap1', 'umap2'])
    st_umap_result.to_csv(os.path.join(args.save_folder, 'st_umap_result.csv'))

    console = Console()
    output = [
        f"All results are saved in: {args.save_folder}",
        "1. st_result.csv: source and target dataset result",
        "2. st_z_result.csv: concatenated source and target embeddings",
        "3. st_umap_result.csv: UMAP embeddings of concatenated source and target embeddings"
        ]
    panel = Panel.fit("\n".join(output), title="train finished")
    console.print(panel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train SCALEX')
    parser.add_argument('--model', type=str, default='scalex')
    parser.add_argument('--data_type', type=str, default='pbmc9')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--method', type=str, default='union', choices=['union', 'intersection', 'subtraction'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    args = parser.parse_args()
    args.save_folder = os.path.join('../results', args.data_type, args.model, str(args.seed))

    safe_make_dir(args.save_folder)
    start_scalex(args)
