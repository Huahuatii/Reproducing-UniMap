import sys
sys.path.append('..')
import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import os
from utils import safe_make_dir
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from data_list import get_scanpy_adata
from rich.console import Console
from rich.panel import Panel
import umap
from scarches.models.scpoli import scPoli

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")


def start_scpoli(args):
    s_data, t_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    s_data.X = s_data.X.astype(np.float32)
    t_data_copy = sc.AnnData(X=t_data.X.astype(np.float32), obs=t_data.obs, var=t_data.var)
    t_data = t_data_copy
    st_data = sc.concat([s_data, t_data])

    # scalex train
    early_stopping_kwargs = {
        "early_stopping_metric": "val_prototype_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    scpoli_model = scPoli(
        adata=s_data,
        condition_keys='batch',
        cell_type_keys='celltype',
        embedding_dims=5,
        recon_loss='nb',
    )
    scpoli_model.train(
        n_epochs=50,
        pretraining_epochs=40,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=5,
    )
    scpoli_query = scPoli.load_query_data(
        adata=t_data,
        reference_model=scpoli_model,
        labeled_indices=[],
    )
    scpoli_query.train(
        n_epochs=50,
        pretraining_epochs=40,
        eta=10
    )
    results_dict = scpoli_query.classify(t_data, scale_uncertainties=True)
    scpoli_query.model.eval()
    s_z = scpoli_query.get_latent(
        s_data,
        mean=True
    )
    t_z = scpoli_query.get_latent(
        t_data,
        mean=True
    )
    # st_result
    s_data.obs['pred_celltype'] = s_data.obs['celltype']
    t_data.obs['pred_celltype'] = results_dict['celltype']['preds']
    st_result = pd.concat([s_data.obs, t_data.obs], axis=0)
    st_result.to_csv(os.path.join(args.save_folder, 'st_result.csv'))

    # st_z_result
    st_z = np.concatenate([s_z, t_z])
    st_z_result = pd.DataFrame(st_z, index=st_data.obs.index, columns=[f'z{i+1}' for i in range(st_z.shape[1])])
    st_z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))

    # st_umap_result
    print('Calculating UMAP...\nThis may take a few minutes...')
    st_umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.8, metric='correlation', random_state=args.seed, n_jobs=1).fit_transform(st_z_result.values)
    st_umap_result = pd.DataFrame(st_umap_embedding, index=st_z_result.index, columns=['umap1', 'umap2'])
    st_umap_result.to_csv(os.path.join(args.save_folder, 'st_umap_result.csv'))

    # t_prob_result
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
    parser = argparse.ArgumentParser(description='train scPoli')
    parser.add_argument('--model', type=str, default='scpoli')
    parser.add_argument('--data_type', type=str, default='pbmc9')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--method', type=str, default='union', choices=['union', 'intersection', 'subtraction'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    args = parser.parse_args()
    args.save_folder = os.path.join('../results', args.data_type, args.model, str(args.seed))

    safe_make_dir(args.save_folder)
    start_scpoli(args)
