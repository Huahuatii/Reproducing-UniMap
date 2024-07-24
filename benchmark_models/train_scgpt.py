import sys
sys.path.append('..')
sys.path.insert(0, "../")
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import os
import sklearn
from utils import safe_make_dir
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from data_list import get_scanpy_adata
from rich.console import Console
from rich.panel import Panel
import umap
from pathlib import Path

from scipy.stats import mode
import faiss
import scgpt as scg
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


def train_scgpt(args):
    model_dir = Path("scgpt")
    cell_type_key = "celltype"
    gene_col = "index"
    k = 10

    s_data, t_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    s_res = scg.tasks.embed_data(s_data,
                                 model_dir,
                                 gene_col=gene_col,
                                 obs_to_save=cell_type_key,
                                 batch_size=64,
                                 return_new_adata=True)

    t_res = scg.tasks.embed_data(t_data,
                                 model_dir,
                                 gene_col=gene_col,
                                 obs_to_save=cell_type_key,
                                 batch_size=64,
                                 return_new_adata=True)
    s_z = s_res.X
    t_z = t_res.X

    index = faiss.IndexFlatL2(s_z.shape[1])
    index.add(s_z)
    distances, labels = index.search(t_z, k)

    idx_list = [i for i in range(t_z.shape[0])]
    pred_celltype = []
    for k in idx_list:
        idx = labels[k]

        pred = s_res.obs[cell_type_key][idx].value_counts()
        pred_celltype.append(pred.index[0])

    s_res, t_res = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)

    s_res.obs['pred_celltype'] = s_res.obs[cell_type_key]
    s_res.obs['domain'] = 'source'
    t_res.obs['pred_celltype'] = pred_celltype
    t_res.obs['domain'] = 'target'

    # save result
    st_result = pd.concat([s_res.obs, t_res.obs])
    st_result.to_csv(os.path.join(args.save_folder, 'st_result.csv'))

    st_z_result = pd.DataFrame(np.concatenate([s_z, t_z]), index=st_result.index, columns=[f'z{i}' for i in range(np.concatenate([s_z, t_z]).shape[1])])
    st_z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))

    st_umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.8, metric='correlation', random_state=args.seed, n_jobs=1).fit_transform(st_z_result.values)
    st_umap_result = pd.DataFrame(st_umap_embedding, index=st_z_result.index, columns=['umap1', 'umap2'])
    st_umap_result.to_csv(os.path.join(args.save_folder, 'st_umap_result.csv'))
    console = Console()
    output = [
             f"All results are saved in: {args.save_folder}",
             "1. st_result.csv: source and target dataset result",
             # "2. train_history.csv: training history",
             # "3. best_model.pth: best model",
             "2. st_z_result.csv: concatenated source and target embeddings",
             # "5. t_prob_result.csv: target dataset prediction probabilities",
             "3. st_umap_result.csv: UMAP embeddings of concatenated source and target embeddings"
             ]
    panel = Panel.fit("\n".join(output), title="train finished")
    console.print(panel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train scGPT')
    parser.add_argument('--model', type=str, default='scgpt')
    parser.add_argument('--data_type', type=str, default='cross_species')  # cross_species
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--method', type=str, default='union', choices=['union', 'intersection', 'subtraction'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    args = parser.parse_args()
    args.save_folder = os.path.join('../results', args.data_type, args.model, str(args.seed))

    safe_make_dir(args.save_folder)
    train_scgpt(args)
