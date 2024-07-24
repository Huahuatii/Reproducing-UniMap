import sys
sys.path.append('..')

import pandas as pd
import scanpy as sc
import harmonypy as hm
import argparse
import os
from utils import safe_make_dir
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from data_list import get_scanpy_adata
from rich.console import Console
from rich.panel import Panel
import umap
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
print('123')

def train_harmony(args):
    s_data, t_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    st_data = sc.concat([s_data, t_data])

    pca = PCA(n_components=50, random_state=args.seed)
    st_pcs = pca.fit_transform(st_data.X.toarray())
    ho = hm.run_harmony(st_pcs, st_data.obs, vars_use='batch', random_state=args.seed)
    res = pd.DataFrame(ho.Z_corr).T
    s_res = res.iloc[:s_data.shape[0], :]
    t_res = res.iloc[s_data.shape[0]:, :]

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(s_res, s_data.obs['celltype'])
    t_yhat = knn.predict(t_res).tolist()

    s_data.obs['pred_celltype'] = s_data.obs['celltype']
    t_data.obs['pred_celltype'] = t_yhat
    st_result = pd.concat([s_data.obs, t_data.obs], axis=0)
    st_result.to_csv(os.path.join(args.save_folder, 'st_result.csv'))

    st_z_result = pd.DataFrame(res.values, index=st_data.obs.index, columns=[f'hpc{i+1}' for i in range(res.shape[1])])
    st_z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))
    print('Calculating UMAP...\nThis may take a few minutes...')
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
    parser = argparse.ArgumentParser(description='train harmony')
    parser.add_argument('--model', type=str, default='harmony')
    parser.add_argument('--data_type', type=str, default='pbmc9')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--method', type=str, default='union', choices=['union', 'inter', 's_union', 's_inter', 'st_union'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    args = parser.parse_args()
    args.save_folder = os.path.join('../results', args.data_type, args.model, str(args.seed))

    safe_make_dir(args.save_folder)
    train_harmony(args)
