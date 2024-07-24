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
import portal
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")


def start_portal(args):
    s_data, t_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    st_data = sc.concat([s_data, t_data])

    pca = PCA(n_components=50, svd_solver="arpack", random_state=2023)
    st_data.obsm["X_pca"] = pca.fit_transform(st_data.X.toarray())
    emb_A = st_data.obsm["X_pca"][:s_data.X.shape[0], :30].copy()
    emb_B = st_data.obsm["X_pca"][s_data.X.shape[0]:, :30].copy()
    model = portal.model.Model(training_steps=1000, seed=args.seed)
    model.emb_A = emb_A
    model.emb_B = emb_B
    model.train()  # train the model
    model.eval()

    st_z_result = pd.DataFrame(model.latent, index=st_data.obs.index, columns=[f'z{i}' for i in range(model.latent.shape[1])])
    st_z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))

    s_res = st_data.obs[st_data.obs.domain == 'source']
    t_res = st_data.obs[st_data.obs.domain == 'target']
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(model.latent[:len(s_res),], s_res.celltype)
    t_yhat = knn.predict(model.latent[len(s_res):,]).tolist()
    s_res['pred_celltype'] = s_res['celltype']
    t_res['pred_celltype'] = t_yhat
    st_result = pd.concat([s_res, t_res])
    st_result.to_csv(os.path.join(args.save_folder, 'st_result.csv'))

    st_umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.8, metric='correlation').fit_transform(st_z_result.values)  # min_dist=0.8
    st_umap_result = pd.DataFrame(st_umap_embedding, index=st_z_result.index, columns=['umap1', 'umap2'])
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
    parser = argparse.ArgumentParser(description='train Portal')
    parser.add_argument('--model', type=str, default='portal')
    parser.add_argument('--data_type', type=str, default='pbmc9')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--method', type=str, default='union', choices=['union', 'intersection', 'subtraction'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    args = parser.parse_args()
    args.save_folder = os.path.join('../results', args.data_type, args.model, str(args.seed))

    safe_make_dir(args.save_folder)
    start_portal(args)
