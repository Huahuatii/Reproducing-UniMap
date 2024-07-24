import sys
sys.path.append('..')
import os
import argparse
import pandas as pd
import scanpy as sc
from utils import safe_make_dir
from data_list import get_scanpy_adata
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")


def run_ingest(args):
    # * 1 ingest part
    s_data, t_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    sc.pp.pca(s_data, random_state=args.seed)
    sc.pp.neighbors(s_data, random_state=args.seed)
    sc.tl.umap(s_data, random_state=args.seed)
    sc.tl.ingest(t_data, s_data, obs='pred_celltype')
    st_result = sc.concat([s_data, t_data])

    # * 2 save ingest result
    z_result = pd.DataFrame(st_result.obsm['X_pca'], index=st_result.obs.index, columns=[f'z{i}' for i in range(st_result.obsm['X_pca'].shape[1])])
    umap_result = pd.DataFrame(st_result.obsm['X_umap'], index=st_result.obs.index, columns=['umap1', 'umap2'])
    st_result.obs.to_csv(os.path.join(args.save_folder, 'st_result.csv'))
    z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))
    umap_result.to_csv(os.path.join(args.save_folder, 'st_umap_result.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train ingest')
    parser.add_argument('--model', type=str, default='ingest')
    parser.add_argument('--data_type', type=str, default='pbmc9')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--method', type=str, default='union', choices=['union', 'intersection', 'subtraction'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    args = parser.parse_args()
    args.save_folder = os.path.join('../results', args.data_type, args.model, str(args.seed))

    safe_make_dir(args.save_folder)
    run_ingest(args)
