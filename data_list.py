import os
import torch
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import pickle as pkl

from sklearn.preprocessing import LabelEncoder
from utils import get_common_genes, set_seed, Label_Encoder
from torch.utils.data import TensorDataset, DataLoader, random_split

warnings.filterwarnings('ignore')

root_data_folder = os.path.dirname(os.path.abspath(__file__)) + '/data'


def get_scanpy_adata(datatype, seed=2023, model='unimap'):
    set_seed(seed)
    if datatype == 'pbmc9':
        # * pbmc9
        pbmc9_s = os.path.join(root_data_folder, 'pbmc9/pbmc9_s.h5ad')
        pbmc9_t = os.path.join(root_data_folder, 'pbmc9/pbmc9_t.h5ad')
        pbmc9_hvg = pkl.load(open(os.path.join(root_data_folder, 'pbmc9/hvg_1815_pbmc9.pkl'), "rb"))
        s_data = sc.read_h5ad(pbmc9_s)
        t_data = sc.read_h5ad(pbmc9_t)
        s_data = s_data[:, s_data.var.index.isin(pbmc9_hvg)]
        t_data = t_data[:, t_data.var.index.isin(pbmc9_hvg)]

    elif datatype == 'pbmc40':
        # * pbmc40
        pbmc10 = os.path.join(root_data_folder, 'pbmc40/pbmc10.h5ad')
        pbmc40 = os.path.join(root_data_folder, 'pbmc40/pbmc40.h5ad')
        pbmc40_hvg = pkl.load(open(os.path.join(root_data_folder, 'pbmc40/hvg_1582_pbmc40.pkl'), "rb"))
        s_data = sc.read_h5ad(pbmc40)
        t_data = sc.read_h5ad(pbmc10)
        s_data = s_data[:, s_data.var.index.isin(pbmc40_hvg)]
        t_data = t_data[:, t_data.var.index.isin(pbmc40_hvg)]

    elif datatype == 'cross_species':
        # * cross_species
        human = os.path.join(root_data_folder, 'cross_species/human.h5ad')
        mouse = os.path.join(root_data_folder, 'cross_species/mouse.h5ad')
        mm = os.path.join(root_data_folder, 'cross_species/macaqueM.h5ad')
        mf = os.path.join(root_data_folder, 'cross_species/macaqueF.h5ad')
        cross_species_hvg = pkl.load(open(os.path.join(root_data_folder, 'cross_species/hvg_1208_cross_species.pkl'), "rb"))  # (1600)
        s_data = sc.read_h5ad(human)
        t_data1 = sc.read_h5ad(mm)
        t_data2 = sc.read_h5ad(mouse)
        t_data3 = sc.read_h5ad(mf)

        s_data = s_data[:, s_data.var.index.isin(cross_species_hvg)]
        t_data1.obs.assign(domain_info='mm')
        t_data2.obs.assign(domain_info='mouse')
        t_data3.obs.assign(domain_info='mf')

        t_data = sc.concat([t_data1, t_data2, t_data3], join='outer', label='all')
        t_data.var['highly_variable'] = t_data.var.index.isin(cross_species_hvg)

    elif datatype == 'mg':
        # * mg
        mg_s = os.path.join(root_data_folder, 'mg/mg_ref.h5ad')
        mg_t = os.path.join(root_data_folder, 'mg/mg_query.h5ad')
        mg_hvg = pkl.load(open(os.path.join(root_data_folder, 'mg/hvg_1649_mg.pkl'), "rb"))
        s_data = sc.read_h5ad(mg_s)
        t_data = sc.read_h5ad(mg_t)
        s_data = s_data[:, s_data.var.index.isin(mg_hvg)]
        t_data = t_data[:, t_data.var.index.isin(mg_hvg)]

    else:
        raise ValueError('Invalid data type')

    s_data.obs['pred_celltype'] = s_data.obs['celltype']
    s_data.obs, t_data.obs = s_data.obs.assign(domain='source'), t_data.obs.assign(domain='target')
    t_data = t_data[:, s_data.var.index]
    return s_data, t_data


def get_unimap_dataset(args):
    s_data, t_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    s_gex, t_gex = s_data.to_df(), t_data.to_df()
    s_celltype = s_data.obs['celltype']
    s_idx, t_idx = s_data.obs.index, t_data.obs.index

    ce = Label_Encoder()
    ce.fit(list(set(s_celltype)))
    s_onehot_celltype = ce.transform(s_celltype)

    be = LabelEncoder()
    be.fit(pd.concat([s_data.obs.batch, t_data.obs.batch]).to_list())

    s_int_batch, t_int_batch = be.transform(s_data.obs.batch), be.transform(t_data.obs.batch)

    # Alignment Features
    s_gex = s_gex.reindex(sorted(s_gex.columns), axis=1)
    t_gex = t_gex.reindex(sorted(t_gex.columns), axis=1)

    if (s_gex.columns != t_gex.columns).all():
        print('Feature in Source and Target are not aligned!')
        raise ValueError('Feature in Source and Target are not aligned!')
    else:
        print('Feature in Source and Target are aligned!')

    s_dataset = TensorDataset(torch.from_numpy(s_gex.values.astype('float32')),
                              torch.from_numpy(s_onehot_celltype.astype('float32')),
                              torch.from_numpy(s_int_batch))

    t_dataset = TensorDataset(torch.from_numpy(t_gex.values.astype('float32')),
                              torch.from_numpy(t_int_batch),
                              torch.from_numpy(t_int_batch),
                              )

    args.in_feature = s_gex.shape[1]
    args.ce, args.be = ce, be
    args.num_classes, args.num_batches = len(ce.classes_), len(be.classes_)
    print(f'Current config is: \n{args}')
    args.s_idx, args.t_idx = s_idx, t_idx
    return s_dataset, t_dataset, args


def get_unimap_dataloader(args):
    s_dataset, t_dataset, args = get_unimap_dataset(args)
    s_loader = DataLoader(s_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    t_loader = DataLoader(t_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return (s_loader, t_loader), args


def get_code_ae_dataloader(args):
    s_dataset, t_dataset, args = get_unimap_dataset(args)
    s_train_size, t_train_size = int(0.8 * len(s_dataset)), int(0.8 * len(t_dataset))
    s_test_size, t_test_size = len(s_dataset) - s_train_size, len(t_dataset) - t_train_size
    s_train_dataset, s_test_dataset = random_split(s_dataset, [s_train_size, s_test_size])
    t_train_dataset, t_test_dataset = random_split(t_dataset, [t_train_size, t_test_size])

    s_train_loader = DataLoader(s_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    s_test_loader = DataLoader(s_test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    t_train_loader = DataLoader(t_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    t_test_loader = DataLoader(t_test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return (s_train_loader, s_test_loader), (t_train_loader, t_test_loader), args
