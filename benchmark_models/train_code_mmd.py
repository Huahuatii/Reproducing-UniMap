import sys
sys.path.append('..')

import os
import json
import numpy as np
import torch
import torch.autograd as autograd
import argparse
import scanpy as sc
import pandas as pd
import umap
from itertools import chain
from code_adv_utils import SharedEncoder, SharedDecoder, DSNAE, mmd_loss
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from utils import model_save_check, safe_make_dir
from data_list import get_scanpy_adata, get_code_ae_dataloader
from rich.console import Console
from rich.panel import Panel


def train_step(s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    optimizer.zero_grad()
    m_loss = mmd_loss(source_features=s_code, target_features=t_code, device=device)
    loss = s_loss_dict['loss'] + t_loss_dict['loss'] + m_loss
    loss.backward()
    optimizer.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    for k, v in loss_dict.items():
        history[k].append(v)
    history['mmd_loss'].append(m_loss.cpu().detach().item())

    return history


def eval_step(model, data_loader, device, history):
    model.eval()
    avg_loss_dict = defaultdict(float)
    for (idx, x_batch) in enumerate(data_loader):
        x = x_batch[0].to(device)
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history


def train_code_mmd(args):
    # loader
    loader = {}
    (loader['s_train'], loader['s_test']), (loader['t_train'], loader['t_test']), args = get_code_ae_dataloader(args)
    # model
    shared_encoder = SharedEncoder(in_feature=args.in_feature, latent_feature=args.out_feature, drop=args.drop, gr_flag=True).to(args.device)
    shared_decoder = SharedDecoder(in_feature=args.out_feature * 2, latent_feature=args.in_feature, drop=args.drop, gr_flag=True).to(args.device)
    s_dsnae = DSNAE(shared_encoder, shared_decoder, in_feature=args.in_feature, alpha=1.0, drop=args.drop).to(args.device)
    t_dsnae = DSNAE(shared_encoder, shared_decoder, in_feature=args.in_feature, alpha=1.0, drop=args.drop).to(args.device)

    # optimizer
    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()]
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=args.lr)

    # train
    train_history = defaultdict(list)
    eval_history = defaultdict(list)
    for epoch in range(int(args.train_epoch)):

        # train
        for (idx, s_batch) in enumerate(loader['s_train']):
            t_batch = next(iter(loader['t_train']))
            train_history = train_step(s_dsnae, t_dsnae, s_batch, t_batch, args.device, ae_optimizer, train_history)
        # eval
        eval_history = eval_step(s_dsnae, loader['s_test'], args.device, eval_history)
        eval_history = eval_step(t_dsnae, loader['t_test'], args.device, eval_history)

        # sum the s_dsnae_loss and t_dsnae_loss
        for k in eval_history:
            if k != 'best_index':
                eval_history[k][-2] += eval_history[k][-1]
                eval_history[k].pop()

        save_flag, stop_flag = model_save_check(history=eval_history, metric_name='loss', tolerance_count=args.tolerance)
        if save_flag:
            torch.save(s_dsnae.state_dict(), os.path.join(args.save_folder, 'best_model.pth'))
        if stop_flag:
            break
        # if epoch % 20 == 0:
        print('epoch: {}\ttrain_loss: {:.4f}\ttest_loss: {:.4f}\tbest_idx: {:d}'.format(epoch, train_history['loss'][-1], eval_history['loss'][-1], eval_history['best_index']))

    # generate code
    s_row_data, t_row_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    s_dsnae.load_state_dict(torch.load(os.path.join(args.save_folder, 'best_model.pth')))
    s_x, t_x = loader['s_train'].dataset.dataset.tensors[0], loader['t_train'].dataset.dataset.tensors[0]

    s_z = s_dsnae.s_encode(s_x.to(args.device)).cpu().detach().numpy()
    t_z = s_dsnae.s_encode(t_x.to(args.device)).cpu().detach().numpy()
    st_z = np.concatenate([s_z, t_z], axis=0)
    st_z_result = pd.DataFrame(st_z, index=list(s_row_data.obs.index) + list(t_row_data.obs.index), columns=[f'z{i}' for i in range(s_z.shape[1])])
    st_z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))

    print('Calculating UMAP...\nThis may take a few minutes...')
    st_umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.8, metric='correlation', random_state=args.seed).fit_transform(st_z_result.values)
    st_umap_result = pd.DataFrame(st_umap_embedding, index=st_z_result.index, columns=['umap1', 'umap2'])
    st_umap_result.to_csv(os.path.join(args.save_folder, 'st_umap_result.csv'))

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(s_z, s_row_data.obs['pred_celltype'])
    t_ycharhat = knn.predict(t_z)
    t_row_data.obs['pred_celltype'] = t_ycharhat
    st_result = pd.DataFrame(pd.concat([s_row_data.obs, t_row_data.obs], axis=0))
    st_result.to_csv(os.path.join(args.save_folder, 'st_result.csv'))

    # umap result

    console = Console()
    output = [
        f"All results are saved in: {args.save_folder}",
        "1. st_result.csv: source and target dataset result",
        "2. best_model.pth: best model",
        "3. st_z_result.csv: concatenated source and target embeddings",
        "4. st_umap_result.csv: UMAP embeddings of concatenated source and target embeddings"
        ]
    panel = Panel.fit("\n".join(output), title="train finished")
    console.print(panel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train code_mmd')
    parser.add_argument('--model', type=str, default='code_mmd')
    parser.add_argument('--data_type', type=str, default='pbmc9')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--method', type=str, default='union', choices=['union', 'intersection', 'subtraction'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--out_feature', type=int, default=128)
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--tolerance', type=int, default=50, help='early stop tolerance')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.save_folder = os.path.join('../results', args.data_type, args.model, str(args.seed))
    print(f'current config is: \n{args}')
    # save dir
    safe_make_dir(args.save_folder)
    train_code_mmd(args)
