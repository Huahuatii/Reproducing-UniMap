import os
import umap
import loss
import torch
import network
import argparse
import warnings

import numpy as np
import pandas as pd

from itertools import chain
from rich.panel import Panel
from rich.console import Console
from collections import defaultdict
from utils import model_save_check, safe_make_dir
from data_list import get_unimap_dataloader, get_scanpy_adata

warnings.filterwarnings('ignore')


def train_step(feature_extractor, ad_nets, celltype_weights, optimizer, scheduler, loader, history, epoch, args):
    ep_total_loss, ep_s_loss, ep_transfer_loss, ep_t_loss, ep_margin_loss = 0, 0, 0, 0, 0
    for (idx, s_batch) in enumerate(loader['source']):

        # * ========== init ==========
        t_batch = next(iter(loader['target']))
        feature_extractor.zero_grad()
        feature_extractor.train()
        for ad_net in ad_nets:
            ad_net.zero_grad()
            ad_net.train()

        # * ========== load data ==========
        s_x = s_batch[0].to(args.device)
        s_y = s_batch[1].to(args.device)
        s_b = s_batch[2].to(args.device)
        s_yint = torch.nonzero(s_y)[:, 1].to(args.device)
        s_z, s_yhat = feature_extractor(s_x, s_b)
        t_x = t_batch[0].to(args.device)
        t_b = t_batch[2].to(args.device)
        t_z, t_yhat = feature_extractor(t_x, t_b)
        st_z = torch.cat((s_z, t_z), dim=0)
        st_yhat = torch.cat((s_yhat, t_yhat), dim=0)

        # * ========== cell weights ==========
        st_cell_entropy = loss.Entropy(torch.nn.Softmax(dim=1)(st_yhat))
        st_cell_r = 1 + torch.exp(-st_cell_entropy)
        st_cell_w = torch.ones(st_yhat.size(0)).to(args.device)
        st_cell_w[:args.batch_size] = st_cell_r[:args.batch_size] * celltype_weights[s_yint.long()]
        st_cell_w[args.batch_size:] = st_cell_r[args.batch_size:]
        s_cell_w = st_cell_w[:args.batch_size]
        t_cell_w = st_cell_w[args.batch_size:]

        # * ========== [1 Source Domain Loss] ==========
        s_cell_loss = loss.FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction=None)(s_yhat, s_y)
        s_loss = torch.sum(s_cell_w * s_cell_loss) / (1e-8 + torch.sum(s_cell_w).item())

        # * ========== [2 Target Domain Loss] ==========
        confidence, pseudo_yhat = torch.max(torch.nn.Softmax(dim=1)(t_yhat).detach(), dim=1)
        t_cell_w_pseudo = t_cell_w * (confidence > args.conf_thres).float()
        t_cell_loss = loss.FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction=None)(t_yhat, pseudo_yhat)
        t_loss = torch.sum(t_cell_w_pseudo * t_cell_loss) / (1e-8 + torch.sum(t_cell_w_pseudo).item())

        # * ========== [3 Margin Loss] ==========
        margin_loss = loss.marginloss(yHat=s_yhat,
                                      y=s_y,
                                      classes=args.num_classes,
                                      alpha=1,
                                      weight=s_cell_w.detach(), device=args.device)

        # * ========== [4 Transfer Loss] ==========
        iter_num = epoch * len(loader['source']) + idx
        coeff = loss.calc_coeff(iter_num, 1, 0, 10, args.max_iter) / args.num_classes
        transfer_loss = loss.DANN_multi(st_z=st_z,
                                        ad_nets=ad_nets,
                                        celltype_weights=celltype_weights,
                                        st_cell_w=st_cell_w,
                                        st_cell_prob=torch.nn.Softmax(dim=1)(st_yhat).detach(),
                                        st_cell_entropy=st_cell_entropy,
                                        coeff=coeff,
                                        cell_weights=st_cell_w,
                                        batch_size=args.batch_size)

        # * ========== [5 Total Loss] ==========
        total_loss = s_loss + args.t_loss_w * t_loss + args.trans_loss_w * transfer_loss + margin_loss * args.margin_w

        # * ========== backward ==========
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # * ========== update history ==========
        ep_total_loss += total_loss.cpu().item()
        ep_s_loss += s_loss.cpu().item()
        ep_transfer_loss += transfer_loss.cpu().item()
        ep_t_loss += t_loss.cpu().item()
        ep_margin_loss = margin_loss.cpu().item()

    loss_dict = {'total_loss': round(ep_total_loss / (idx + 1), 4),
                 's_loss': round(ep_s_loss / (idx + 1), 4),
                 'transfer_loss': round(ep_transfer_loss / (idx + 1), 4),
                 't_loss': round(ep_t_loss / (idx + 1), 4),
                 'margin_loss': round(ep_margin_loss / (idx + 1), 4),
                 }
    for key, value in loss_dict.items():
        history[key].append(value)
    return history


def eval_step(feature_extractor, loader, history, device, args):
    feature_extractor.eval()
    with torch.no_grad():
        t_x = loader['target'].dataset.tensors[0].to(args.device)
        t_b = loader['target'].dataset.tensors[2].to(args.device)
        t_z, t_yhat = feature_extractor(t_x, t_b)

    mean_ent = torch.mean(loss.Entropy(torch.nn.Softmax(dim=1)(t_yhat))).cpu().item().__round__(4)

    ct_w_no_norm = torch.nn.Softmax(dim=1)(t_yhat).sum(dim=0)
    celltype_weights = (ct_w_no_norm / ct_w_no_norm.max()).to(device).detach()

    history['mean_ent'].append(mean_ent)
    history['celltype_weights'] = celltype_weights.cpu().numpy()
    return celltype_weights, history


def train_unimap(args):
    # * ========== loader ==========
    loader = {}
    (loader['source'], loader['target']), args = get_unimap_dataloader(args)
    args.max_iter = len(loader['source']) * args.epoch

    # * ========== model ==========
    feature_extractor = network.CMLP(in_feature=args.in_feature,
                                    latent_feature=args.latent_feature,
                                    num_classes=args.num_classes,
                                    drop=args.drop,
                                    num_batches=args.num_batches).to(args.device)

    ad_nets = [network.AdversarialNetwork(in_feature=args.latent_feature,
                                          hidden_size=32,
                                          dp=args.drop,
                                          max_iter=args.max_iter).to(args.device) for _ in range(args.num_classes)]

    # * ========== optimizer ==========
    params_list = [feature_extractor.parameters()]
    for ad_net in ad_nets:
        params_list.append(ad_net.parameters())
    optimizer = torch.optim.Adam(chain(*params_list),
                                 lr=args.lr,
                                 weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=len(loader['source']) * 5,
                                                gamma=0.6)

    # * ========== train ==========
    train_history = defaultdict(list)
    eval_history = defaultdict(list)

    for epoch in range(int(args.max_epoch)):
        # * ========== eval ==========
        celltype_weights, eval_history = eval_step(feature_extractor=feature_extractor,
                                                   loader=loader,
                                                   history=eval_history,
                                                   device=args.device,
                                                   args=args)

        #  * ========== save check ==========
        save_flag, stop_flag = model_save_check(history=eval_history,
                                                metric_name='mean_ent',
                                                tolerance_count=args.tolerance)
        if save_flag:
            torch.save(feature_extractor.state_dict(), os.path.join(args.save_folder, 'best_model.pth'))
        if stop_flag or (epoch >= args.max_epoch):
            break

        #  * ========== train ==========
        train_history = train_step(feature_extractor=feature_extractor,
                                   ad_nets=ad_nets,
                                   celltype_weights=celltype_weights,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   loader=loader,
                                   history=train_history,
                                   epoch=epoch,
                                   args=args)

        tag = 0 if len(str(epoch)) == 1 else ''
        print(f"epoch:{tag}{epoch}",
              f"total_loss:{train_history['total_loss'][-1]:.4f}",
              f"s_loss:{train_history['s_loss'][-1]:.4f}",
              f"t_loss:{train_history['t_loss'][-1]:.4f}",
              f"transfer_loss:{train_history['transfer_loss'][-1]:.4f}",
              f"margin_loss:{train_history['margin_loss'][-1]:.4f}",
              f"mean_ent:{eval_history['mean_ent'][-1]:.4f}",
              f"best_idx:{eval_history['best_index']}",
              sep='\t')

    # 1 history
    del eval_history['best_index']
    celltype_weights = eval_history['celltype_weights'] / np.max(eval_history['celltype_weights'])
    celltype_weights_dict = {args.ce.label_encoder.classes_[i]: celltype_weights[i] for i in range(len(celltype_weights))}
    del eval_history['celltype_weights']
    train_history.update(eval_history)
    train_history = {key: value[:epoch] for key, value in train_history.items()}
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(os.path.join(args.save_folder, 'history.csv'))

    # generate code
    s_raw_data, t_raw_data = get_scanpy_adata(args.data_type, seed=args.seed, model=args.model)
    feature_extractor.load_state_dict(torch.load(os.path.join(args.save_folder, 'best_model.pth')))
    feature_extractor.eval()
    s_x, s_y, s_b = loader['source'].dataset.tensors[0], loader['source'].dataset.tensors[1], loader['source'].dataset.tensors[2].to(int)
    t_x, _, t_b = loader['target'].dataset.tensors[0], loader['target'].dataset.tensors[1].to(int), loader['target'].dataset.tensors[2].to(int)
    s_z, s_yhat = feature_extractor(s_x.to(args.device), s_b.to(args.device))
    t_z, t_yhat = feature_extractor(t_x.to(args.device), t_b.to(args.device))

    st_yhat = torch.cat((s_yhat, t_yhat), dim=0)
    st_cell_entropy = loss.Entropy(torch.nn.Softmax(dim=1)(st_yhat))
    st_cell_r = 1 + torch.exp(-st_cell_entropy)
    st_cell_w = torch.ones(st_yhat.size(0))
    st_cell_w[:s_raw_data.shape[0]] = st_cell_r[:s_raw_data.shape[0]] * torch.tensor(celltype_weights[s_y.nonzero()[:, 1].long()]).to(args.device)
    st_cell_w[s_raw_data.shape[0]:] = st_cell_r[s_raw_data.shape[0]:]

    s_z, t_z, s_yhat, t_yhat = s_z.cpu().detach().numpy(), t_z.cpu().detach().numpy(), s_yhat.cpu().detach().numpy(), t_yhat.cpu().detach().numpy()
    _, _, t_yinthat = np.argmax(s_y, axis=1), np.argmax(s_yhat, axis=1), np.argmax(t_yhat, axis=1)
    t_softmax_arr = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), axis=1, arr=t_yhat)

    # 2 t_prob_result.csv
    t_prob_result = pd.DataFrame(t_softmax_arr, index=t_raw_data.obs_names, columns=args.ce.label_encoder.classes_)
    t_prob_result.to_csv(os.path.join(args.save_folder, 't_prob_result.csv'))

    # 3 st_z_result
    t_ycharhat = args.ce.label_encoder.inverse_transform(t_yinthat)
    st_z_result = pd.DataFrame(np.concatenate([s_z, t_z], axis=0), index=np.concatenate([s_raw_data.obs_names, t_raw_data.obs_names], axis=0), columns=[f'z{i+1}' for i in range(s_z.shape[1])])
    st_z_result.to_csv(os.path.join(args.save_folder, 'st_z_result.csv'))

    # 4 st_umap_result
    if args.need_umap:
        print('Calculating UMAP...\nThis may take a few minutes...')
        st_umap_embedding = umap.UMAP(n_neighbors=15, min_dist=0.8, metric='correlation', transform_seed=args.seed).fit_transform(st_z_result.values)  # min_dist=0.8
        st_umap_result = pd.DataFrame(st_umap_embedding, index=st_z_result.index, columns=['umap1', 'umap2'])
        st_umap_result.to_csv(os.path.join(args.save_folder, 'st_umap_result.csv'))
    else:
        print('Skip UMAP calculation...')

    # 5 st_result
    t_raw_data.obs['pred_celltype'] = t_ycharhat
    st_result = pd.concat([s_raw_data.obs, t_raw_data.obs], axis=0)
    st_result['pred_celltype_prob'] = st_result['pred_celltype'].map(celltype_weights_dict)
    st_result['pred_cell_prob'] = t_prob_result.max(axis=1)
    st_result['cell_weights'] = st_cell_w.detach().numpy()
    st_result.to_csv(os.path.join(args.save_folder, 'st_result.csv'))

    console = Console()
    output = [
        f"All results are saved in: {args.save_folder}",
        "1. st_result.csv",  # : source and target dataset result
        "2. history.csv",  # : training history
        "3. best_model.pth",  # : best model
        "4. st_z_result.csv",  # : concatenated source and target embeddings
        "5. t_prob_result.csv",  # : target dataset prediction probabilities
        "6. st_umap_result.csv"]  # : UMAP embeddings of concatenated source and target embeddings

    panel = Panel.fit("\n".join(output), title=f"Unimap {args.data_type.upper()} Train Finished")
    console.print(panel)
    t_result = st_result[st_result.domain == 'target']
    acc = np.sum(t_result['celltype'] == t_result['pred_celltype']) / len(t_result)
    print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train unimap')
    parser.add_argument('--model', type=str, default='unimap')
    parser.add_argument('--method', type=str, default='union', choices=['union', 'inter', 's_union', 's_inter', 'st_union'])
    parser.add_argument('--var_name', type=str, default='highly_variable')
    parser.add_argument('--need_umap', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_feature', type=int, default=128)
    parser.add_argument('--tolerance', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=50)  # 4_species 50 # mg 15 # pbmc 3
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--data_type', type=str, default='pbmc9')
    parser.add_argument('--drop', type=float, default=0.10)  # default=0.15
    parser.add_argument('--conf_thres', type=float, default=0.90)  # default=0.95
    parser.add_argument('--trans_loss_w', type=float, default=0.5)  # default=0.5
    parser.add_argument('--t_loss_w', type=float, default=0.5)  # default=0.5
    parser.add_argument('--margin_w', type=float, default=1)  # choices=[0, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8]

    parser.add_argument('--epoch', type=int, default=25000)  # 200
    parser.add_argument('--focal_alpha', type=float, default=1)
    parser.add_argument('--focal_gamma', type=float, default=2)
    # parser.add_argument('--dir_name', type=str, default='2023')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # max_epoch_dict = {'mg': 15, 'pbmc40': 3, 'default': 50}
    # args.max_epoch = max_epoch_dict.get(args.data_type, max_epoch_dict['default'])

    args.save_folder = os.path.join('results', args.data_type, args.model, str(args.seed))

    safe_make_dir(args.save_folder)
    train_unimap(args)
