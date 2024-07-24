import os
import scib
import torch
import scipy
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from typing import TypeVar
from rich.table import Table
from rich.console import Console
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings('ignore')
Anndata = TypeVar('Anndata')


def overcorrection_score(adata, label_key, type_="embed", use_rep="X_emb", n_neighbors=100, n_pools=100, n_samples_per_pool=100, seed=2023):
    np.random.seed(seed)
    n_neighbors = min(n_neighbors, len(adata) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(adata.obsm[use_rep])
    kmatrix = nne.kneighbors_graph(adata.obsm[use_rep]) - scipy.sparse.identity(adata.obsm[use_rep].shape[0])

    score = 0
    celltype_dict = adata.obs[label_key].value_counts().to_dict()

    for t in range(n_pools):
        indices = np.random.choice(np.arange(adata.obsm[use_rep].shape[0]), size=n_samples_per_pool, replace=False)
        score += np.mean([np.mean(adata.obs[label_key].iloc[kmatrix[i].nonzero()[1]][:min(celltype_dict[adata.obs[label_key].iloc[i]], n_neighbors)] == adata.obs[label_key].iloc[i]) for i in indices])
    return 1 - (1- score / float(n_pools))


def shannons_score(adata, pred_label_key='pred_celltype', label_key='celltype'):
    pred_ct = adata.obs[pred_label_key]
    pred_ct_dict = {ct: adata.obs[adata.obs[pred_label_key] == ct] for ct in pred_ct.unique()}
    shannons = []
    for pred_ct_df in pred_ct_dict.values():
        pred_celltype_proportions = pred_ct_df[label_key].value_counts(normalize=True)
        shannon_metric = -np.sum(pred_celltype_proportions * np.log(pred_celltype_proportions))
        shannons.append(shannon_metric)
    pred_ct_proportions = [1 / len(adata.obs[pred_label_key].value_counts(normalize=True))] * len(adata.obs[pred_label_key].value_counts(normalize=True))

    shannon_index = sum([cp * sn for cp, sn in zip(pred_ct_proportions, shannons)])
    shannon_index = (np.log(len(adata.obs[label_key].value_counts())) - shannon_index) / np.log(len(adata.obs[label_key].value_counts()))
    return shannon_index


def acc(adata, pred_label_key='pred_celltype', label_key='celltype'):
    return accuracy_score(adata.obs[label_key], adata.obs[pred_label_key])


def f1(adata, pred_label_key='pred_celltype', label_key='celltype'):
    return f1_score(adata.obs[label_key], adata.obs[pred_label_key], average='weighted')  # None


def model_save_check(history, metric_name, tolerance_count=5):
    save_flag = False
    stop_flag = False
    if 'best_index' not in history:
        history['best_index'] = 0

    if history[metric_name][-1] < history[metric_name][history['best_index']]:
        history['best_index'] = len(history[metric_name]) - 1
        save_flag = True
    if history['best_index'] == 0:
        save_flag = True

    if len(history[metric_name]) - history['best_index'] > tolerance_count and history['best_index'] > 0:
        stop_flag = True
    return save_flag, stop_flag


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
        print(new_folder_name, 'created!\n')
    else:
        print(new_folder_name, 'exists!\n')


def generate_positions(w, h, l, b):
    position = [[l, b+h, w, h], [l+w*1, b+h, w, h], [l+w*2, b+h, w, h], [l+w*3, b+h, w, h], [l+w*4, b+h, w, h], [l+w*5, b+h, w, h],
                [l, b+0, w, h], [l+w*1, b+0, w, h], [l+w*2, b+0, w, h], [l+w*3, b+0, w, h], [l+w*4, b+0, w, h], [l+w*5, b+0, w, h]]
    return position


def get_common_genes(s_data: Anndata, t_data: Anndata, method: str = 'union', var_name: str = 'highly_variable') -> Anndata:

    s_genes = set(s_data.var.index)
    t_genes = set(t_data.var.index)

    s_h_genes = set(s_data.var[s_data.var[var_name] is True].index)
    t_h_genes = set(t_data.var[t_data.var[var_name] is True].index)

    if method == 'union':  # This is the default method
        common_genes = s_h_genes | t_h_genes
        common_genes = s_genes & t_genes & common_genes
    elif method == 'inter':
        common_genes = s_genes & t_genes & t_h_genes & s_h_genes
    elif method == 's_union':
        common_genes = s_genes & t_h_genes
        common_genes = common_genes | s_h_genes
    elif method == 's_inter':
        common_genes = s_h_genes
    elif method == "st_union":
        common_genes = s_h_genes | t_h_genes
    else:
        raise ValueError('Invalid method')
    s_genes = s_genes & common_genes
    t_genes = t_genes & common_genes

    s_data = s_data[:, s_data.var.index.isin(s_genes)]
    t_data = t_data[:, t_data.var.index.isin(t_genes)]
    st_data = sc.concat([s_data, t_data], join='outer', fill_value=0)
    s_data = st_data[st_data.obs['domain'] == 'source']
    t_data = st_data[st_data.obs['domain'] == 'target']
    return s_data, t_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def set_plot_theme():
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": 'Arial',
        'axes.linewidth': 0.5,
        'axes.edgecolor': 'black',
        'figure.dpi': 180,
        # 'axes.facecolor': (1, 1, 1, 0),
        # 'figure.facecolor': (1, 1, 1, 0)
    })


def check_result(dataset, model, root_dir='results', detailed=True):
    result_dir = os.path.join(root_dir, dataset, model)
    console = Console()
    table = Table(show_header=True, header_style="bold cyan")
    table.title = f"Checking {dataset} {model} result"
    table.add_column("File", style="dim", width=20)
    table.add_column("Exist", style="dim", width=8)

    if os.path.exists(os.path.join(result_dir, 'st_result.csv')):
        st_result = pd.read_csv(os.path.join(result_dir, 'st_result.csv'), index_col=0)
        table.add_row("st_result.csv", "✔️", style="green")
    else:
        st_result = None
        table.add_row("st_result.csv", "✖️")

    if os.path.exists(os.path.join(result_dir, 'history.csv')):
        with open(os.path.join(result_dir, 'history.csv')) as file:
            history = pd.read_csv(file, index_col=0)
        table.add_row("history.csv", "✔️", style="green")
    else:
        history = None
        table.add_row("history.csv", "✖️")

    if os.path.exists(os.path.join(result_dir, 'st_z_result.csv')):
        st_z_result = pd.read_csv(os.path.join(result_dir, 'st_z_result.csv'), index_col=0)
        table.add_row("st_z_result.csv", "✔️", style="green")
    else:
        st_z_result = None
        table.add_row("st_z_result.csv", "✖️")

    if os.path.exists(os.path.join(result_dir, 'st_umap_result.csv')):
        st_umap_result = pd.read_csv(os.path.join(result_dir, 'st_umap_result.csv'), index_col=0)
        table.add_row("st_umap_result.csv", "✔️", style="green")
    else:
        st_umap_result = None
        table.add_row("st_umap_result.csv", "✖️")

    if os.path.exists(os.path.join(result_dir, 't_prob_result.csv')):
        t_prob_result = pd.read_csv(os.path.join(result_dir, 't_prob_result.csv'), index_col=0)
        table.add_row("t_prob_result.csv", "✔️", style="green")
    else:
        t_prob_result = None
        table.add_row("t_prob_result.csv", "✖️")
    if detailed:
        console.print(table)
    return st_result, history, st_z_result, st_umap_result, t_prob_result


def pre_process_data(adata, min_genes=200, min_cells=3, n_top_genes=800, target_sum=1e6, batch_name='batch'):
    adata.var_names_make_unique()
    adata = adata[adata.obs['celltype'] != 'nan', :]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    batch_list = adata.obs[batch_name].unique()
    batch_dict = {}
    for i in batch_list:
        i_adata = adata[adata.obs[batch_name] == i, :]
        sc.pp.normalize_total(i_adata, target_sum=target_sum)  # cpm
        sc.pp.log1p(i_adata)
        batch_dict[i] = i_adata
    if len(batch_dict) > 1:
        adata = sc.concat(list(batch_dict.values()), join='outer')
    else:
        adata = batch_dict[batch_list[0]]
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    return adata


def get_common_hvg(adata_list, supplement_list=[]):
    s_genes, t_genes = adata_list[0].var.index, adata_list[1].var.index
    inter_stg = set(s_genes).intersection(t_genes)
    if len(adata_list) > 2:
        for i in range(2, len(adata_list)):
            inter_stg = inter_stg.intersection(adata_list[i].var.index)

    s_hvg, t_hvg = adata_list[0].var.index[adata_list[0].var['highly_variable']], adata_list[1].var.index[adata_list[1].var['highly_variable']]
    union_hvg = set(s_hvg).union(t_hvg).union(*supplement_list)
    return union_hvg & inter_stg


class Label_Encoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.classes_ = None

    def fit(self, labels):
        self.label_encoder.fit(labels)
        self.onehot_encoder.fit(self.label_encoder.transform(labels).reshape(-1, 1))
        self.classes_ = self.label_encoder.classes_

    def transform(self, labels):
        return self.onehot_encoder.transform(self.label_encoder.transform(labels).reshape(-1, 1))

    def get_labels(self, yhat: np.ndarray):
        if self.classes_ is None:
            raise ValueError('You must call `fit` method first')

        if len(yhat.shape) == 2:
            yhat = np.argmax(yhat, axis=1)
        return self.label_encoder.inverse_transform(yhat)


class UnimapResult:
    def __init__(self, dataset, model, root_dir='results', detailed=True) -> None:
        self.dataset = dataset
        self.model = model
        self.detailed = detailed
        self.st_result, self.history, self.st_z_result, self.st_umap_result, self.t_prob_result = check_result(dataset, model, root_dir=root_dir, detailed=detailed)

        s_index = self.st_result[self.st_result['domain'] == 'source'].index
        t_index = self.st_result[self.st_result['domain'] == 'target'].index

        self.s_result = self.st_result.loc[s_index]
        self.t_result = self.st_result.loc[t_index]
        if self.st_z_result is not None:
            self.s_z_result = self.st_z_result.loc[s_index]
            self.t_z_result = self.st_z_result.loc[t_index]
            self.s_umap_result = self.st_umap_result.loc[s_index]
            self.t_umap_result = self.st_umap_result.loc[t_index]
        self.color20 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',]
        self.color40 = ['#6baed6', '#fd8d3c', '#74c476', '#9e9ac8', '#969696', '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
                        '#9ecae1', '#fdae6b', '#a1d99b', '#bcbddc', '#bdbdbd', '#6b6ecf', '#b5cf6b', '#e7ba52', '#d6616b', '#ce6dbd',
                        '#c6dbef', '#fdd0a2', '#c7e9c0', '#dadaeb', '#d9d9d9', '#9c9ede', '#cedb9c', '#e7cb94', '#e7969c', '#de9ed6',
                        '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173']
        self.color60 = self.color40 + ["#ff0000", "#ff6d00", "#ffa500", "#ffc000", "#ffe100", "#b6d957", "#a7e3a7", "#7cbc5e", "#389c90", "#4a86e8",
                                       "#8543e0", "#a2c8ec", "#72d4e4", "#fad46b", "#fdab9f", "#b5b5b5", "#f3f3f3", "#d9d9d9", "#595959", "#262626"]

    def remove_spines(self, p, sp_v=False, remove_legend=False):
        p.spines['top'].set_visible(sp_v)
        p.spines['right'].set_visible(sp_v)
        p.spines['left'].set_visible(sp_v)
        p.spines['bottom'].set_visible(sp_v)
        if remove_legend:
            p.legend_.remove()
        return p

    def remove_ticks(self, p, remove_legend=False):
        p.set_xticks([])
        p.set_yticks([])
        p.set_xlabel('')
        p.set_ylabel('')
        if remove_legend:
            p.legend_.remove()
        return p

    def get_history(self, figsize=(16, 4)):
        if self.history is not None:
            self.history.plot(figsize=figsize)
        else:
            print('No history result.')

    def get_evaluation_index(self):
        metrics = pd.DataFrame()
        unimap_ad = sc.AnnData(X=self.t_z_result.values, obs=self.t_result)
        unimap_ad.obsm['X_emb'] = self.t_z_result.values
        for col in unimap_ad.obs.columns:
            unimap_ad.obs[col] = unimap_ad.obs[col].astype('category')

        # * Biological conservation
        sc.pp.neighbors(unimap_ad, use_rep="X_emb")
        metrics.loc[self.model, 'ari'] = scib.me.ari(unimap_ad, cluster_key="celltype", label_key="pred_celltype") 
        metrics.loc[self.model, 'isolated_labels_f1'] = scib.me.isolated_labels_f1(unimap_ad, batch_key="batch", label_key="celltype", embed="X_emb")
        metrics.loc[self.model, 'clisi_graph'] = scib.me.clisi_graph(unimap_ad, label_key="celltype", type_="embed", use_rep="X_emb")
        metrics.loc[self.model, 'isolated_labels_asw'] = scib.me.isolated_labels_asw(unimap_ad, batch_key="batch", label_key="celltype", embed="X_emb")
        metrics.loc[self.model, 'nmi'] = scib.me.nmi(unimap_ad, cluster_key="celltype", label_key="pred_celltype")
        metrics.loc[self.model, 'silhouette'] = scib.me.silhouette(unimap_ad, label_key="celltype", embed="X_emb")
        metrics.loc[self.model, 'overcorrection_score'] = overcorrection_score(unimap_ad, label_key='celltype', type_="embed", use_rep="X_emb")
        metrics.loc[self.model, 'graph_connectivity'] = scib.me.graph_connectivity(unimap_ad, label_key="celltype")
        metrics.loc[self.model, 'acc'] = acc(unimap_ad, pred_label_key='pred_celltype', label_key='celltype')
        metrics.loc[self.model, 'f1_score'] = f1(unimap_ad, pred_label_key='pred_celltype', label_key='celltype')
        metrics.loc[self.model, 'average_shannons_score'] = shannons_score(unimap_ad, pred_label_key='pred_celltype', label_key='celltype')

        # * Batch correction
        metrics.loc[self.model, 'ilisi_graph'] = scib.me.ilisi_graph(unimap_ad, batch_key="batch", type_="embed", use_rep="X_emb")
        metrics.loc[self.model, 'silhouette_batch'] = scib.me.silhouette_batch(unimap_ad, batch_key="batch", label_key="celltype", embed="X_emb")
        metrics.loc[self.model, 'kBET'] = scib.me.kBET(unimap_ad, batch_key="batch", label_key="celltype", type_="embed", embed="X_emb")
        metrics.loc[self.model, 'graph_connectivity'] = scib.me.graph_connectivity(unimap_ad, label_key="celltype")
        return metrics

    def get_cm(self, ct_labels=None, pd_ct_labels=None, percentage_direction=0):
        cm = confusion_matrix(self.t_result['celltype'], self.t_result['pred_celltype'])
        celltype_labels = sorted(set(self.t_result['celltype']) | set(self.t_result['pred_celltype']))
        cm = pd.DataFrame(cm, index=celltype_labels, columns=celltype_labels)

        if percentage_direction == 0:  # by row
            row_sums = cm.sum(axis=1)
            cm = cm.div(row_sums, axis=0)
        elif percentage_direction == 1:  # by column
            column_sums = cm.sum(axis=0)
            cm = cm.div(column_sums, axis=1)

        if ct_labels is None and pd_ct_labels is None:
            ct_labels = cm.index
            pd_ct_labels = cm.columns

        cm = cm.reindex(index=ct_labels, columns=pd_ct_labels, fill_value=0)
        cm = cm.fillna(0)
        return cm

    def umap_visual(self, save_path=None):
        def umap_afig(umap1, umap2, labels, title, ax, color=None):
            ncol = len(set(labels)) // 20 + 1
            if color is not None:
                color = color
            elif len(set(labels)) < 20:
                color = self.color20
            elif len(set(labels)) < 40:
                color = self.color40
            else:
                color = self.color60
            color = ['#E0E0E0'] + color if 'reference' in set(labels) else color

            p = sns.scatterplot(x=umap1, y=umap2, alpha=0.9, s=1, hue=labels, edgecolor='none', palette=color[:len(set(labels))], ax=ax)
            p.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=ncol, fontsize=14, markerscale=7, frameon=False)
            p = self.remove_spines(p)
            p = self.remove_ticks(p)
            p.set_title(title)
            return p

        # * umap
        s_umap = self.s_umap_result
        t_umap = self.t_umap_result
        st_umap = self.st_umap_result

        # * labels
        domain = self.st_result['domain']
        batch = self.st_result['batch']
        s_ct = self.s_result['celltype']
        t_ct = len(s_ct) * ['reference'] + list(self.t_result['celltype'])
        t_pred_ct = len(s_ct) * ['reference'] + list(self.t_result['pred_celltype'])
        if self.model == 'unimap/2023':
            t_cell_w = self.t_result['pred_cell_prob']
            s_ct_w = self.s_result['pred_celltype_prob'].rank(method='dense', ascending=False)

        fig, axs = plt.subplots(4, 2, figsize=(20, 30))
        plt.subplots_adjust(wspace=1, hspace=0.2)
        umap_afig(st_umap.iloc[:, 0], st_umap.iloc[:, 1], domain, 'Domain', axs[0, 0])  # domain_p
        umap_afig(st_umap.iloc[:, 0], st_umap.iloc[:, 1], batch, 'Batch', axs[0, 1])  # batch_p
        umap_afig(s_umap.iloc[:, 0], s_umap.iloc[:, 1], s_ct, 'Reference Celltype', axs[1, 0])  # s_ct_p
        axs[1, 1].remove()
        umap_afig(st_umap.iloc[:, 0], st_umap.iloc[:, 1], t_ct, 'Query Celltype', axs[2, 0])  # t_ct_p
        umap_afig(st_umap.iloc[:, 0], st_umap.iloc[:, 1], t_pred_ct, 'Query Predicted Celltype', axs[2, 1])  # t_pred_ct_p
        if self.model == 'unimap/2023':
            t_prob_w_p = umap_afig(t_umap.iloc[:, 0], t_umap.iloc[:, 1], t_cell_w, 'Query Predicted Cell Confidence', axs[3, 0], color='viridis')
            t_prob_w_p.legend_.remove()
            s_ct_w_p = umap_afig(s_umap.iloc[:, 0], s_umap.iloc[:, 1], s_ct_w, 'Reference Celltype Weight', axs[3, 1], color='viridis')
            s_ct_w_p.legend_.remove()

        if save_path is not None:
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300, transparent=True)
        plt.show()

    def umap_genex(self, adata, genelist, celltype=False):
        fig, axs = plt.subplots(figsize=(185 / 25.4, 250 / 25.4))
        axs.remove()

        positions = []
        rows, cols = 6, 5
        l, b, w, h, x = 0.04, 0.01, 0.12, 0.12 / (250/185), 0.03
        for i in range(rows):
            for j in range(cols):
                left = l + j * (w + x)
                bottom = b + (rows - 1 - i) * (h + x)
                positions.append((left, bottom, w, h))
        # positions = positions[::-1]

        for idx, gene in enumerate(genelist):
            custom_cmap = 'viridis'

            p = sns.scatterplot(x=adata.obsm['X_umap'][:, 0], y=adata.obsm['X_umap'][:, 1], alpha=0.9, s=0.4, hue=adata.X[:, adata.var_names.get_loc(gene)].toarray().flatten(), edgecolor='none', palette=custom_cmap, ax=fig.add_axes(positions[idx]), rasterized=True)
            
            p.set_title(gene, fontsize=6, pad=3)
            self.remove_ticks(p, remove_legend=True)

            bar_p = [positions[idx][0]+w+0.002, positions[idx][1], 0.005, h]
            cbar_ax1 = fig.add_axes(bar_p)
            norm1 = plt.Normalize(vmin=0, vmax=6)
            sm1 = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm1)
            sm1.set_array([])
            cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
            cbar1.set_ticks([0, 3, 6])
            cbar1.set_ticklabels(['0', '3', '6'])
            cbar1.ax.tick_params(labelsize=5, length=1.5, pad=1, width=0.5)

        if celltype:
            p = sns.scatterplot(x=adata.obsm['X_umap'][:, 0], y=adata.obsm['X_umap'][:, 1], alpha=0.9, s=0.4, hue=adata.obs[celltype], edgecolor='none', palette=self.color20, ax=fig.add_axes(positions[len(genelist)]), rasterized=True)
            self.remove_ticks(p)
            p.set_title(celltype, fontsize=6, pad=3)
            p.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, markerscale=2, fontsize=5, labelspacing=0.3, handletextpad=0.2)
        return fig
