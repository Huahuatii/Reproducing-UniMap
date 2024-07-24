# import json
# import pickle
import umap
import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pyensembl import EnsemblRelease
from typing import TypeVar
from rich.console import Console
from rich.table import Table
import scipy
from sklearn.neighbors import NearestNeighbors
import scib


# * 4 score used in benchmark
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


def shannons_score(adata, pred_label_key='pred_celltype', label_key='celltype', mode='weighted average'):
    pred_ct = adata.obs[pred_label_key]
    pred_ct_dict = {ct: adata.obs[adata.obs[pred_label_key] == ct] for ct in pred_ct.unique()}
    shannons = []
    for pred_ct_df in pred_ct_dict.values():
        pred_celltype_proportions = pred_ct_df[label_key].value_counts(normalize=True)
        shannon_metric = -np.sum(pred_celltype_proportions * np.log(pred_celltype_proportions))
        shannons.append(shannon_metric)
    if mode == 'average':
        pred_ct_proportions = [1 / len(adata.obs[pred_label_key].value_counts(normalize=True))] * len(adata.obs[pred_label_key].value_counts(normalize=True))
    if mode == 'weighted average':
        pred_ct_proportions = list(adata.obs[pred_label_key].value_counts(normalize=True))  # 加权平均
    # ct_proportions = [1] * len(adata.obs[label_key].value_counts(normalize=True))  # 平均
    shannon_index = sum([cp * sn for cp, sn in zip(pred_ct_proportions, shannons)])
    shannon_index = (np.log(len(adata.obs[label_key].value_counts())) - shannon_index) / np.log(len(adata.obs[label_key].value_counts()))
    return shannon_index


def acc(adata, pred_label_key='pred_celltype', label_key='celltype'):
    return accuracy_score(adata.obs[label_key], adata.obs[pred_label_key])


def f1(adata, pred_label_key='pred_celltype', label_key='celltype'):
    return f1_score(adata.obs[label_key], adata.obs[pred_label_key], average='weighted')  # None



Anndata = TypeVar('Anndata')
color40 = ['#6baed6', '#fd8d3c', '#74c476', '#9e9ac8', '#969696', '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
           '#9ecae1', '#fdae6b', '#a1d99b', '#bcbddc', '#bdbdbd', '#6b6ecf', '#b5cf6b', '#e7ba52', '#d6616b', '#ce6dbd',
           '#c6dbef', '#fdd0a2', '#c7e9c0', '#dadaeb', '#d9d9d9', '#9c9ede', '#cedb9c', '#e7cb94', '#e7969c', '#de9ed6',
           '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173']

color20 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
           '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',]

color60 = ['#6baed6', '#fd8d3c', '#74c476', '#9e9ac8', '#969696', '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
           '#9ecae1', '#fdae6b', '#a1d99b', '#bcbddc', '#bdbdbd', '#6b6ecf', '#b5cf6b', '#e7ba52', '#d6616b', '#ce6dbd',
           '#c6dbef', '#fdd0a2', '#c7e9c0', '#dadaeb', '#d9d9d9', '#9c9ede', '#cedb9c', '#e7cb94', '#e7969c', '#de9ed6',
           '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
           "#ff0000", "#ff6d00", "#ffa500", "#ffc000", "#ffe100", "#b6d957", "#a7e3a7", "#7cbc5e", "#389c90", "#4a86e8",
           "#8543e0", "#a2c8ec", "#72d4e4", "#fad46b", "#fdab9f", "#b5b5b5", "#f3f3f3", "#d9d9d9", "#595959", "#262626"]


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


# def change_GENE_to_ID_for_dfcols(gex_dataframe: pd.DataFrame) -> pd.DataFrame:
#     result_list = []
#     gene_names_list = list(gex_dataframe.columns)
#     ensembl104 = EnsemblRelease(104)
#     ensembl75 = EnsemblRelease(75)
#     cache = {}
#     for gene_name in gene_names_list:
#         if gene_name in cache:
#             result_list.append(cache[gene_name])
#             continue
#         try:
#             gene104 = ensembl104.genes_by_name(gene_name)
#             gene_id = gene104[0].gene_id
#             result_list.append(gene_id)
#             cache[gene_name] = gene_id
#         except Exception:
#             try:
#                 gene75 = ensembl75.genes_by_name(gene_name)
#                 gene_id = gene75[0].gene_id
#                 result_list.append(gene_id)
#                 cache[gene_name] = gene_id
#             except Exception:
#                 result_list.append(gene_name)

#     ID_count = len(
#         [string for string in result_list if string.startswith('ENSG')])
#     name_count = len(result_list) - ID_count
#     gex_dataframe.columns = result_list
#     print('-------------------------RESULT-------------------------')
#     print(f"Number of strings not starting with 'ENSG': {name_count}")
#     print(f"Number of strings starting with 'ENSG': {ID_count}")
#     print(f"Number of total strings: {len(gene_names_list)}")
#     return gex_dataframe


# def change_GENE_to_ID_list(gene_names_list: list, species='human') -> list:
#     result_list = []
#     ensembl104 = EnsemblRelease(104)
#     ensembl75 = EnsemblRelease(75)
#     GRCm38 = EnsemblRelease(104, species='mouse')
#     cache = {}
#     if species == 'human':
#         for gene_name in gene_names_list:
#             if gene_name in cache:
#                 result_list.append(cache[gene_name])
#                 continue
#             try:
#                 gene104 = ensembl104.genes_by_name(gene_name)
#                 gene_id = gene104[0].gene_id
#                 result_list.append(gene_id)
#                 cache[gene_name] = gene_id
#             except Exception:
#                 try:
#                     gene75 = ensembl75.genes_by_name(gene_name)
#                     gene_id = gene75[0].gene_id
#                     result_list.append(gene_id)
#                     cache[gene_name] = gene_id
#                 except Exception:
#                     result_list.append(gene_name)
#     elif species == 'mouse':
#         for gene_name in gene_names_list:
#             if gene_name in cache:
#                 result_list.append(cache[gene_name])
#                 continue
#             try:
#                 gene104 = GRCm38.genes_by_name(gene_name)
#                 gene_id = gene104[0].gene_id
#                 result_list.append(gene_id)
#                 cache[gene_name] = gene_id
#             except Exception:
#                 try:
#                     gene75 = ensembl75.genes_by_name(gene_name)
#                     gene_id = gene75[0].gene_id
#                     result_list.append(gene_id)
#                     cache[gene_name] = gene_id
#                 except Exception:
#                     result_list.append(gene_name)

#     ID_count = len(
#         [string for string in result_list if string.startswith('ENSG')])
#     name_count = len(result_list) - ID_count
#     print(f"Number of strings not starting with 'ENSG': {name_count}")
#     print(f"Number of strings starting with 'ENSG': {ID_count}")
#     print(f"Number of total strings: {len(gene_names_list)}")
#     return result_list


# def get_umap_embedding(X: np.ndarray):
#     reducer = umap.UMAP(random_state=2023)
#     embedding = reducer.fit_transform(X)
#     return embedding


# def umap_visualization(embedding: np.ndarray, labels: list, fig_label: str = 'UMAP_Result', colors=[], legend='in', figsize=(7, 7), ax=None, *args, **kargs) -> None:
#     cmap_colors = color20 if len(set(labels)) <= 20 else color60
#     cmap_colors = cmap_colors if len(colors) == 0 else colors
#     cmap_colors = cmap_colors[:len(set(labels))]
#     sns.set(style='white', context='notebook', rc={'figure.figsize': figsize, 'legend.markerscale': 4, 'font.size': 18})
#     ncol = 1 if len(set(labels)) < 16 else (len(set(labels)) - 1) // 15 + 1

#     if ax is None:
#         fig, ax = plt.subplots(figsize=figsize)  # 如果没有传入ax，则创建一个新的Figure和Axes
#     else:
#         fig = ax.figure  # 如果传入了ax，则使用传入的Axes对象的Figure

#     scatterplot = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette=cmap_colors, legend='full', alpha=kargs.get('alpha', 0.9), s=4, edgecolor='none', ax=ax)
#     ax.set_yticks([])  # 关闭y轴刻度线
#     ax.set_xticks([])  # 关闭x轴刻度线
#     ax.set_title(fig_label)

#     if legend == 'out':
#         ax.legend(bbox_to_anchor=(1, 1), fontsize=kargs.get('fontsize', 18), loc=2, handlelength=1, handletextpad=0.2, borderpad=0.5, frameon=True, framealpha=0, borderaxespad=0., ncol=ncol)
#     elif legend == 'no':
#         ax.legend().remove()
#     elif legend == 'bar':
#         ax.legend().remove()
#         plt.colorbar(scatterplot.collections[0])
#         scatterplot.collections[0].set_cmap(cmap_colors)


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


# lr scheduler
def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i += 1

    return optimizer


schedule_dict = {"inv": inv_lr_scheduler}

'''
def shannons_score(st_result, col_name='pred_celltype', weight_col='celltype'):

    def split_df(df, col_name):
        col_classes = df[col_name].unique()
        df_dict = {}
        for col_class in col_classes:
            df_dict[col_class] = df[df[col_name] == col_class]
        return df_dict

    def calculate_shannons(relative_abundance):
        shannon_index = -np.sum(relative_abundance * np.log(relative_abundance))
        return shannon_index

    t_result = st_result[st_result['domain'] == 'target']
    celltype_dicts = split_df(t_result, col_name=col_name)
    shannons = []
    for celltype_df in celltype_dicts.values():
        pred_celltype_proportions = celltype_df.celltype.value_counts(normalize=True)
        shannons.append(calculate_shannons(pred_celltype_proportions))
    celltype_proortions = list(t_result.groupby(weight_col).size() / t_result.shape[0])
    weighted_shannon_index = sum([cp * sn for cp, sn in zip(celltype_proortions, shannons)])
    return weighted_shannon_index
'''

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
        self.color20 = color20
        self.color40 = color40
        self.color60 = color60


    def get_history(self, figsize=(16, 4)):
        if self.history is not None:
            self.history.plot(figsize=figsize)
        else:
            print('No history result.')

    '''
    def get_eval_result(self, t_data_type='all', mode='all'):
        if t_data_type == 'in_ref':
            self.t_result_eval = self.t_result[self.t_result['celltype'].isin(set(self.s_result.celltype))]
        # elif t_data_type == 'out_ref':
            # self.t_result_eval = self.t_result[~self.t_result['celltype'].isin(set(self.s_result.celltype))]
        else:
            self.t_result_eval = self.t_result

        if self.st_z_result is not None:
            self.t_z_result = self.t_z_result.loc[self.t_result_eval.index]
            self.t_umap_result = self.t_umap_result.loc[self.t_result_eval.index]

        t_celltype, t_pred_celltype = self.t_result_eval['celltype'].to_list(), self.t_result_eval['pred_celltype'].to_list()
        classification_report_dict = classification_report(t_celltype, t_pred_celltype, digits=4, zero_division=0, output_dict=True)

        # eval_results
        f1_score = classification_report_dict['weighted avg']['f1-score']
        acc = classification_report_dict['accuracy']
        pre = classification_report_dict['weighted avg']['precision']
        recall = classification_report_dict['weighted avg']['recall']
        nmi = normalized_mutual_info_score(t_celltype, t_pred_celltype)
        ari = adjusted_rand_score(t_celltype, t_pred_celltype)

        if mode == "all":
            shannons = 1 - shannons_score(self.st_result)
            if self.st_umap_result is None:
                over_correction = 0
                silhouette = 0
            else:
                over_correction = 1 - overcorrection_score(self.t_umap_result, self.t_result['pred_celltype'])
                silhouette = silhouette_score(self.st_umap_result, self.st_result['pred_celltype'])
        else :
            shannons = over_correction = silhouette = 0

        self.evaluation_index = {'model': self.model,
                                 'dataset': self.dataset,
                                 'acc': acc,
                                 'pre': pre,
                                 'recall': recall,
                                 'f1': f1_score,
                                 'nmi': nmi,
                                 'ari': ari,
                                 'over_correction_score': over_correction,
                                 'silhouette_score': silhouette,
                                 'weighted_shannons_score': shannons
                                 }
        evaluation_index_df = pd.DataFrame(self.evaluation_index, index=[f'{self.model}_{self.dataset}'])
        return evaluation_index_df



    # def get_umap(self, mode='manual', embedding=None, labels=None, fig_label='UMAP RESULT', colors=[], legend='out', figsize=(5, 5), ax=None, *args, **kargs):
    #     if mode == 'manual':
    #         return umap_visualization(embedding=embedding, labels=labels, fig_label=fig_label, colors=colors, legend=legend, figsize=figsize, *args, **kargs)
    #     elif mode == 'domain':
    #         fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    #         plt.subplots_adjust(wspace=0.4)
    #         umap_visualization(embedding=self.st_umap_result.values, labels=self.st_result['domain'], fig_label='Domain Result', colors=colors, legend='out', figsize=figsize, ax=axs[0], *args, **kargs)
    #         umap_visualization(embedding=self.s_umap_result.values, labels=self.s_result['batch'], fig_label='Source Batch Result', colors=colors, legend='out', figsize=figsize, ax=axs[1], *args, **kargs)
    #         umap_visualization(embedding=self.t_umap_result.values, labels=self.t_result['batch'], fig_label='Target Batch Result', colors=colors, legend='out', figsize=figsize, ax=axs[2], *args, **kargs)
    #     elif mode == 'target':
    #         fig, axs = plt.subplots(2, 2, figsize=(13, 12))
    #         plt.subplots_adjust(wspace=0.8)
    #         extra_celltype = len(self.s_result) * ['reference'] + self.t_result['celltype'].to_list()
    #         extra_pred_celltype = len(self.s_result) * ['reference'] + self.t_result['pred_celltype'].to_list()
    #         extra_colors = ['#d5d5d5'] + (color20 if len(set(self.t_result['pred_celltype'])) <= 20 else color60)
    #         umap_visualization(embedding=self.s_umap_result.values, labels=self.s_result['celltype'], fig_label='Source Celltype Result', colors=colors, legend='out', figsize=figsize, ax=axs[0, 0], *args, **kargs)
    #         umap_visualization(embedding=self.st_umap_result.values, labels=extra_celltype, fig_label='Target Celltype Result', colors=extra_colors, legend='out', figsize=figsize, ax=axs[1, 0], *args, **kargs)
    #         umap_visualization(embedding=self.st_umap_result.values, labels=extra_pred_celltype, fig_label='Target Pred Celltype Result', colors=extra_colors, legend='out', figsize=figsize, ax=axs[1, 1], *args, **kargs)
    #         axs[0, 1].set_yticks([])
    #         axs[0, 1].set_xticks([])
    #         # umap_visualization(embedding=self.t_umap_result.values, labels=self.t_result['celltype'], fig_label='Target Celltype Result', colors=colors, legend='no', figsize=figsize, ax=axs[1, 1], *args, **kargs)
    #         # umap_visualization(embedding=self.t_umap_result.values, labels=self.t_result['pred_celltype'], fig_label='Target Pred Celltype Result', colors=colors, legend='out', figsize=figsize, ax=axs[1, 2], *args, **kargs)
    #     elif mode == 'prob':
    #         fig, axs = plt.subplots(1, 2, figsize=(17, 5))
    #         umap_visualization(embedding=self.s_umap_result.values, labels=self.s_result['pred_celltype_prob'], fig_label='Source Weights Result', colors=kargs.get('color', 'plasma'), legend='bar', figsize=figsize, ax=axs[0], alpha=0.6, *args, **kargs)
    #         umap_visualization(embedding=self.t_umap_result.values, labels=self.t_result['pred_cell_prob'], fig_label='Target Prob Result', colors=kargs.get('color', 'viridis'), legend='bar', figsize=figsize, ax=axs[1], alpha=0.6, *args, **kargs)
    '''

    def get_evaluation_index(self, raw_t_data=None):
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
        metrics.loc[self.model, 'average_shannons_score'] = shannons_score(unimap_ad, pred_label_key='pred_celltype', label_key='celltype', mode='average')

        # * Batch correction
        metrics.loc[self.model, 'ilisi_graph'] = scib.me.ilisi_graph(unimap_ad, batch_key="batch", type_="embed", use_rep="X_emb")
        metrics.loc[self.model, 'silhouette_batch'] = scib.me.silhouette_batch(unimap_ad, batch_key="batch", label_key="celltype", embed="X_emb")
        metrics.loc[self.model, 'kBET'] = scib.me.kBET(unimap_ad, batch_key="batch", label_key="celltype", type_="embed", embed="X_emb")
        # metrics.loc[self.model, 'pcr_comparison'] = scib.me.pcr_comparison(raw_t_data, unimap_ad, covariate="batch", embed="X_emb")
        metrics.loc[self.model, 'graph_connectivity'] = scib.me.graph_connectivity(unimap_ad, label_key="celltype")
        return metrics


    def get_heatmap(self, title='', ct_labels=None, pd_ct_labels=None, axs=None, percentage_direction=0):
        cm = confusion_matrix(self.t_result['celltype'], self.t_result['pred_celltype'])
        celltype_labels = sorted(set(self.t_result['celltype']) | set(self.t_result['pred_celltype']))
        cm = pd.DataFrame(cm, index=celltype_labels, columns=celltype_labels)

        if percentage_direction == 0:
            # 计算每行的总和并求百分比
            row_sums = cm.sum(axis=1)
            cm = cm.div(row_sums, axis=0)
        elif percentage_direction == 1:
            # 计算每列的总和并求百分比
            column_sums = cm.sum(axis=0)
            cm = cm.div(column_sums, axis=1)

        if ct_labels is None and pd_ct_labels is None:
            ct_labels = cm.index
            pd_ct_labels = cm.columns

        cm = cm.loc[ct_labels, pd_ct_labels]
        cm = cm.fillna(0)

        ax = sns.heatmap(cm, annot=True, cmap='Oranges', annot_kws={'size': 8}, fmt='.2f', cbar=False, square=True, mask=cm < 0.02, ax=axs)  # 绘制热图
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        rect = patches.Rectangle((0, 0), cm.shape[1], cm.shape[0], linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.text(0.5, 1.02, title, ha='center', va='center', transform=ax.transAxes)
        return ax
    
    def get_cm(self, ct_labels=None, pd_ct_labels=None, percentage_direction=0):
        cm = confusion_matrix(self.t_result['celltype'], self.t_result['pred_celltype'])
        celltype_labels = sorted(set(self.t_result['celltype']) | set(self.t_result['pred_celltype']))
        cm = pd.DataFrame(cm, index=celltype_labels, columns=celltype_labels)

        if percentage_direction == 0:
            # 计算每行的总和并求百分比
            row_sums = cm.sum(axis=1)
            cm = cm.div(row_sums, axis=0)
        elif percentage_direction == 1:
            # 计算每列的总和并求百分比
            column_sums = cm.sum(axis=0)
            cm = cm.div(column_sums, axis=1)

        if ct_labels is None and pd_ct_labels is None:
            ct_labels = cm.index
            pd_ct_labels = cm.columns

        cm = cm.reindex(index=ct_labels, columns=pd_ct_labels, fill_value=0)
        cm = cm.fillna(0)
        # ax = sns.heatmap(cm, annot=True, cmap='Oranges', annot_kws={'size': 8}, fmt='.2f', cbar=False, square=True, mask=cm < 0.02, ax=axs)  # 绘制热图
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # rect = patches.Rectangle((0, 0), cm.shape[1], cm.shape[0], linewidth=3, edgecolor='black', facecolor='none')
        # ax.add_patch(rect)
        # ax.text(0.5, 1.02, title, ha='center', va='center', transform=ax.transAxes)
        return cm


    # def get_prob(self, celltype='celltype', figsize=(5, 5)):
    #     medians = self.t_result.groupby(celltype)['pred_cell_prob'].median().sort_values()
    #     plt.figure(figsize=figsize, dpi=120)
    #     sns.boxenplot(x=celltype, y='pred_cell_prob', data=self.t_result, order=medians.index, palette='viridis', showfliers=False)
    #     # 添加散点
    #     sns.stripplot(x=celltype, y='pred_cell_prob', data=self.t_result, order=medians.index, color='black', size=2, alpha=0.5)
    #     plt.title('Boxplot of Predicted Probabilities by Cell Type', fontsize=16)
    #     # plt.xlabel('Cell Type', fontsize=14)
    #     # plt.ylabel('Predicted Probability', fontsize=14)

    #     # 设置坐标轴标签字体大小
    #     plt.xticks(fontsize=12, rotation=45, ha='right')
    #     plt.yticks(fontsize=12)

    #     # 显示图形
    #     plt.show()

    def remove_spine(self, p):
        p.spines['top'].set_visible(False)
        p.spines['right'].set_visible(False)
        p.spines['left'].set_visible(False)
        p.spines['bottom'].set_visible(False)
        # 删掉p的坐标轴
        p.set_xticks([])
        p.set_yticks([])
        p.set_xlabel('')
        p.set_ylabel('')
        return p

    def umap_visual(self, save_path=None):
        def remove_spine(p):
            p.spines['top'].set_visible(False)
            p.spines['right'].set_visible(False)
            p.spines['left'].set_visible(False)
            p.spines['bottom'].set_visible(False)
            # 删掉p的坐标轴
            p.set_xticks([])
            p.set_yticks([])
            p.set_xlabel('')
            p.set_ylabel('')
            return p

        def umap_afig(umap1, umap2, labels, title, ax, color=None):
            # ncol值为len(set(labels))是15的几倍
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
            p = remove_spine(p)
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
        domain_p = umap_afig(st_umap.iloc[:, 0], st_umap.iloc[:, 1], domain, 'Domain', axs[0, 0])
        batch_p = umap_afig(st_umap.iloc[:, 0], st_umap.iloc[:, 1], batch, 'Batch', axs[0, 1])
        s_ct_p = umap_afig(s_umap.iloc[:, 0], s_umap.iloc[:, 1], s_ct, 'Reference Celltype', axs[1, 0])
        axs[1, 1].remove()
        t_ct_p = umap_afig(st_umap.iloc[:, 0], st_umap.iloc[:, 1], t_ct, 'Query Celltype', axs[2, 0])
        t_pred_ct_p = umap_afig(st_umap.iloc[:, 0], st_umap.iloc[:, 1], t_pred_ct, 'Query Predicted Celltype', axs[2, 1])
        if self.model == 'unimap/2023':
            t_prob_w_p = umap_afig(t_umap.iloc[:, 0], t_umap.iloc[:, 1], t_cell_w, 'Query Predicted Cell Confidence', axs[3, 0], color='viridis')
            t_prob_w_p.legend_.remove()
            s_ct_w_p = umap_afig(s_umap.iloc[:, 0], s_umap.iloc[:, 1], s_ct_w, 'Reference Celltype Weight', axs[3, 1], color='viridis')
            s_ct_w_p.legend_.remove()

        if save_path is not None:
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300, transparent=True)
        plt.show()


# NEED
def get_common_genes(s_data: Anndata, t_data: Anndata, method: str = 'union', var_name: str = 'highly_variable') -> Anndata:
    '''
    This function takes two single-cell datasets as input and returns a new dataset that only contains genes that are common and highly variable in both datasets.
    :param s_data: Single-cell dataset 1
    :param t_data: Single-cell dataset 2
    :return: A new dataset that only contains genes that are common and highly variable in both datasets.
    '''
    # * 对anndata数据进行裁剪，取出两个数据集中高变基因部分的交集
    s_genes = set(s_data.var.index)
    t_genes = set(t_data.var.index)

    s_h_genes = set(s_data.var[s_data.var[var_name] is True].index)
    t_h_genes = set(t_data.var[t_data.var[var_name] is True].index)

    if method == 'union':
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


def get_evaluation_index(st_result, st_umap_result, model='xx', dataset='xx'):
    t_result = st_result[st_result['domain'] == 'target']

    t_celltype, t_pred_celltype = t_result['celltype'].to_list(), t_result['pred_celltype'].to_list()
    classification_report_dict = classification_report(t_celltype, t_pred_celltype, digits=4, zero_division=0, output_dict=True)

    f1_score = classification_report_dict['weighted avg']['f1-score']
    acc = classification_report_dict['accuracy']
    pre = classification_report_dict['weighted avg']['precision']
    recall = classification_report_dict['weighted avg']['recall']
    nmi = normalized_mutual_info_score(t_celltype, t_pred_celltype)
    ari = adjusted_rand_score(t_celltype, t_pred_celltype)
    shannons = 1 - shannons_score(st_result)

    if st_umap_result is None:
        over_correction = 0
        silhouette = 0
    else:
        t_umap_result = st_umap_result[sum(st_result['domain'] == 'source'):]
        over_correction = 1 - overcorrection_score(t_umap_result, t_result['pred_celltype'])
        silhouette = silhouette_score(st_umap_result, st_result['pred_celltype'])

    evaluation_index = {'model': model,
                        'dataset': dataset,
                        'acc': acc,
                        'pre': pre,
                        'recall': recall,
                        'f1': f1_score,
                        'nmi': nmi,
                        'ari': ari,
                        'over_correction_score': over_correction,
                        'silhouette_score': silhouette,
                        'weighted_shannons_score': shannons}
    evaluation_index_df = pd.DataFrame(evaluation_index, index=[f'{model}_{dataset}'])

    return evaluation_index_df


'''
def overcorrection_score(emb, celltype, n_neighbors=100, n_pools=100, n_samples_per_pool=100, seed=2023):
    """
    """
    np.random.seed(seed)
    n_neighbors = min(n_neighbors, len(emb) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(emb)
    kmatrix = nne.kneighbors_graph(emb) - scipy.sparse.identity(emb.shape[0])

    score = 0
    celltype_dict = celltype.value_counts().to_dict()

    for t in range(n_pools):
        indices = np.random.choice(np.arange(emb.shape[0]), size=n_samples_per_pool, replace=False)
        score += np.mean([np.mean(celltype.iloc[kmatrix[i].nonzero()[1]][:min(celltype_dict[celltype.iloc[i]], n_neighbors)] == celltype.iloc[i]) for i in indices])
    return 1-score / float(n_pools)
'''


def get_hv_genes(adata, min_genes=200, min_cells=3, n_top_genes=800, target_sum=1e6):
    adata.var_names_make_unique()
    adata = adata[adata.obs['celltype'] != 'nan', :]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum) # cpm
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    return adata


def pre_process_data(adata, min_genes=200, min_cells=3, n_top_genes=800, target_sum=1e6, batch_name='batch'):
    '''
    逐批次进行标准化
    '''
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


def browse_log(dir, line, thres=0.9):
    params_list = []
    for (idx, filename) in enumerate(os.listdir(dir)):
        if filename.endswith(".log"):
            with open(os.path.join(dir, filename), "r") as file:
                # 读取最后一行
                try:
                    read_line = file.readlines()[line]
                    acc = float(read_line.split('\t')[1].split(':')[1])
                except:
                    acc = 0
                if acc > thres:
                    params_list.append(filename)
                    print(filename, '\t\tacc:', acc, sep=' ')
    return params_list


def remove_ticks(ax, remove_legend=True, titles={'top': '', 'left': ''}):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel(titles['left'], fontsize=26)
    # 设置图片的title
    # ax.set_title(titles['top'], fontsize=22, pad=10)
    if remove_legend:
        ax.legend_.remove()
    # 添加标题文本到右上角位置
    plt.text(0.96, 0.96, titles['top'], fontsize=26, ha='right', va='top', transform=ax.transAxes)
    return ax


def cal_positions(frame, spacing_x=0.02, spacing_y=0.06):
    height = 0.96 / frame[0]
    width = 0.96 / frame[1]
    positions = []

    # 初始化 positions 列表
    for _ in range(frame[0] * frame[1]):
        positions.append([0, 0, width, height])

    # 计算每个图形的位置信息并添加到 positions 列表中
    for i in range(frame[0]):
        for j in range(frame[1]):
            idx = i * frame[1] + j
            positions[idx][0] = 0.1 + j * (width + spacing_x)  # x 起始位置，加上横向间距
            positions[idx][1] = 0.96 - (0.1 + i * (height + spacing_y) + height)  # y 起始位置，加上纵向间距

    return positions


