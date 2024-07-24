library(SeuratData)
library(Seurat)
library(ggplot2)
library(reticulate)

rm(list = ls())
setwd("/home/hht/Myapps/Reproducing UniMap")
.libPaths('/home/hht/.conda/envs/seurat5/lib/R/library')
use_condaenv("/home/hht/.conda/envs/unimap")
data_list <- import("data_list")
getwd()
.libPaths()

get_st_ad <- function(datatype = "pbmc9", seed = 2023){
  st_ad <- data_list$get_scanpy_adata(datatype = datatype)
  rownames(st_ad[[2]]$var) <- rownames(st_ad[[1]]$var)
  return(st_ad)
}  # 获得数据
get_st_list <- function(st_ad){
  s_ad <- st_ad[[1]]
  t_ad <- st_ad[[2]]
  s_se <- CreateSeuratObject(counts = t(as.matrix(s_ad$X)), meta.data = s_ad$obs)
  t_se <- CreateSeuratObject(counts = t(as.matrix(t_ad$X)), meta.data = t_ad$obs)
  rownames(s_se) <- row.names(s_ad$var)
  rownames(t_se) <- row.names(t_ad$var)
  VariableFeatures(s_se) <- row.names(s_ad$var)
  VariableFeatures(t_se) <- row.names(t_ad$var)
  s_se@assays$RNA$data <- s_se@assays$RNA$counts
  t_se@assays$RNA$data <- t_se@assays$RNA$counts
  return(c(s_se, t_se))
}  # 获得数据
get_seurat_result <- function(st_se, seed = 2023){
  s_se <- st_se[[1]]
  t_se <- st_se[[2]]

  s_se[["RNA"]] <- split(s_se[["RNA"]], f = s_se$batch)
  s_se <- ScaleData(s_se)
  s_se <- RunPCA(s_se, seed.use=seed)
  if (length(unique(s_se$batch)) > 1){
    s_se <- IntegrateLayers(object = s_se, method = CCAIntegration, orig.reduction = "pca",
                            new.reduction = "integrated.cca", verbose = FALSE, features = VariableFeatures(s_se))
  }
  anchors <- FindTransferAnchors(reference = s_se, query = t_se, dims = 1:30, reference.reduction = "pca")
  pred_celltype <- TransferData(anchorset = anchors, refdata = s_se$celltype, dims = 1:30)
  t_se <- AddMetaData(t_se, metadata = pred_celltype)
  if (length(unique(s_se$batch)) > 1){
    s_se <- RunUMAP(s_se, dims = 1:30, reduction = "integrated.cca", return.model = TRUE, seed.use=seed)
  } else {
    s_se <- RunUMAP(s_se, dims = 1:30, reduction = "pca", return.model = TRUE, seed.use=seed)
  }
  t_se <- MapQuery(anchorset = anchors, reference = s_se, query = t_se,
                   refdata = list(celltype = "celltype"), reference.reduction = "pca", reduction.model = "umap")
  return(c(s_se, t_se))
}  # 跑seurat
save_result <- function(st_ad, st_se_result, datatype, seed){
  # 创建文件夹
  result_dir <- paste0("results/", datatype, '/seurat/', seed)
  if (!file.exists(result_dir)){
    dir.create(result_dir)
    cat(result_dir, 'Created!')
  } else{
    cat(result_dir, 'Exists!')
  }
  # save
  s_ad <- st_ad[[1]]
  t_ad <- st_ad[[2]]
  s_se_result <- st_se_result[[1]]
  t_se_result <- st_se_result[[2]]
  
  # st_result.csv
  s_result_df <- data.frame(row.names = rownames(s_ad$obs),
                            celltype = s_ad$obs$celltype,
                            pred_celltype = s_ad$obs$celltype,
                            batch = s_ad$obs$batch,
                            domain = s_ad$obs$domain)
  t_result_df <- data.frame(row.names = rownames(t_ad$obs),
                            celltype = t_ad$obs$celltype,
                            pred_celltype = t_se_result$predicted.id,
                            batch = t_ad$obs$batch,
                            domain = t_ad$obs$domain)
  st_result_df <- rbind(s_result_df, t_result_df)
  write.csv(st_result_df, file = paste0(result_dir, "/st_result.csv"), row.names = TRUE)
  
  # st_z_result.csv

  st_z_result_df <- rbind(s_se_result@reductions$pca@cell.embeddings[, 1:30], t_se_result@reductions$ref.pca@cell.embeddings)
  colnames(st_z_result_df) <- paste0('z', 1:ncol(st_z_result_df))
  rownames(st_z_result_df) <- c(rownames(s_ad$obs), rownames(t_ad$obs))
  write.csv(st_z_result_df, file = paste0(result_dir, "/st_z_result.csv"), row.names = TRUE)
  
  # st_umap_result.csv
  st_umap_result_df <- rbind(s_se_result@reductions$umap@cell.embeddings, t_se_result@reductions$ref.umap@cell.embeddings)
  colnames(st_umap_result_df) <- paste0("umap", 1:ncol(st_umap_result_df))
  rownames(st_umap_result_df) <- c(rownames(s_ad$obs), rownames(t_ad$obs))
  write.csv(st_umap_result_df, file = paste0(result_dir, "/st_umap_result.csv"), row.names = TRUE)
}  # 保存数据

########## 执行
datatype_list <- c("pbmc9", "pbmc40", "cross_species", "mg")
seed_list <- c(2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032)


datatype_list <- c("pbmc9")
seed_list <- c(2023)

for (datatype in datatype_list){
  for (seed in seed_list){
    st_ad <- get_st_ad(datatype, seed = seed)
    st_se <- get_st_list(st_ad)
    st_se_result <- get_seurat_result(st_se, seed = seed)
    save_result(st_ad, st_se_result, datatype, seed = seed)
  }
}

# datatype <- "bc15"
# st_ad <- get_st_ad(datatype)
# st_se <- get_st_list(st_ad)
# st_se_result <- get_seurat_result(st_se)
# save_result(st_ad, st_se_result, datatype)




