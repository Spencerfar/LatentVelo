import torch as th
import numpy as np
import scanpy as sc
import scipy as scp

from sklearn.model_selection import train_test_split

from latentvelo.utils import normalize, sparse_mx_to_torch_sparse_tensor

class DatasetNoGCN(th.utils.data.Dataset):
    def __init__(self, adata, shuffle=True, test=0.1, random_seed=42):
        
        if shuffle:
            
            np.random.seed(random_seed)
            selected = np.random.choice(adata.shape[0], size = adata.shape[0], replace=False)
            adata = adata[selected]
        
        velo_genes_mask = np.zeros(adata.layers['spliced'].shape)
        velo_genes_mask[:,adata.var['velocity_genes'].values] = 1
        adata.layers['velo_genes_mask'] = velo_genes_mask
        print(velo_genes_mask[0].sum().astype(int), 'velocity genes used')

        if 'exp_time' not in adata.obs.columns.values:
            adata.obs['exp_time'] = 0
        
        adj = adata.obsp['adj']
            
        self.adata = adata
        self.IDs = np.arange(self.adata.shape[0], dtype = int)
        
        
    def __len__(self):
        return self.IDs.shape[0]
    
    def __getitem__(self, index):
        i = self.IDs[index]
        
        return (th.Tensor(self.adata[i].layers['spliced_counts'].astype(float)),
    th.Tensor(self.adata[i].layers['unspliced_counts'].astype(float)),
    th.Tensor(self.adata[i].layers['spliced'].astype(float)),
    th.Tensor(self.adata[i].layers['unspliced'].astype(float)),
                th.Tensor(self.adata[i].layers['mask_spliced'].astype(float)),
                th.Tensor(self.adata[i].layers['mask_unspliced'].astype(float)),
    th.Tensor(self.adata[i].obs['spliced_size_factor'].astype(float)),
                th.Tensor(self.adata[i].obs['unspliced_size_factor'].astype(float)),
              th.Tensor(self.adata[i].obs['root'].astype(int)).long(),
              th.Tensor(self.adata[i].layers['velo_genes_mask'].astype(float)),
                th.Tensor(self.adata[i].obsm['batch_onehot']).long(),
                th.Tensor(self.adata[i].obs['batch_id']).float(),
                th.Tensor(self.adata[i].obs['exp_time']).float(),
                th.Tensor(self.adata[i].obs['celltype_id']))
