"""
Assorted utility functions used throughout
"""
import scanpy as sc
import scvelo as scv
import numpy as np
import scipy as scp
import torch as th
import anndata as ad
import gc
from .velocity_genes import compute_velocity_genes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def unique_index(x):
    """
    find the index of the unique times for the ODE solver
    """
    sort_index = th.argsort(x)
    
    sorted_x = x[sort_index]
    index = th.Tensor([th.max(th.where(sorted_x==i)[0]) for i in th.unique(sorted_x)]).long()
    return sort_index, index

def average_velocity(adata_raw, vkey='velocity', n_pcs=30, n_neighbors=100):
    """
    Average velocity over nearest neighbors
    """
    adata = adata_raw.copy()

    scv.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    
    
    adata.layers['avg_velo'] = adata.layers[vkey].copy()
    for i in range(adata.shape[0]):
        neighbors = adata.uns['neighbors']['indices'][i]
        avg_velo = adata.layers[vkey][neighbors].mean(0)
        adata.layers['avg_velo'][neighbors] = np.stack(n_neighbors*[avg_velo])
        
    return adata.layers['avg_velo']

     
def paired_cosine_similarity(x, y):
    return th.nn.CosineSimilarity()(th.Tensor(x), th.Tensor(y)).detach().numpy()

def paired_correlation(x, y, mask = None, dim=0):
    """
    Adapated from DeepVelo (Cao et al 2022, bioRxiv)
    """
    if mask is None:
        x = x - th.mean(x, dim=0)
        y = y - th.mean(y, dim=0)
        x = x / (th.std(x, dim=0) + 1e-6)
        y = y / (th.std(y, dim=0) + 1e-6)
        return th.mean(x * y, dim=0) 
    else:
        
        with th.no_grad():
            mask = mask.detach().float()
            num_valid_data = th.sum(mask, dim=0)  # (D,)
        
        masked_y = y * mask
        masked_x = x * mask
        delta_x = mask * (masked_x - th.sum(masked_x, dim=0) / (num_valid_data + 1e-8)) # make the invalid data to zero again to ignore
        delta_y = mask * (masked_y - th.sum(masked_y, dim=0) / (num_valid_data + 1e-8))
        
        norm_x = delta_x / (th.sqrt(th.sum(th.pow(delta_x, 2), dim=0) + 1e-8) + 1e-8)
        norm_y = delta_y / (th.sqrt(th.sum(th.pow(delta_y, 2), dim=0) + 1e-8) + 1e-8)
        return th.sum(norm_x * norm_y, dim=0)  # (D,)

def paired_correlation_numpy(x, y, axis=1):

    if axis == 1:
        vx = x - np.mean(x, axis=axis)[:,None]
        vy = y - np.mean(y, axis=axis)[:,None]
    else:
        vx = x - np.mean(x, axis=axis)[None]
        vy = y - np.mean(y, axis=axis)[None]

    corr = np.sum(vx * vy, axis=axis) / (np.sqrt(np.sum(vx ** 2, axis=axis)) * np.sqrt(np.sum(vy ** 2, axis=axis)))
    return corr

    
def standard_clean_recipe(adata, spliced_key = 'spliced', unspliced_key = 'unspliced', batch_key = None, root_cells = None,
                          normalize_library=True, n_top_genes = 2000, smooth = True, umap=False, log=True, r2_adjust=True, share_normalization=False, center=False, celltype_key=None):

    """
    Clean and setup data for LatentVelo
    """
    if normalize_library:
        spliced_library_sizes = adata.layers[spliced_key].sum(1)
        unspliced_library_sizes = adata.layers[unspliced_key].sum(1)
        
        if len(spliced_library_sizes.shape) == 1:
            spliced_library_sizes = spliced_library_sizes[:,None]
        if len(unspliced_library_sizes.shape) == 1:
            unspliced_library_sizes = unspliced_library_sizes[:,None]
        
        if share_normalization:
            library_size = spliced_library_sizes + unspliced_library_sizes

        if share_normalization:
            spliced_median_library_sizes = np.median(np.array(library_size)[:,0])
            unspliced_median_library_sizes = np.median(np.array(library_size)[:,0])
        else:
            spliced_median_library_sizes = np.median(np.array(spliced_library_sizes)[:,0])
            unspliced_median_library_sizes = np.median(np.array(unspliced_library_sizes)[:,0])
        
        spliced_all_size_factors = spliced_library_sizes/spliced_median_library_sizes
        unspliced_all_size_factors = unspliced_library_sizes/unspliced_median_library_sizes
        
        adata.layers[spliced_key] = adata.layers[spliced_key]/spliced_all_size_factors
        adata.layers[unspliced_key] = adata.layers[unspliced_key]/unspliced_all_size_factors
        
        adata.obs['spliced_size_factor'] = spliced_library_sizes #spliced_all_size_factors
        adata.obs['unspliced_size_factor'] = unspliced_library_sizes #unspliced_all_size_factors
    
    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())
    
    if n_top_genes != None:
        scv.pp.filter_genes_dispersion(adata, n_top_genes = n_top_genes)

    gc.collect()
    
    if scp.sparse.issparse(adata.layers[spliced_key]):
        adata.layers[spliced_key] = adata.layers[spliced_key].todense()
        adata.layers[unspliced_key] = adata.layers[unspliced_key].todense()
        
    else:
        adata.layers[spliced_key] = scp.sparse.csr_matrix(adata.layers[spliced_key]).todense()
        adata.layers[unspliced_key] = scp.sparse.csr_matrix(adata.layers[unspliced_key]).todense()   

    # include raw counts
    adata.layers['spliced_counts'] = np.array(adata.layers[spliced_key])
    adata.layers['unspliced_counts'] = np.array(adata.layers[unspliced_key])
    
    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())

    adata.layers['mask_spliced'] = np.array((adata.layers[spliced_key] > 0) + (adata.layers[unspliced_key] > 0))*1 #
    adata.layers['mask_unspliced'] = np.array((adata.layers[unspliced_key] > 0) + (adata.layers[spliced_key] > 0))*1 # + 
    
    if log:
        scv.pp.log1p(adata)
    
    sc.pp.pca(adata)
    
    adata.layers['spliced'] = adata.layers[spliced_key]
    adata.layers['unspliced'] = adata.layers[unspliced_key]
    
    scv.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    adata.obsp['adj'] = adata.obsp['connectivities']
    
    compute_velocity_genes(adata, n_top_genes=n_top_genes,r2_adjust=r2_adjust)
    
    if umap:
        sc.tl.umap(adata)
    
    if smooth:
        adata.uns['scale_spliced'] = 4*(1+np.std(adata.layers['Ms'], axis=0)[None])
        adata.uns['scale_unspliced'] = 4*(1+np.std(adata.layers['Mu'], axis=0)[None])

        adata.layers['spliced_raw'] = np.array(adata.layers['spliced'])
        adata.layers['unspliced_raw'] = np.array(adata.layers['unspliced'])

        if center:
            
            adata.uns['mean_spliced'] = np.mean(adata.layers['Ms'], axis=0)[None]
            adata.uns['mean_unspliced'] = np.mean(adata.layers['Mu'], axis=0)[None]
            
            adata.layers['spliced'] = np.array((adata.layers['Ms'] - adata.uns['mean_spliced'])/adata.uns['scale_spliced'])
            adata.layers[spliced_key] = np.array((adata.layers['Ms'] - adata.uns['mean_spliced'])/adata.uns['scale_spliced'])
            adata.layers['unspliced'] = np.array((adata.layers['Mu'] - adata.uns['mean_unspliced'])/adata.uns['scale_unspliced'])
            adata.layers[unspliced_key] = np.array((adata.layers['Mu'] - adata.uns['mean_unspliced'])/adata.uns['scale_unspliced'])
        else:
            adata.layers['spliced'] = np.array(adata.layers['Ms']/adata.uns['scale_spliced'])
            adata.layers[spliced_key] = np.array(adata.layers['Ms']/adata.uns['scale_spliced'])
            adata.layers['unspliced'] = np.array(adata.layers['Mu']/adata.uns['scale_unspliced'])
            adata.layers[unspliced_key] = np.array(adata.layers['Mu']/adata.uns['scale_unspliced'])
        
    else:
        adata.uns['scale_spliced'] = 4*(1+np.std(adata.layers[spliced_key], axis=0)[None])
        adata.uns['scale_unspliced'] = 4*(1+np.std(adata.layers[unspliced_key], axis=0)[None])
        
        adata.layers['spliced'] = adata.layers[spliced_key]/adata.uns['scale_spliced']
        adata.layers[spliced_key] = adata.layers[spliced_key]/adata.uns['scale_spliced']
        adata.layers['unspliced'] = adata.layers[unspliced_key]/adata.uns['scale_unspliced']
        adata.layers[unspliced_key] = adata.layers[unspliced_key]/adata.uns['scale_unspliced']
    
    
    # use label encoder
    if batch_key != None:
        label_encoder = LabelEncoder()
        batch_id = label_encoder.fit_transform(adata.obs[batch_key])
        adata.obs['batch_id'] = batch_id

        onehotbatch = OneHotEncoder(sparse=False).fit_transform(batch_id[:,None])
        adata.obsm['batch_onehot'] = onehotbatch
        
    else:
        batch_key = 'batch_id'
        adata.obs[batch_key] = 0
        label_encoder = LabelEncoder()
        batch_id = label_encoder.fit_transform(adata.obs[batch_key])
        adata.obs['batch_id'] = batch_id

        onehotbatch = OneHotEncoder(sparse=False).fit_transform(batch_id[:,None])
        adata.obsm['batch_onehot'] = onehotbatch
    if celltype_key != None:
        label_encoder = LabelEncoder()
        celltype = label_encoder.fit_transform(adata.obs[celltype_key])
        adata.obs['celltype_id'] = celltype
    else:
        adata.obs['celltype_id'] = 0

    if root_cells == 'precalced':
        print('using precalced root cells')
    elif celltype_key != None and root_cells != None:
        adata.obs['root'] = 0
        adata.obs['root'][adata.obs[celltype_key] == root_cells] = 1
    else:
        adata.obs['root'] = 0    


def anvi_clean_recipe(adata, spliced_key = 'spliced', unspliced_key = 'unspliced', batch_key = None, root_cells=None,
                          normalize_library=True, n_top_genes = 2000, smooth = True, umap=False, log=True, celltype_key='celltype', r2_adjust=True, share_normalization=False):

    """
    Clean and setup data for celltype annotated version of LatentVelo
    """
    if normalize_library:
        spliced_library_sizes = adata.layers[spliced_key].sum(1)
        unspliced_library_sizes = adata.layers[unspliced_key].sum(1)
        
        if len(spliced_library_sizes.shape) == 1:
            spliced_library_sizes = spliced_library_sizes[:,None]
        if len(unspliced_library_sizes.shape) == 1:
            unspliced_library_sizes = unspliced_library_sizes[:,None]
        
        if share_normalization:
            library_size = spliced_library_sizes + unspliced_library_sizes

        if share_normalization:
            spliced_median_library_sizes = np.median(np.array(library_size)[:,0])
            unspliced_median_library_sizes = np.median(np.array(library_size)[:,0])
        else:
            spliced_median_library_sizes = np.median(np.array(spliced_library_sizes)[:,0])
            unspliced_median_library_sizes = np.median(np.array(unspliced_library_sizes)[:,0])
        
        spliced_all_size_factors = spliced_library_sizes/spliced_median_library_sizes
        unspliced_all_size_factors = unspliced_library_sizes/unspliced_median_library_sizes
        
        adata.layers[spliced_key] = adata.layers[spliced_key]/spliced_all_size_factors
        adata.layers[unspliced_key] = adata.layers[unspliced_key]/unspliced_all_size_factors
        
        adata.obs['spliced_size_factor'] = spliced_library_sizes #spliced_all_size_factors
        adata.obs['unspliced_size_factor'] = unspliced_library_sizes #unspliced_all_size_factors

    
    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())
    
    if n_top_genes != None:
        scv.pp.filter_genes_dispersion(adata, n_top_genes = n_top_genes)
    
    if scp.sparse.issparse(adata.layers[spliced_key]):
        adata.layers[spliced_key] = adata.layers[spliced_key].todense()
        adata.layers[unspliced_key] = adata.layers[unspliced_key].todense()
        
    else:
        adata.layers[spliced_key] = scp.sparse.csr_matrix(adata.layers[spliced_key]).todense()
        adata.layers[unspliced_key] = scp.sparse.csr_matrix(adata.layers[unspliced_key]).todense()
    
    # include raw counts
    adata.layers['spliced_counts'] = np.array(adata.layers[spliced_key])
    adata.layers['unspliced_counts'] = np.array(adata.layers[unspliced_key])

    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())
    
    adata.layers['mask_spliced'] = np.array((adata.layers[spliced_key] > 0) + (adata.layers[unspliced_key] > 0))*1 #
    adata.layers['mask_unspliced'] = np.array((adata.layers[unspliced_key] > 0) + (adata.layers[spliced_key] > 0))*1 # + 
    
    if log:
        scv.pp.log1p(adata)
    
    sc.pp.pca(adata)
    
    adata.layers['spliced'] = adata.layers[spliced_key]
    adata.layers['unspliced'] = adata.layers[unspliced_key]
    
    scv.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    adata.obsp['adj'] = adata.obsp['connectivities']
    
    compute_velocity_genes(adata, n_top_genes=n_top_genes,r2_adjust=r2_adjust)
    
    if umap:
        sc.tl.umap(adata)
    
    if smooth:
        adata.uns['scale_spliced'] = 4*(1+np.std(adata.layers['Ms'], axis=0)[None])
        adata.uns['scale_unspliced'] = 4*(1+np.std(adata.layers['Mu'], axis=0)[None])

        adata.layers['spliced_raw'] = np.array(adata.layers['spliced'])
        adata.layers['unspliced_raw'] = np.array(adata.layers['unspliced'])
        
        adata.layers['spliced'] = np.array(adata.layers['Ms']/adata.uns['scale_spliced'])
        adata.layers[spliced_key] = np.array(adata.layers['Ms']/adata.uns['scale_spliced'])
        adata.layers['unspliced'] = np.array(adata.layers['Mu']/adata.uns['scale_unspliced'])
        adata.layers[unspliced_key] = np.array(adata.layers['Mu']/adata.uns['scale_unspliced'])
        
    else:
        adata.uns['scale_spliced'] = 4*(1+np.std(adata.layers[spliced_key], axis=0)[None])
        adata.uns['scale_unspliced'] = 4*(1+np.std(adata.layers[unspliced_key], axis=0)[None])
        
        adata.layers['spliced'] = adata.layers[spliced_key]/adata.uns['scale_spliced']
        adata.layers[spliced_key] = adata.layers[spliced_key]/adata.uns['scale_spliced']
        adata.layers['unspliced'] = adata.layers[unspliced_key]/adata.uns['scale_unspliced']
        adata.layers[unspliced_key] = adata.layers[unspliced_key]/adata.uns['scale_unspliced']
    
    # use label encoder
    if batch_key != None:
        label_encoder = LabelEncoder()
        batch_id = label_encoder.fit_transform(adata.obs[batch_key])
        adata.obs['batch_id'] = batch_id

        onehotbatch = OneHotEncoder(sparse=False).fit_transform(batch_id[:,None])
        adata.obsm['batch_onehot'] = onehotbatch
        
    else:
        batch_key = 'batch_id'
        adata.obs[batch_key] = 0
        label_encoder = LabelEncoder()
        batch_id = label_encoder.fit_transform(adata.obs[batch_key])
        adata.obs['batch_id'] = batch_id

        onehotbatch = OneHotEncoder(sparse=False).fit_transform(batch_id[:,None])
        adata.obsm['batch_onehot'] = onehotbatch
    
    label_encoder = LabelEncoder()
    celltype = label_encoder.fit_transform(adata.obs[celltype_key])
    adata.obs['celltype'] = celltype
    
    onehotcelltype = OneHotEncoder(sparse=False).fit_transform(celltype[:,None])
    adata.obsm['celltype'] = onehotcelltype
    

    if celltype_key != None:
        label_encoder = LabelEncoder()
        celltype = label_encoder.fit_transform(adata.obs[celltype_key])
        adata.obs['celltype_id'] = celltype
    else:
        adata.obs['celltype_id'] = 0

    
    if root_cells == 'precalced':
        print('using precalced root cells')
    elif celltype_key != None and root_cells != None:
        adata.obs['root'] = 0
        adata.obs['root'][adata.obs[celltype_key] == root_cells] = 1
    else:
        adata.obs['root'] = 0  
    

def atac_clean_recipe(adata, spliced_key = 'spliced', unspliced_key = 'unspliced', batch_key = None,
                          normalize_library=True, n_top_genes = 2000, smooth = True, umap=False, log=False, n_neighbors=30, connectivities = None):

    """
    Clean and setup data for ATAC+RNA version of LatentVelo
    """
    
    if scp.sparse.issparse(adata.layers[spliced_key]):
        adata.layers[spliced_key] = adata.layers[spliced_key].todense()
        adata.layers[unspliced_key] = adata.layers[unspliced_key].todense()
        
    else:
        adata.layers[spliced_key] = scp.sparse.csr_matrix(adata.layers[spliced_key]).todense()
        adata.layers[unspliced_key] = scp.sparse.csr_matrix(adata.layers[unspliced_key]).todense()   

    # include raw counts
    adata.layers['spliced_counts'] = adata.layers[spliced_key]
    adata.layers['unspliced_counts'] = adata.layers[unspliced_key]
    adata.layers['atac'] = adata.layers['Mc'].todense()
    
    if normalize_library:
        spliced_library_sizes = adata.layers[spliced_key].sum(1)
        unspliced_library_sizes = adata.layers[unspliced_key].sum(1)
        
        spliced_median_library_sizes = np.median(np.array(spliced_library_sizes)[:,0])
        unspliced_median_library_sizes = np.median(np.array(unspliced_library_sizes)[:,0])
        
        spliced_all_size_factors = spliced_library_sizes/spliced_median_library_sizes
        unspliced_all_size_factors = unspliced_library_sizes/unspliced_median_library_sizes
        
        adata.layers[spliced_key] = adata.layers[spliced_key]/spliced_all_size_factors
        adata.layers[unspliced_key] = adata.layers[unspliced_key]/unspliced_all_size_factors
        
        adata.obs['spliced_size_factor'] = spliced_all_size_factors
        adata.obs['unspliced_size_factor'] = unspliced_all_size_factors

    
    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())
    
    if n_top_genes != None:
        scv.pp.filter_genes_dispersion(adata, n_top_genes = n_top_genes)

    adata.layers['mask_spliced'] = ((adata.layers[spliced_key] > 0) )*1
    adata.layers['mask_unspliced'] = ((adata.layers[unspliced_key] > 0))*1
    
    if log:
        scv.pp.log1p(adata)
    
    sc.pp.pca(adata)

    adata.obsp['adj'] = adata.obsp['connectivities']
    
    adata.layers['spliced'] = adata.layers[spliced_key]
    adata.layers['unspliced'] = adata.layers[unspliced_key]
    
    scv.pp.neighbors(adata, n_pcs=30, n_neighbors=n_neighbors)
    if connectivities != None:
        adata.obsp['connectivities'] = connectivities
    
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    compute_velocity_genes(adata, n_top_genes=n_top_genes)
    
    if umap:
        sc.tl.umap(adata)
    
    if smooth:
        adata.uns['scale_spliced'] = 4*(1+np.std(adata.layers['Ms'], axis=0)[None])
        adata.uns['scale_unspliced'] = 4*(1+np.std(adata.layers['Mu'], axis=0)[None])

        adata.layers['spliced_raw'] = adata.layers['spliced']
        adata.layers['unspliced_raw'] = adata.layers['unspliced']
        
        adata.layers['spliced'] = adata.layers['Ms']/adata.uns['scale_spliced']
        adata.layers[spliced_key] = adata.layers['Ms']/adata.uns['scale_spliced']
        adata.layers['unspliced'] = adata.layers['Mu']/adata.uns['scale_unspliced']
        adata.layers[unspliced_key] = adata.layers['Mu']/adata.uns['scale_unspliced']
        
    else:
        adata.uns['scale_spliced'] = 4*(1+np.std(adata.layers[spliced_key], axis=0)[None])
        adata.uns['scale_unspliced'] = 4*(1+np.std(adata.layers[unspliced_key], axis=0)[None])
        
        adata.layers['spliced'] = adata.layers[spliced_key]/adata.uns['scale_spliced']
        adata.layers[spliced_key] = adata.layers[spliced_key]/adata.uns['scale_spliced']
        adata.layers['unspliced'] = adata.layers[unspliced_key]/adata.uns['scale_unspliced']
        adata.layers[unspliced_key] = adata.layers[unspliced_key]/adata.uns['scale_unspliced']
    
    # use label encoder
    if batch_key != None:
        label_encoder = LabelEncoder()
        batch_id = label_encoder.fit_transform(adata.obs[batch_key])
        adata.obs['batch_id'] = batch_id

        onehotbatch = OneHotEncoder(sparse=False).fit_transform(batch_id[:,None])
        adata.obsm['batch_onehot'] = onehotbatch
        
    else:
        batch_key = 'batch_id'
        adata.obs[batch_key] = 0
        label_encoder = LabelEncoder()
        batch_id = label_encoder.fit_transform(adata.obs[batch_key])
        adata.obs['batch_id'] = batch_id

        onehotbatch = OneHotEncoder(sparse=False).fit_transform(batch_id[:,None])
        adata.obsm['batch_onehot'] = onehotbatch


# efficiently compute a batch of jacobians
def batch_jacobian(func, x, create_graph=True):
    f_sum = lambda x: th.sum(func(x), axis=0)
    return th.autograd.functional.jacobian(f_sum, x, create_graph=create_graph)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scp.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def gaussian_kl(mu, logvar, mu0 = 0, logvar0 = 0):
    """KL divergence between diagonal gaussians."""
    return -0.5 * th.sum(1. + logvar - logvar0 - (mu-mu0)**2/np.exp(0.5*logvar0) - th.exp(logvar)/np.exp(logvar0), dim=-1)


def batch_func(func, inputs, num_outputs, split_size = 500):
    """
    compute functions in batches/chunks to save memory for large datasets
    """
    outputs = [[] for j in range(num_outputs)]
    
    for i in range(split_size, inputs[0].shape[0] + split_size, split_size):
        
        inputs_i = []
        for input in inputs:
            if input==None or input == (None, None) or input == (None, None, None) or input == (None, None, None, None) or type(input) == int or type(input) == float or (type(input) != tuple and len(input.shape)) == 1:
                inputs_i.append(input)
            elif type(input) == tuple:
                #print(input)
                if len(input) == 2:
                    inputs_i.append((input[0][i-split_size:i], input[1][i-split_size:i]))
                elif len(input) == 3:
                    if input[:2] == (None, None):
                        inputs_i.append(input)
                    elif input[0] != None and input[1] != None and input[2] == None:
                        inputs_i.append((input[0][i-split_size:i], input[1][i-split_size:i], None))
                    elif input[0] != None and input[1] == None and input[2] != None:
                        inputs_i.append((input[0][i-split_size:i], None, input[2][i-split_size:i]))
                    else:
                        inputs_i.append((input[0][i-split_size:i], input[1][i-split_size:i], input[2][i-split_size:i]))
                elif len(input) == 4:
                    #print(input)
                    if input[:2] == (None, None):
                        inputs_i.append(input)
                        #elif input[0] != None and input[1] != None and input[2] == None:
                        #    #print(input)
                        #    inputs_i.append((input[0][i-split_size:i], input[1][i-split_size:i], None, None))
                    elif input[0] != None and input[1] != None and input[2] == None and input[3] == None:
                        inputs_i.append((input[0][i-split_size:i], input[1][i-split_size:i], None, None))
                      
                    elif input[0] != None and input[1] != None and input[2] != None and input[3] == None:
                        inputs_i.append((input[0][i-split_size:i], input[1][i-split_size:i], input[2][i-split_size:i], None))
                        
                    elif input[0] != None and input[1] != None and input[2] == None and input[3] != None:
                        inputs_i.append((input[0][i-split_size:i], input[1][i-split_size:i], None, input[3][i-split_size:i]))
                    else:
                        inputs_i.append((input[0][i-split_size:i], input[1][i-split_size:i], input[2][i-split_size:i], input[3][i-split_size:i]))
            elif input.shape[0] != input.shape[1]:
                inputs_i.append(input[i-split_size:i])
            else:
                inputs_i.append(sparse_mx_to_torch_sparse_tensor(normalize(input[i-split_size:i, i-split_size:i])).cuda())
            
        outputs_i = func(*inputs_i)
        if type(outputs_i) != tuple:
            outputs_i = tuple((outputs_i,))
            
        if len(outputs_i) != num_outputs:
            print('error, expected different number of outputs')
        
        for j in range(num_outputs):
            outputs[j].append(outputs_i[j].cpu())
    
    outputs_tensor = [None for j in range(num_outputs)]
    for j in range(num_outputs):
        outputs_tensor[j] = th.cat(outputs[j], dim=0)
    return tuple(outputs_tensor)
