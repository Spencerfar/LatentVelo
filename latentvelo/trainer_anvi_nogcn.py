import torch as th
import torch.nn as nn
from latentvelo.anvi_dataloader_nogcn import ANVIDatasetNoGCN
from latentvelo.utils import normalize, sparse_mx_to_torch_sparse_tensor
import os
import numpy as np
import matplotlib.pyplot as plt
from latentvelo.utils import batch_func
from .collate import collate_anvi

def train_anvi_nogcn(model, adata, epochs = 50, learning_rate = 1e-2, batch_size = 200, grad_clip = 1, shuffle=True, test=0.1, name = '', optimizer='adam', random_seed=42):

    results_folder = './' + name + '/'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    else:
        print('Warning, folder already exists. This may overwrite a previous fit.')
    
    if optimizer == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr = learning_rate)
    elif optimizer == 'adamW':
        optimizer = th.optim.AdamW(model.parameters(), lr = learning_rate)
    
    scheduler_plateau = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.75, threshold = 0.05, threshold_mode ='rel', patience = 5, min_lr = 1e-5, verbose=True)

    if 'root' not in adata.obs:
        adata.obs['root'] = 0
    
    gcn = model.gcn
    batch_correction = model.batch_correction
    if not batch_correction:
        batch_id = None
        batch_onehot = None
    if not model.exp_time:
        exp_time = None

    if model.likelihood_model == 'gaussian':
        s_count_key = 'normedS'
        u_count_key = 'normedU'
    else:
        s_count_key = 'S'
        u_count_key = 'U'
        
    dataset = ANVIDatasetNoGCN(adata, shuffle=shuffle, test=test, random_seed=random_seed)
    loader = th.utils.data.DataLoader(dataset,
                                     batch_size = batch_size,
                                     shuffle = True, drop_last = False,
                                     num_workers = 0, pin_memory=True,
                                      collate_fn= lambda x: collate_anvi(x))
    
    model = model.cuda()
    model_state_history = [model.state_dict()]
    epoch_history = [0]
    val_ae_history = [np.inf]
    val_traj_history = [np.inf]
    val_traj_rel_history = [np.inf]
    
    for epoch in range(epochs):
        
        model = model.train()
        val_loss = 0.0
        val_num = 0.0
        for batch in loader:
            
            optimizer.zero_grad()
            
            s = batch[s_count_key].cuda()
            normed_s = batch['normedS'].cuda()
            mask_s = batch['maskS'].cuda()
            s_size_factors = batch['spliced_size_factor'].cuda()[:,None]
            
            u = batch[u_count_key].cuda()
            normed_u = batch['normedU'].cuda()
            mask_u = batch['maskU'].cuda()
            u_size_factors = batch['unspliced_size_factor'].cuda()[:,None]  
            
            velo_genes_mask = batch['velo_genes_mask'].cuda()
            
            root_cells = batch['root'].cuda()
            
            obs_celltype = batch['celltype'].cuda()

            celltype_id = batch['celltype_id'].cuda()
            
            if model.exp_time:
                exp_time = batch['exp_time'].cuda()[:,None]
                obs_celltype = (obs_celltype, exp_time, celltype_id)
            else:
                obs_celltype = (obs_celltype, None, celltype_id)
            
            index_train, index_test = th.arange(normed_s.shape[0]).cuda(), th.arange(normed_s.shape[0]).cuda()
            
            adj = None
            if batch_correction:
                batch_id = batch['batch_id'].cuda()[:,None]
                batch_onehot = batch['batch_onehot'].cuda()
            loss, validation_ae, validation_traj, validation_velo, orig_index = model.loss(normed_s, s, s_size_factors, mask_s, normed_u, u, u_size_factors, mask_u, velo_genes_mask, adj, root_cells, obs_celltype, batch_id=(batch_id, batch_onehot), epoch=epoch)
            
            curr_index = th.arange(loss.shape[0]).cuda()
            
            train_loss = th.mean(loss)
            
            vloss = loss.detach()
            
            val_loss += th.sum(vloss)
            val_num += vloss.shape[0]
            
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        model = model.eval()
        with th.no_grad():
            
            adata = dataset.adata
            if model.likelihood_model == 'gaussian':
                s = th.Tensor(adata.layers['spliced'].astype(float)).cuda()
                u = th.Tensor(adata.layers['unspliced'].astype(float)).cuda()
            else:
                s = th.Tensor(adata.layers['spliced_counts'].astype(float)).cuda()
                u = th.Tensor(adata.layers['unspliced_counts'].astype(float)).cuda()
            
            normed_s = th.Tensor(adata.layers['spliced'].astype(float)).cuda()
            normed_u = th.Tensor(adata.layers['unspliced'].astype(float)).cuda()
            
            s_size_factors = th.Tensor(adata.obs['spliced_size_factor'].astype(float)).cuda()[:,None]
            u_size_factors = th.Tensor(adata.obs['unspliced_size_factor'].astype(float)).cuda()[:,None]

            mask_s = th.Tensor(adata.layers['mask_spliced'].astype(float)).cuda()
            mask_u = th.Tensor(adata.layers['mask_unspliced'].astype(float)).cuda()

            velo_genes_mask = th.Tensor(adata.layers['velo_genes_mask'].astype(float)).cuda()
            
            root_cells = th.Tensor(adata.obs['root'].astype(float)).cuda()[:,None]

            obs_celltype = th.Tensor(adata.obsm['celltype']).cuda()
            celltype_id = th.Tensor(adata.obs['celltype_id']).cuda()
            
            if model.exp_time:
                exp_time = th.Tensor(adata.obs['exp_time']).cuda()[:,None]
                obs_celltype = (obs_celltype, exp_time, celltype_id)
            else:
                obs_celltype = (obs_celltype, None, celltype_id)
            
            adj = None
            if batch_correction:
                batch_id = th.Tensor(adata.obs['batch_id']).cuda()[:,None]
                batch_onehot = th.Tensor(adata.obsm['batch_onehot']).cuda()
            loss, validation_ae, validation_traj, validation_velo, _ = batch_func(model.loss, (normed_s, s, s_size_factors, mask_s, normed_u, u, u_size_factors, mask_u, velo_genes_mask, adj, root_cells, obs_celltype, (batch_id, batch_onehot)), 5, split_size = batch_size)

            loss = loss.mean().cpu().numpy()
            validation_ae = validation_ae.mean().cpu().numpy()
            validation_traj = validation_traj.mean().cpu().numpy()
            validation_velo = validation_velo.mean().cpu().numpy()
                
            print('epoch %d, full loss %.3f, val loss %.3f, ae MSE %.3f, traj MSE %.3f, reg loss %.3f'%(epoch,
                                                                                 loss,
                                                                                 (val_loss/val_num).cpu().numpy(),
                                                                                           validation_ae, validation_traj, validation_velo) )
        
        scheduler_plateau.step(validation_traj + validation_ae)
        epoch_history.append(epoch)
        val_ae_history.append(validation_ae)
        val_traj_history.append(validation_traj)
        model_state_history.append(model.state_dict())
        
        th.save(model.state_dict(), results_folder+'model_state_epoch%d.params'%(epoch))
        
        if epoch == epochs - 1:
            th.save(model.state_dict(), results_folder+'model_state_epoch%d.params'%(epoch))
        
        del model_state_history[0]
        
    # determine best model
    val_history = np.array(val_ae_history) + np.array(val_traj_history)
    best_index = np.argmin(val_history)

    print('Loading best model at %d epochs.'%epoch_history[best_index])
    model.load_state_dict(th.load(results_folder+'model_state_epoch%d.params'%epoch_history[best_index], map_location=th.device('cuda')))
    return np.array(epoch_history)[1:], np.array(val_ae_history)[1:], np.array(val_traj_history)[1:]

def plot_history(epochs, val_traj, val_ae):
    
    plt.plot(epochs, val_traj, label = 'Velocity field')
    plt.plot(epochs, val_ae, label = 'Autoencoder')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
