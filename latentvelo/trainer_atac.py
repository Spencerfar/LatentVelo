import torch as th
import torch.nn as nn
from latentvelo.atac_dataloader import ATACDataset
from latentvelo.utils import normalize, sparse_mx_to_torch_sparse_tensor
import os
import numpy as np
import matplotlib.pyplot as plt
#class Trainer:

def train_atac(model, adata, epochs = 50, learning_rate = 1e-2, batch_size = 200, grad_clip = 1, shuffle=True, test=0.1, name = '', optimizer='adam'):

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

    if model.likelihood_model == 'gaussian':
        s_count_key = 'normedS'
        u_count_key = 'normedU'
    else:
        s_count_key = 'S'
        u_count_key = 'U'
        
    dataset = ATACDataset(adata, batch_size, shuffle=shuffle, test=test)
    loader = th.utils.data.DataLoader(dataset,
                                     batch_size = 1,
                                     shuffle = True, drop_last = False,
                                     num_workers = 0, pin_memory=True,
                                      collate_fn=lambda x: x[0])

    adata.uns['index_test'] = dataset.adata.uns['index_test']
    
    model = model.cuda()
    model_state_history = [model.state_dict()]
    epoch_history = [0]
    val_ae_history = [np.inf]
    val_traj_history = [np.inf]
    
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
            
            normed_a = batch['normedA'].cuda()
            
            velo_genes_mask = batch['velo_genes_mask'].cuda()
            
            root_cells = batch['root'].cuda()
            
            index_train, index_test = batch['index_train'].cuda(), batch['index_test'].cuda()
            
            if gcn:
                adj = sparse_mx_to_torch_sparse_tensor(batch['adj']).cuda()
                if batch_correction:
                    batch_id = batch['batch_id'].cuda()[:,None]
                loss, validation_ae, validation_traj, validation_velo, orig_index = model.loss(normed_s, s, s_size_factors, mask_s, normed_u, u, u_size_factors, mask_u, normed_a, velo_genes_mask, adj, root_cells, batch_id=batch_id, epoch=epoch)
            else:
                batch_id = batch['batch_onehot'].cuda() #[:,None]
                loss, validation_ae, validation_traj, validation_velo, orig_index = model.loss(normed_s, s, s_size_factors, mask_s, normed_u, u, u_size_factors, mask_u,  normed_a, velo_genes_mask, root_cells, batch_id=batch_id, epoch=epoch)
            
            curr_index = th.arange(loss.shape[0]).cuda()
            index_train = th.stack([i for i in curr_index if orig_index[i] in index_train])
            index_test = th.stack([i for i in curr_index  if orig_index[i] in index_test])
            
            train_loss = th.mean(loss[index_train])
            
            vloss = loss.detach()[index_test]
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
            normed_a = th.Tensor(adata.layers['atac'].astype(float)).cuda()
            
            s_size_factors = th.Tensor(adata.obs['spliced_size_factor'].astype(float)).cuda()[:,None]
            u_size_factors = th.Tensor(adata.obs['unspliced_size_factor'].astype(float)).cuda()[:,None]

            mask_s = th.Tensor(adata.layers['mask_spliced'].astype(float)).cuda()
            mask_u = th.Tensor(adata.layers['mask_unspliced'].astype(float)).cuda()

            velo_genes_mask = th.Tensor(adata.layers['velo_genes_mask'].astype(float)).cuda()
            
            root_cells = th.Tensor(adata.obs['root'].astype(float)).cuda()[:,None]
            
            if gcn:
                adj = adata.obsp['adj']
                if batch_correction:
                    batch_id = th.Tensor(adata.obs['batch_id']).cuda()[:,None]
                loss, validation_ae, validation_traj, validation_velo = model.batch_func(model.loss, (normed_s, s, s_size_factors, mask_s, normed_u, u, u_size_factors, mask_u, normed_a, velo_genes_mask, adj, root_cells, batch_id), 5, split_size = batch_size)[:4]
            else:
                batch_id = th.Tensor(adata.obsm['batch_onehot']).cuda() #[:,None]
                loss, validation_ae, validation_traj, validation_velo = model.batch_func(model.loss, (normed_s, s, s_size_factors, mask_s, normed_u, u, u_size_factors, mask_u, normed_a, velo_genes_mask, root_cells, batch_id, epoch), 5, split_size = batch_size)[:4]

            loss = loss.mean().cpu().numpy()
            validation_ae = validation_ae.mean().cpu().numpy()
            validation_traj = validation_traj.mean().cpu().numpy()
            validation_velo = validation_velo.mean().cpu().numpy()
                
            print('epoch %d, full loss %.3f, val loss %.3f, ae dist %.3f, traj dist %.3f, velo %.3f'%(epoch,
                                                                                 loss,
                                                                                 (val_loss/val_num).cpu().numpy(),
                                                                                           validation_ae, validation_traj, validation_velo) )
        
        scheduler_plateau.step(validation_traj + validation_ae)
        epoch_history.append(epoch)
        val_ae_history.append(validation_ae)
        val_traj_history.append(validation_traj)
        model_state_history.append(model.state_dict())
        
        if val_ae_history[-1] > val_ae_history[-2] or val_traj_history[-1] > val_traj_history[-2]:
            th.save(model_state_history[-2], results_folder+'model_state_epoch%d.params'%(epoch-1))
        
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
    
