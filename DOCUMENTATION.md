# Documentation



## Data preparation settings

Data is prepared for LatentVelo:
```
import latentvelo as ltv
ltv.utils.standard_clean_recipe(adata, spliced_key='spliced', unspliced_key='unspliced',
                                batch_key='batch', celltype_key='celltype')
```

Or for the annotated model, data is prepared:
```
ltv.utils.anvi_clean_recipe(adata, spliced_key='spliced', unspliced_key='unspliced',
                                batch_key='batch', celltype_key='celltype')
```

Several other keyword arguments are accepted:

-`root_cells`: Name of the celltype of rootcells from the `cluster_key` argument. Default `None`.
-`n_top_genes`: Select the number of highly variable genes to use. Default `2000`.
`log`: log is applied to normalized counts before computing principle components for nearest neighbors. Default `True`.
`smooth`: Apply nearest neighbor smoothing to unspliced and spliced data. Default `True`.
`normalize_library`: Apply library size normalization. Default `True`.



## Model settings

The LatentVelo model is initialized:
```
import latentvelo as ltv
model = ltv.models.VAE(observed = number_of_genes, latent_dim = latent_dimension,
                       zr_dim = latent_regulation_dimension,
					   h_dim = conditioning_dimension)
```

Or for the annotated version:

```
model = ltv.models.AnnotVAE(observed = number_of_genes, latent_dim = latent_dimension,
                       zr_dim = latent_regulation_dimension,
					   h_dim = conditioning_dimension,
					   celltypes = number_of_celltypes)
```

Several other keyword arguments are accepted:

-`root_weight`: the weight in the loss function for specified root cells. Default `0`.
-`correlation_reg`: Use gene-space correlation regularization. Default `True`.
-`corr_weight_u`: weight in the loss function for unspliced correlation regularization, `correlation(vs, u)`. Default `0.1`.
-`corr_weight_s`: weight in the loss function for spliced correlation regularization `correlation(vs, -s)`. Default `0.1`.
-`celltype_corr`: celltype specific correlation regularization. Default `False`.
-`celltypes`: Number of celltypes in the celltype_key. Default `1`.
-`use_velo_genes`: Use only velocity genes in the likelihood instead of all genes. Default `False`.
-`corr_velo_mask`: Use only velocity genes in the correlation regulatization. Default `True`.
-`time_reg`: Include regularization by experimental time `exp_time` key in anndata. Default `False`.
-`time_reg_weight`: weight in the loss function for time regulation by the correlation between latent time and experimental time `correlation(latent_time, exp_time)`. Default `0.1`.
-`batch_correction`: perform batch correction by using the specified batch key as input to the encoder and decoder. Default `False`.
-`batches`: Number of batches. Default `1`.
-`gcn`: Use a graph convolutional network encoder that incorporates the nearest neighbor graph. Default `True`.
-`linear_decoder`: Use a linear decoder. Default `True`.
-`encoder_hidden`: the hidden layer size of the encoder. Default `25`.
-`decoder_hidden`: the hidden layer size of the decoder (if it is not linear). Default `25`.
-`exp_time`: Include experimental time in the encoder. Default `True`.
-`velo_reg`: Regularize on gene space by comparing decoded velocites to linear splicing velocity  instead of correlation, `|vs - (beta*u - gamma*s)|^2`. Default `False`.
-`velo_reg_weight`: Velocity regularization weight. Default `0.0001`.
-`latent_reg`: Regularize the derivative of the latent splicing dynamics. Default `False`.
-`kl_final_weight`: kl-divergence weight in the loss function. Default `1`.
-`kl_warmup_steps`: Number of steps linearly increasing the kl-divergence weight of the VAE. Default `25`.
-`num_steps`: Max number of steps in the ODE solver. Default `100`.
-`shared`: Share encoder/decoder for unspliced/spliced. Default `False`.
-`include_time`: Include time in the unspliced velocity function. Default `False`.