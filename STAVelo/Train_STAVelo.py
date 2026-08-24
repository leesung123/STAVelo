import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from .STAVelo import STAVelo
from .utils import cosine_loss, Transfer_pytorch_Data

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F


def train_STAVelo(adata, hidden_dims=[512, 30, 512], n_epochs=1000, lr=0.001,
                  gradient_clipping=5., weight_decay=0.0001, verbose=True,
                  random_seed=0, save_loss=False, save_rep=True, loss_type='cosine', w1 = 1, w2 = 1,
                  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package. Need to be preprocessed by scVelo.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed=random_seed

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    adata = adata[adata.obs.index.isin(adata.uns['Spatial_Net']['Cell1'])]
    adata.X = sp.csr_matrix(adata.X)

    if verbose:
        print('Size of Input: ', adata.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    if adata.uns['Spatial_Net']['Cell1'].unique().shape[0] < adata.shape[0]:
        raise ValueError("Please extend the scope of spatial neighborhood!")
    if 'spliced' not in adata.layers or 'unspliced' not in adata.layers:
        raise ValueError("Spliced and unspliced data are not existed!")

    data = Transfer_pytorch_Data(adata)

    gene_number = data.u.shape[1]
    model = STAVelo(hidden_dims = [gene_number] + hidden_dims).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #loss_list = []
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        z1, z2, z3, out = model(data.x, data.edge_index)
        alpha_tmp = out[:, 0:gene_number]
        beta_tmp = out[:, gene_number:gene_number * 2]
        gamma_tmp = out[:, gene_number * 2:gene_number * 3]
        ds_tmp = beta_tmp * data.u - gamma_tmp * data.s
        du_tmp = alpha_tmp - beta_tmp * data.u
        if loss_type == "cosine":
            loss = w1 * cosine_loss(data.ds, ds_tmp) + w2 * cosine_loss(data.du, du_tmp)
        if loss_type == "mse":
            loss = w1 * F.mse_loss(data.ds, ds_tmp) + w2 * F.mse_loss(data.du, du_tmp) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        #loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    model.eval()
    z1, z2, z3, out = model(data.x, data.edge_index)
    alpha_tmp = out[:, 0:gene_number]
    beta_tmp = out[:, gene_number:gene_number * 2]
    gamma_tmp = out[:, gene_number * 2:gene_number * 3]
    ds_tmp = beta_tmp * data.u - gamma_tmp * data.s

    adata.layers['velocity'] = ds_tmp.to('cpu').detach().numpy()

    adata.obsm['rate'] = out.to('cpu').detach().numpy()

    adata.obsm['alpha'] = alpha_tmp.to('cpu').detach().numpy()
    adata.obsm['beta'] = beta_tmp.to('cpu').detach().numpy()
    adata.obsm['gamma'] = gamma_tmp.to('cpu').detach().numpy()

    if save_rep:
        adata.obsm['rep_1'] = z1.to('cpu').detach().numpy()
        adata.obsm['rep_2'] = z2.to('cpu').detach().numpy()
        adata.obsm['rep_3'] = z3.to('cpu').detach().numpy()

    if save_loss:
        adata.uns['STAVelo_loss'] = loss

    return adata
