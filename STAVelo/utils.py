import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
import torch.nn.functional as F


def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)

    # u and s
    if isinstance(adata.layers['unspliced'], np.ndarray) and isinstance(adata.layers['spliced'], np.ndarray):
        X_concatenated = np.concatenate([adata.layers['unspliced'], adata.layers['spliced']], axis=1)
        X_u = adata.layers['unspliced']
        X_s = adata.layers['spliced']
    else:
        X_concatenated = np.concatenate([adata.layers['unspliced'].todense(), adata.layers['spliced'].todense()], axis=1)
        X_u = adata.layers['unspliced'].todense()
        X_s = adata.layers['spliced'].todense()

    # observe ds/dt
    d_s = []
    for i in range(adata.shape[0]):
        barcode_tmp = adata.uns['Spatial_Net'][adata.uns['Spatial_Net']['Cell1'] == adata.obs.index[i]]['Cell2']
        weights_tmp = 1 / adata.uns['Spatial_Net'][adata.uns['Spatial_Net']['Cell1'] == adata.obs.index[i]]['Distance']
        weights_tmp = weights_tmp / weights_tmp.sum()
        d_tmp = adata[barcode_tmp].layers['spliced'].todense() - adata[adata.obs.index[i]].layers[
            'spliced'].todense()
        d_tmp = np.average(d_tmp, axis=0, weights=weights_tmp)
        d_s.append(d_tmp)
    d_s = np.concatenate(d_s, axis=0)

    # observe du/dt
    d_u = []
    for i in range(adata.shape[0]):
        barcode_tmp = adata.uns['Spatial_Net'][adata.uns['Spatial_Net']['Cell1'] == adata.obs.index[i]]['Cell2']
        weights_tmp = 1 / adata.uns['Spatial_Net'][adata.uns['Spatial_Net']['Cell1'] == adata.obs.index[i]]['Distance']
        weights_tmp = weights_tmp / weights_tmp.sum()
        d_tmp = adata[barcode_tmp].layers['unspliced'].todense() - adata[adata.obs.index[i]].layers[
            'unspliced'].todense()
        d_tmp = np.average(d_tmp, axis=0, weights=weights_tmp)
        d_u.append(d_tmp)
    d_u = np.concatenate(d_u, axis=0)

    # Create PyTorch Geometric Data object
    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                x=torch.FloatTensor(X_concatenated),
                u=torch.FloatTensor(X_u),
                s=torch.FloatTensor(X_s),
                du=torch.FloatTensor(d_u),
                ds=torch.FloatTensor(d_s))
    return data

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)

def cosine_loss(velo_obs, velo_pred):
    cosine_sim = F.cosine_similarity(velo_obs, velo_pred, dim=1)
    losses = 1 - cosine_sim
    mean_loss = losses.mean()
    return mean_loss
