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

    if 'Spatial_Net' not in adata.uns:
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    spatial_net = adata.uns['Spatial_Net']
    required_cols = {'Cell1', 'Cell2', 'Distance'}
    if not required_cols.issubset(spatial_net.columns):
        raise ValueError("Spatial_Net must contain columns Cell1, Cell2, Distance.")

    dist_raw = np.asarray(spatial_net['Distance'], dtype=np.float64)
    invalid_dist = ~np.isfinite(dist_raw) | (dist_raw <= 0)
    if dist_raw.size == 0 or invalid_dist.any():
        raise ValueError(
            "Spatial_Net Distance must be finite and strictly greater than 0; "
            f"found {int(invalid_dist.sum())} invalid values."
        )

    n_dup = int(spatial_net.duplicated(subset=['Cell1', 'Cell2']).sum())
    if n_dup > 0:
        raise ValueError(
            f"Spatial_Net contains {n_dup} duplicate edges (Cell1, Cell2)."
        )

    n_obs = adata.n_obs
    cells_id_tran = dict(zip(np.asarray(adata.obs_names), range(n_obs)))

    G_df = spatial_net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    valid = G_df['Cell1'].notna() & G_df['Cell2'].notna()
    c1 = G_df.loc[valid, 'Cell1'].to_numpy(dtype=np.int64)
    c2 = G_df.loc[valid, 'Cell2'].to_numpy(dtype=np.int64)
    has_neighbor = np.zeros(n_obs, dtype=bool)
    if c1.size:
        has_neighbor[c1] = True
    if not has_neighbor.all():
        missing = np.asarray(adata.obs_names)[~has_neighbor]
        raise ValueError(
            "Every cell must have at least one neighbor in Spatial_Net; "
            f"{missing.size} cells have none (e.g. {missing[:5].tolist()})."
        )

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])),
                      shape=(n_obs, n_obs))
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)

    dist = G_df.loc[valid, 'Distance'].to_numpy(dtype=np.float64)
    W = sp.csr_matrix((1.0 / dist, (c1, c2)), shape=(n_obs, n_obs))
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    inv_row = np.zeros_like(row_sum)
    nz = row_sum > 0
    inv_row[nz] = 1.0 / row_sum[nz]
    W = sp.diags(inv_row) @ W

    def _layer_matrix(layer):
        if sp.issparse(layer):
            return layer.tocsr()
        return np.asarray(layer)

    def _to_dense(mat):
        if sp.issparse(mat):
            return mat.toarray()
        return np.asarray(mat)

    def _obs_deriv(X):
        dX = W @ X - X
        return _to_dense(dX)

    X_u = _layer_matrix(adata.layers['unspliced'])
    X_s = _layer_matrix(adata.layers['spliced'])
    d_u = _obs_deriv(X_u)
    d_s = _obs_deriv(X_s)
    X_u = _to_dense(X_u)
    X_s = _to_dense(X_s)
    X_concatenated = np.concatenate([X_u, X_s], axis=1)

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
