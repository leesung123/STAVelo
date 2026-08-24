# Calling functions in scVelo
# https://github.com/theislab/scvelo/releases#release-v0.2.5
# Generalizing RNA velocity to transient cell states through dynamical modeling, Nature Biotechnology, 2020

import scvelo as scv


settings = scv.settings
set_figure_params = scv.set_figure_params

def show_proportions(adata, **kwargs):
    return scv.utils.show_proportions(adata, **kwargs)

def filter_and_normalize(adata, **kwargs):
    return scv.pp.filter_and_normalize(adata, **kwargs)

def moments(adata, **kwargs):
    return scv.pp.moments(adata, **kwargs)

def velocity_graph(adata, **kwargs):
    return scv.tl.velocity_graph(adata, **kwargs)

def velocity_embedding(adata, **kwargs):
    return scv.pl.velocity_embedding(adata, **kwargs)

def velocity_embedding_stream(adata, **kwargs):
    return scv.pl.velocity_embedding_stream(adata, **kwargs)

def velocity_embedding_grid(adata, **kwargs):
    return scv.pl.velocity_embedding_grid(adata, **kwargs)

def terminal_states(adata, **kwargs):
    return scv.tl.terminal_states(adata, **kwargs)

def transition_matrix(adata, **kwargs):
    return scv.tl.transition_matrix(adata, **kwargs)

def paga(adata, **kwargs):
    return scv.tl.paga(adata, **kwargs)

def scatter(adata, **kwargs):
    return scv.pl.scatter(adata, **kwargs)

def velocity(adata, **kwargs):
    return scv.pl.velocity(adata, **kwargs)


__all__ = [
    "settings",
    "set_figure_params",
    "show_proportions",
    "filter_and_normalize",
    "moments",
    "velocity_graph",
    "velocity_embedding",
    "velocity_embedding_stream",
    "velocity_embedding_grid",
    "terminal_states",
    "transition_matrix",
    "paga",
    "scatter",
    "velocity"
]