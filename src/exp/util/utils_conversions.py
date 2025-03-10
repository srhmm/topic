import numpy as np


def convert_contexts_to_stack(datasets):
    data_combined = None
    for ci, c in enumerate(datasets):
        data_combined = datasets[c] if data_combined is None \
            else np.vstack((data_combined, datasets[c]))
    return data_combined


def convert_contexts_to_labelled_stack(datasets):
    data_combined, c_indx = None, None
    for ci, c in enumerate(datasets):
        data_combined = datasets[c] if data_combined is None \
            else np.vstack((data_combined, datasets[c]))
        c_indx = ci * np.ones(datasets[c].shape[0]) if c_indx is None else np.vstack(
            (c_indx, ci * np.ones(datasets[c].shape[0])))
    return np.hstack((data_combined, c_indx.reshape(-1,1)))

def convert_hard_interventions_to_idls(datasets, intervention_targets):
    N = datasets[0].shape[1]
    assert all([datasets[k].shape[1]==N for k in range(len(datasets))])
    idls = [None for _ in range(N)]
    for ni in range(N):
        c_indx = None
        for ci, c in enumerate(datasets):
            nx = 0 if ni not in intervention_targets[ci] else 1
            c_indx = nx * np.ones(datasets[c].shape[0]) if c_indx is None else np.vstack(
                (c_indx, nx * np.ones(datasets[c].shape[0])))
        idls[ni] = c_indx.reshape(-1)
    return idls