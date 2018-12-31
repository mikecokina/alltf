import numpy as np

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
input_size = 1
batch_size = 10000
learning_rate = 1e-3
train_epochs = 1000
precession_tol = 1e-1

sin_start = -2.5 * np.pi
sin_end = 2.5 * np.pi

def serialize_nn_params():
    return dict(
        n_nodes_hl1=n_nodes_hl1,
        n_nodes_hl2=n_nodes_hl2,
        input_size=input_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_epochs=train_epochs
    )
