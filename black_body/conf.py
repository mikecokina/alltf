import numpy as np

n_nodes_hl1 = 1500
n_nodes_hl2 = 1000
input_size = 1
batch_size = 100000
learning_rate = 1e-4
train_epochs = 3000
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
