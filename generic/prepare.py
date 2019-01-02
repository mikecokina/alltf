import numpy as np


def split_training_set(x, y, test_size):
    train_samples = np.array(list(zip(x, y)))
    np.random.shuffle(train_samples)

    all_entities = len(x)

    if test_size > 0:
        test_samples = train_samples[:int(all_entities * test_size)]
        train_samples = train_samples[int(all_entities * test_size):]

        test_zipped = list(zip(*test_samples))
        test_X, test_y = test_zipped[0], test_zipped[1]
        test_samples = np.array(list(zip(test_X, test_y)))
    else:
        test_samples = [None, None]
    return train_samples, test_samples
