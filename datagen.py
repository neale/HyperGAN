import os
import numpy as np
from glob import glob

def param_generator(args, data, id):

    filter_volume = data
    layer_size = data.shape[2]
    filter_shape = args.shapes[id]
    filters = filter_volume.reshape((-1, *filter_shape))
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    filters = filters[len(filters)%layer_size:]
    def get_epoch():
        np.random.shuffle(filters)
        params_batches = filters.reshape(-1, layer_size, *filter_shape)

        for i in range(len(params_batches)):
            yield (np.copy(params_batches[i]))
    return get_epoch


def load(args, id):
    pdir = './params/{}/{}/{}/'.format(
            args.dataset, args.model, args.stat['layer_names'][id])
    pshape = args.shapes[id]
    paths = glob(pdir+'*.npy')
    data = np.zeros((len(paths), *pshape))
    for i in range(len(paths)):
        data[i] = np.load(paths[i])
    len_t = int(len(data) * .9)
    train_data = data[:len_t]
    val_data = data[len_t:]
    
    return (param_generator(args, train_data, id), 
            param_generator(args, val_data, id))
