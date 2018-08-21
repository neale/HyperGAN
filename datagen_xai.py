import os
import numpy as np
from glob import glob
import utils


def param_generator(args, data):

    filter_volume = data
    layer_size = data.shape[2]
    filter_shape = args.shapes[args.id]
    filters = filter_volume.reshape((-1, *filter_shape))
    # print ('filters a', filters.shape)
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    # filters = filters[len(filters)%layer_size:]
    #print ('layer size ', layer_size)
    #print ('filter shape', filter_shape)
    #print ('filters', filters.shape)
    def get_epoch():
        # np.random.shuffle(filters)
        params_batches = filters.reshape(-1, *filter_shape)
        # print (params_batches.shape)
        for i in range(len(params_batches)):
            yield (np.copy(params_batches[i]))
    return get_epoch


def load(args):
    generators = []
    for i in range(args.stat['n_layers']):
        args.id = id = i
        pdir = './params/{}/{}/{}/'.format(
                args.dataset, args.model, args.stat['layer_names'][id])
        pshape = args.shapes[id]
        paths = glob(pdir+'*.npy')
        data = np.zeros((len(paths), *pshape))
        for i in range(len(paths)):
            data[i] = np.load(paths[i])
        # len_t = int(len(data) * .9)
        train_data = data  # [:len_t]
        # val_data = data[len_t:]
        generators.append(param_generator(args, train_data))
    return generators

    # return (param_generator(args, train_data, id), 
    #         param_generator(args, val_data, id))
