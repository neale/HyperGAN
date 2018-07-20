def nets():
    networks = {}
    networks['wide'] = {
            'n_layers': 4,
            'conv': [(1, 128, 3, 3), (128, 256, 3, 3)],
            'linear': [(1024, 1024), (1024, 10)],
    }
    networks['wide7'] = {
            'n_layers': 4,
            'conv': [(1, 128, 7, 7), (128, 256, 7, 7)],
            'linear': [(1024, 1024), (1024, 10)],
    }
    networks['net'] = {
            'n_layers': 4,
            'conv': [(1, 128, 7, 7), (128, 256, 7, 7)],
            'linear': [(1024, 1024), (1024, 10)],
            }
    networks['fcn'] = {
            'n_layers': 3,
            'conv': [(1, 32, 3, 3), (32, 64, 3, 3)],
            'linear': [(1600, 10)],
            }
    networks['fcn2'] = {
            'n_layers': 2,
            'conv': [(64, 1, 7, 7)],
            'linear': [(10, 1600)],
            }
    networks['small'] = {
            'n_layers': 2,
            'layers': [(64, 1, 7, 7), (10, 3136)],
            'base_shape': 49
            }
    networks['tiny'] = {
            'n_layers': 4,
            'conv': [(8, 1, 3, 3), (16, 8, 3, 3)],
            'linear': [(128, 256), (10, 128)],
            'base_shape': 9
            }
    return networks
