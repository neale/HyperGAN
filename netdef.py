def nets():
    networks = {}
    networks['wide'] = {
            'name': 'wide',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2'],
            'shapes': [(1, 128, 3, 3), (128, 256, 3, 3), (1024, 1024), (1024, 10)],
            'base_shape': 3
    }
    networks['wide7'] = {
            'name': 'wide7',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2', 'linear1', 'linear2'],
            'shapes': [(1, 128, 7, 7), (128, 256, 7, 7), (1024, 1024), (1024, 10)],
            'base_shape': 7
    }
    networks['net'] = {
            'name': 'net',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2'],
            'shapes': [(1, 128, 7, 7), (128, 256, 7, 7), (1024, 1024), (1024, 10)],
            'base_shape': 7
            }
    networks['fcn'] = {
            'name': 'fcn',
            'n_layers': 3,
            'layer_names': ['conv1', 'conv2', 'linear'],
            'shapes': [(1, 32, 3, 3), (32, 64, 3, 3), (1600, 10)],
            'base_shape': 3
            }
    networks['fcn2'] = {
            'name': 'fcn2',
            'n_layers': 2,
            'layer_names': ['conv1', 'linear'],
            'shapes': [(64, 1, 7, 7), (10, 1600)],
            'base_shape': 7
            }
    networks['small'] = {
            'name': 'small',
            'n_layers': 2,
            'layer_names': ['conv1', 'linear'],
            'shapes': [(64, 1, 7, 7), (10, 3136)],
            'base_shape': 7
            }
    networks['tiny'] = {
            'name': 'tiny',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2', 'linear1', 'linear2'],
            'shapes': [(8, 1, 3, 3), (16, 8, 3, 3), (128, 256), (10, 128)],
            'base_shape': 3
            }
    return networks
