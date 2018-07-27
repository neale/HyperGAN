def nets():
    networks = {}
    networks['wide'] = {
            'name': 'Wide',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2'],
            'shapes': [(1, 128, 3, 3), (128, 256, 3, 3), (1024, 1024), (1024, 10)],
            'base_shape': 3
    }
    networks['wide7'] = {
            'name': 'Wide7',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2', 'linear1', 'linear2'],
            'shapes': [(1, 128, 7, 7), (128, 256, 7, 7), (1024, 1024), (1024, 10)],
            'base_shape': 7
    }
    networks['net'] = {
            'name': '1x',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2'],
            'shapes': [(1, 128, 7, 7), (128, 256, 7, 7), (1024, 1024), (1024, 10)],
            'base_shape': 7
            }
    networks['fcn'] = {
            'name': 'FCN',
            'n_layers': 3,
            'layer_names': ['conv1', 'conv2', 'linear'],
            'shapes': [(1, 32, 3, 3), (32, 64, 3, 3), (1600, 10)],
            'base_shape': 3
            }
    networks['fcn2'] = {
            'name': 'FCN2',
            'n_layers': 2,
            'layer_names': ['conv1', 'linear'],
            'shapes': [(64, 1, 7, 7), (10, 1600)],
            'base_shape': 7
            }
    networks['small'] = {
            'name': 'Small',
            'n_layers': 2,
            'layer_names': ['conv1.0', 'linear'],
            'shapes': [(64, 1, 7, 7), (10, 3136)],
            'base_shape': 7
            }
    networks['tiny'] = {
            'name': 'Tiny',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2', 'linear1', 'linear2'],
            'shapes': [(8, 1, 3, 3), (16, 8, 3, 3), (128, 256), (10, 128)],
            'base_shape': 3
            }
    return networks
