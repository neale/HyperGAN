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
    networks['small2'] = {
            'name': 'Small2',
            'n_layers': 3,
            'layer_names': ['conv1.0', 'conv2.0', 'linear'],
            'shapes': [(32, 1, 5, 5), (32, 32, 5, 5), (10, 512)],
            'base_shape': 5
            }
    networks['tiny'] = {
            'name': 'Tiny',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2', 'linear1', 'linear2'],
            'shapes': [(8, 1, 3, 3), (16, 8, 3, 3), (128, 256), (10, 128)],
            'base_shape': 3
            }
    networks['cnet'] = {
            'name': 'CNet',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2', 'fc1', 'fc2'],
            'shapes': [(64, 3, 3, 3), (128, 64, 3, 3), (128, 8192), (10, 128)],
            'base_shape': 3
            }
    networks['ctiny'] = {
            'name': 'CTiny',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2', 'conv3', 'linear2'],
            'shapes': [(16, 3, 3, 3), (32, 16, 3, 3), (32, 32, 3, 3), (10, 128)],
            'base_shape': 3
            }
    networks['lenet'] = { 
            'name': 'LeNet',
            'n_layers': 5, 
            'layer_names': ['conv1', 'conv2', 'linear1', 'linear2', 'linear3'],
            'shapes': [(6, 3, 5, 5), (16, 6, 5, 5), (120, 400), (84, 120), (10, 84)],
            'base_shape': 5
            }
    networks['mednet'] = { 
            'name': 'MedNet',
            'n_layers': 5, 
            'layer_names': ['conv1', 'conv2', 'conv3', 'linear1', 'linear2'],
            'shapes': [(16, 3, 3, 3), (32, 16, 3, 3), (32, 32, 3, 3), (84, 128), (10, 64)],
            'base_shape': 3
            }
    return networks
