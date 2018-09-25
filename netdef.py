def nets():
    networks = {}
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
    networks['mc'] = {
            'name': 'Small2_MC',
            'n_layers': 3,
            'layer_names': ['conv1.0', 'conv2.0', 'linear'],
            'shapes': [(32, 1, 5, 5), (32, 32, 5, 5), (10, 512)],
            'base_shape': 5
            }
    networks['cnet'] = {
            'name': 'CNet',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2', 'fc1', 'fc2'],
            'shapes': [(64, 3, 3, 3), (128, 64, 3, 3), (128, 8192), (10, 128)],
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
            'layer_names': ['conv1', 'conv2', 'conv3', 'fc1', 'fc2'],
            'shapes': [(16, 3, 3, 3), (32, 16, 3, 3), (32, 32, 3, 3), (64, 128), (10, 64)],
            'base_shape': 3
            }
    return networks
