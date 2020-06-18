def nets():
    networks = {}
    networks['small'] = {
            'name': 'Small',
            'n_layers': 3,
            'layer_names': ['conv1.0', 'conv2.0', 'linear'],
            'shapes': [(32, 1, 5, 5), (32, 32, 5, 5), (10, 512)],
            'base_shape': 5
            }
    networks['small3'] = {
            'name': 'Small3',
            'n_layers': 5,
            'layer_names': ['conv1.0', 'conv2.0', 'linear1', 'linear2'],
            'shapes': [(32, 1, 5, 5), (32, 64, 5, 5), (1024, 1024), (10, 1024)],
            'base_shape': 5
            }
    networks['lenet'] = { 
            'name': 'LeNet',
            'n_layers': 5, 
            'layer_names': ['conv1', 'conv2', 'linear1', 'linear2', 'linear3'],
            'shapes': [(6, 3, 5, 5), (16, 6, 5, 5), (120, 400), (84, 120), (10, 84)],
            'base_shape': 5
            }
    networks['mednet2'] = { 
            'name': 'MedNet',
            'n_layers': 5, 
            'layer_names': ['conv1', 'conv2', 'conv3', 'fc1', 'fc2'],
            'shapes': [(16, 3, 3, 3), (32, 16, 3, 3), (32, 32, 3, 3), (64, 128), (10, 64)],
            'base_shape': 3
            }
    return networks
