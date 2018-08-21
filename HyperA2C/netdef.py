def nets():
    networks = {}
    networks['a2c'] = {
            'name': 'A2C',
            'n_layers': 4,
            'layer_names': ['conv1', 'conv2'],
            'shapes': [(1, 128, 3, 3), (128, 256, 3, 3), (1024, 1024), (1024, 10)],
            'base_shape': 3
    }
    return networks
