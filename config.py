# model params - *args as tuple (in_channels, height, width, num_filters, out_dim, augmented_dim, tolerance)
# train params - *args as tuple (learning rate, epochs, batch size, number of workers)

config_mnist_node = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_mnist_anode = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_cifar_node = {
    'model' : (3, 32, 32, 92, 10, 0, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_cifar_anode = {
    'model' : (3, 32, 32, 64, 10, 5, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}