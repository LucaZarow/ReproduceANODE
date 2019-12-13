# model params - *args as tuple (in_channels, height, width, num_filters, out_dim, augmented_dim, tolerance)
# train params - *args as tuple (learning rate, epochs, batch size, number of workers)


#-----BASELINES-----
baseline_config_mnist_node = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-3, 15, 32, 12)
}

baseline_config_mnist_anode = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-3, 15, 32, 12)
}

baseline_config_cifar_node = {
    'model' : (3, 32, 32, 125, 10, 0, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

baseline_config_cifar_anode = {
    'model' : (3, 32, 32, 64, 10, 10, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

#-----MNIST NODE-----
config_mnist_node1 = {
    'model' : (1, 28, 28, 32, 10, 0, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_node2 = {
    'model' : (1, 28, 28, 64, 10, 0, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}


config_mnist_node3 = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}


#-----MNIST ANODE-----
config_mnist_anode1 = {
    'model' : (1, 28, 28, 32, 10, 3, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode2 = {
    'model' : (1, 28, 28, 64, 10, 3, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode3 = {
    'model' : (1, 28, 28, 92, 10, 3, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode4 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode5 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode6 = {
    'model' : (1, 28, 28, 92, 10, 8, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode7 = {
    'model' : (1, 28, 28, 32, 10, 8, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode8 = {
    'model' : (1, 28, 28, 64, 10, 8, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode9 = {
    'model' : (1, 28, 28, 92, 10, 8, 1e-3),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode10 = {
    'model' : (1, 28, 28, 32, 10, 3, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode11 = {
    'model' : (1, 28, 28, 64, 10, 3, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode12 = {
    'model' : (1, 28, 28, 92, 10, 3, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode13 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode14 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode15 = {
    'model' : (1, 28, 28, 92, 10, 8, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode16 = {
    'model' : (1, 28, 28, 32, 10, 8, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode17 = {
    'model' : (1, 28, 28, 64, 10, 8, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode18 = {
    'model' : (1, 28, 28, 92, 10, 8, 1e-2),
    'train' : (1e-3, 10, 32, 12)
}

config_mnist_anode19 = {
    'model' : (1, 28, 28, 32, 10, 3, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode20 = {
    'model' : (1, 28, 28, 64, 10, 3, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode21 = {
    'model' : (1, 28, 28, 92, 10, 3, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode22 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode23 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode24 = {
    'model' : (1, 28, 28, 92, 10, 8, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode25 = {
    'model' : (1, 28, 28, 32, 10, 8, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode26 = {
    'model' : (1, 28, 28, 64, 10, 8, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode27 = {
    'model' : (1, 28, 28, 92, 10, 8, 1e-3),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode28 = {
    'model' : (1, 28, 28, 32, 10, 3, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode29 = {
    'model' : (1, 28, 28, 64, 10, 3, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode30 = {
    'model' : (1, 28, 28, 92, 10, 3, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode31 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode32 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode33 = {
    'model' : (1, 28, 28, 92, 10, 8, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode34 = {
    'model' : (1, 28, 28, 32, 10, 8, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode35 = {
    'model' : (1, 28, 28, 64, 10, 8, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}

config_mnist_anode36 = {
    'model' : (1, 28, 28, 92, 10, 8, 1e-2),
    'train' : (1e-2, 10, 32, 12)
}



# Hyperparam tuning
config_mnist_node = [config_mnist_node1, config_mnist_node2, config_mnist_node3]

config_mnist_anode = [config_mnist_anode1, config_mnist_anode2, config_mnist_anode3, config_mnist_anode4, 
                     config_mnist_anode5, config_mnist_anode6, config_mnist_anode7, config_mnist_anode8,
                     config_mnist_anode9, config_mnist_anode10, config_mnist_anode11, config_mnist_anode12,
                     config_mnist_anode13, config_mnist_anode14, config_mnist_anode15, config_mnist_anode16,
                     config_mnist_anode17, config_mnist_anode18, config_mnist_anode19, config_mnist_anode20,
                     config_mnist_anode21, config_mnist_anode22, config_mnist_anode23, config_mnist_anode24,
                     config_mnist_anode25, config_mnist_anode26, config_mnist_anode27, config_mnist_anode28,
                     config_mnist_anode29, config_mnist_anode30, config_mnist_anode31, config_mnist_anode32,
                     config_mnist_anode33, config_mnist_anode34, config_mnist_anode35, config_mnist_anode36]