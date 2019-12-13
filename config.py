# model params - *args as tuple (in_channels, height, width, num_filters, out_dim, augmented_dim, tolerance)
# train params - *args as tuple (learning rate, epochs, batch size, number of workers)

# Hyperparam tuning
config_mnist_node = [config_mnist_node1, config_mnist_node2, config_mnist_node3, config_mnist_node4, 
                     config_mnist_node5, config_mnist_node6, config_mnist_node7, config_mnist_node8,
                     config_mnist_node9, config_mnist_node10, config_mnist_node11, config_mnist_node12,
                     config_mnist_node13, config_mnist_node14, config_mnist_node15, config_mnist_node16,
                     config_mnist_node17, config_mnist_node18]
config_mnist_anode = [config_mnist_anode1, config_mnist_anode2, config_mnist_anode3, config_mnist_anode4, 
                     config_mnist_anode5, config_mnist_anode6, config_mnist_anode7, config_mnist_anode8,
                     config_mnist_anode9, config_mnist_anode10, config_mnist_anode11, config_mnist_anode12,
                     config_mnist_anode13, config_mnist_anode14, config_mnist_anode15, config_mnist_anode16,
                     config_mnist_anode17, config_mnist_anode18]
config_cifar_node = [config_cifar_anode1, config_cifar_node2, config_cifar_node3, config_cifar_node4,
                     config_cifar_node5, config_cifar_node6, config_cifar_node7, config_cifar_node8, 
                     config_cifar_node9, config_cifar_node10, config_cifar_node11, config_cifar_node12,
                     config_cifar_node13, config_cifar_node14, config_cifar_node15, config_cifar_node16,
                     config_cifar_node17, config_cifar_node18]
config_cifar_anode = [config_cifar_anode1, config_cifar_anode2, config_cifar_anode3, config_cifar_anode4,
                     config_cifar_anode5, config_cifar_anode6, config_cifar_anode7, config_cifar_anode8, 
                     config_cifar_anode9, config_cifar_anode10, config_cifar_anode11, config_cifar_anode12,
                     config_cifar_anode13, config_cifar_anode14, config_cifar_anode15, config_cifar_anode16,
                     config_cifar_anode17, config_cifar_anode18]

#-----MNIST NODE-----
config_mnist_node1 = {
    'model' : (1, 28, 28, 32, 10, 0, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_mnist_node2 = {
    'model' : (1, 28, 28, 32, 10, 0, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_mnist_node3 = {
    'model' : (1, 28, 28, 32, 10, 0, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_mnist_node4 = {
    'model' : (1, 28, 28, 32, 10, 0, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_mnist_node5 = {
    'model' : (1, 28, 28, 32, 10, 0, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_mnist_node6 = {
    'model' : (1, 28, 28, 32, 10, 0, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

config_mnist_node7 = {
    'model' : (1, 28, 28, 64, 10, 0, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_mnist_node8 = {
    'model' : (1, 28, 28, 64, 10, 0, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_mnist_node9 = {
    'model' : (1, 28, 28, 64, 10, 0, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_mnist_node10 = {
    'model' : (1, 28, 28, 64, 10, 0, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_mnist_node11 = {
    'model' : (1, 28, 28, 64, 10, 0, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_mnist_node12 = {
    'model' : (1, 28, 28, 64, 10, 0, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

config_mnist_node13 = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_mnist_node14 = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_mnist_node15 = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_mnist_node16 = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_mnist_node17 = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_mnist_node18 = {
    'model' : (1, 28, 28, 92, 10, 0, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

#-----MNIST ANODE-----
config_mnist_anode1 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_mnist_anode2 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_mnist_anode3 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_mnist_anode4 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_mnist_anode5 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_mnist_anode6 = {
    'model' : (1, 28, 28, 32, 10, 5, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

config_mnist_anode7 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_mnist_anode8 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_mnist_anode9 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_mnist_anode10 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_mnist_anode11 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_mnist_anode12 = {
    'model' : (1, 28, 28, 64, 10, 5, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

config_mnist_anode13 = {
    'model' : (1, 28, 28, 92, 10, 5, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_mnist_anode14 = {
    'model' : (1, 28, 28, 92, 10, 5, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_mnist_anode15 = {
    'model' : (1, 28, 28, 92, 10, 5, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_mnist_anode16 = {
    'model' : (1, 28, 28, 92, 10, 5, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_mnist_anode17 = {
    'model' : (1, 28, 28, 92, 10, 5, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_mnist_anode18 = {
    'model' : (1, 28, 28, 92, 10, 5, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

#-----CIFAR NODE-----
config_cifar_node1 = {
    'model' : (3, 32, 32, 32, 10, 0, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_cifar_node2 = {
    'model' : (3, 32, 32, 32, 10, 0, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_cifar_node3 = {
    'model' : (3, 32, 32, 32, 10, 0, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_cifar_node4 = {
    'model' : (3, 32, 32, 32, 10, 0, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_cifar_node5 = {
    'model' : (3, 32, 32, 32, 10, 0, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_cifar_node6 = {
    'model' : (3, 32, 32, 32, 10, 0, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

config_cifar_node7 = {
    'model' : (3, 32, 32, 64, 10, 0, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_cifar_node8 = {
    'model' : (3, 32, 32, 64, 10, 0, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_cifar_node9 = {
    'model' : (3, 32, 32, 64, 10, 0, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_cifar_node10 = {
    'model' : (3, 32, 32, 64, 10, 0, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_cifar_node11 = {
    'model' : (3, 32, 32, 64, 10, 0, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_cifar_node12 = {
    'model' : (3, 32, 32, 64, 10, 0, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

config_cifar_node13 = {
    'model' : (3, 32, 32, 92, 10, 0, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_cifar_node14 = {
    'model' : (3, 32, 32, 92, 10, 0, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_cifar_node15 = {
    'model' : (3, 32, 32, 92, 10, 0, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_cifar_node16 = {
    'model' : (3, 32, 32, 92, 10, 0, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_cifar_node17 = {
    'model' : (3, 32, 32, 92, 10, 0, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_cifar_node18 = {
    'model' : (3, 32, 32, 92, 10, 0, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

#-----CIFAR ANODE-----
config_cifar_anode1 = {
    'model' : (3, 32, 32, 32, 10, 5, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_cifar_anode2 = {
    'model' : (3, 32, 32, 32, 10, 5, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_cifar_anode3 = {
    'model' : (3, 32, 32, 32, 10, 5, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_cifar_anode4 = {
    'model' : (3, 32, 32, 32, 10, 5, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_cifar_anode5 = {
    'model' : (3, 32, 32, 32, 10, 5, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_cifar_anode6 = {
    'model' : (3, 32, 32, 32, 10, 5, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

config_cifar_anode7 = {
    'model' : (3, 32, 32, 64, 10, 5, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_cifar_anode8 = {
    'model' : (3, 32, 32, 64, 10, 5, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_cifar_anode9 = {
    'model' : (3, 32, 32, 64, 10, 5, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_cifar_anode10 = {
    'model' : (3, 32, 32, 64, 10, 5, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_cifar_anode11 = {
    'model' : (3, 32, 32, 64, 10, 5, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_cifar_anode12 = {
    'model' : (3, 32, 32, 64, 10, 5, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}

config_cifar_anode13 = {
    'model' : (3, 32, 32, 92, 10, 5, 1e-3),
    'train' : (1e-2, 3, 32, 8)
}

config_cifar_anode14 = {
    'model' : (3, 32, 32, 92, 10, 5, 1e-3),
    'train' : (1e-2, 3, 64, 8)
}

config_cifar_anode15 = {
    'model' : (3, 32, 32, 92, 10, 5, 1e-3),
    'train' : (1e-2, 5, 32, 8)
}

config_cifar_anode16 = {
    'model' : (3, 32, 32, 92, 10, 5, 1e-3),
    'train' : (1e-2, 5, 64, 8)
}

config_cifar_anode17 = {
    'model' : (3, 32, 32, 92, 10, 5, 1e-3),
    'train' : (1e-2, 8, 32, 8)
}

config_cifar_anode18 = {
    'model' : (3, 32, 32, 92, 10, 5, 1e-3),
    'train' : (1e-2, 8, 64, 8)
}