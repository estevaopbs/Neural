import random
from copy import deepcopy

from ..entities.Network import Network


def layer_crossover(parent: Network, donor: Network, layers: int = 1) -> Network:
    if len(parent.layers) != len(donor.layers):
        raise ValueError("Networks are incompatible. Layer counts do not match.")
    for parent_layer, donor_layer in zip(parent.layers, donor.layers):
        if parent_layer.size != donor_layer.size:
            raise ValueError("Networks are incompatible. Layer sizes do not match.")
    if layers > len(parent.layers) / 2 - 1 or layers < 1:
        raise ValueError(
            "Invalid number of layers to crossover. Must be between 1 and half the number of layers - 1."
        )
    donor_layer_indexes = random.sample(range(len(parent.layers)), layers)
    layers = deepcopy(parent.layers)
    for donor_layer_index in donor_layer_indexes:
        layers[donor_layer_index] = deepcopy(donor.layers[donor_layer_index])
    return parent.__class__(layers)


def neuron_crossover(parent: Network, donor: Network, neurons: int) -> Network:
    if len(parent.layers) != len(donor.layers):
        raise ValueError("Networks are incompatible. Layer counts do not match.")
    neurons_count = 0
    layers_count = 0
    neurons_indexes = []
    for parent_layer, donor_layer in zip(parent.layers, donor.layers):
        if parent_layer.size != donor_layer.size:
            raise ValueError("Networks are incompatible. Layer sizes do not match.")
        neurons_indexes = [(layers_count, n) for n in range(parent_layer.size)]
        neurons_count += parent_layer.size
        layers_count += 1
    if neurons > neurons_count / 2 - 1 or neurons < 1:
        raise ValueError(
            "Invalid number of neurons to crossover. Must be between 1 and half the number of neurons - 1."
        )
    children_layers = deepcopy(parent.layers)
    donor_neurons_indexes = random.sample(neurons_indexes, neurons)
    for layer_index, neuron_index in donor_neurons_indexes:
        children_layers[layer_index].neurons[neuron_index] = deepcopy(
            donor.layers[layer_index].neurons[neuron_index]
        )
    return parent.__class__(children_layers)
