from enum import Enum

from ..entities.Neuron import Neuron


class NeuronFunction(Enum):
    LINEAR = Neuron.linear
    SIGMOID = Neuron.sigmoid
    BINARY = Neuron.binary
    TANH = Neuron.tanh
    RELU = Neuron.relu
    LEAKY_RELU = Neuron.leaky_relu
