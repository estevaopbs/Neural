from enum import Enum

from ..entities.Neuron import Neuron


class NeuronDerivative(Enum):
    LINEAR = Neuron.linear_derivative
    SIGMOID = Neuron.sigmoid_derivative
    BINARY = Neuron.binary_derivative
    TANH = Neuron.tanh_derivative
    RELU = Neuron.relu_derivative
    LEAKY_RELU = Neuron.leaky_relu_derivative
