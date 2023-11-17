from enum import Enum

from ..entities.Neuron import Neuron


class NeuronRandomGenerator(Enum):
    RANDOMF = Neuron.random_f
    RANDOMI8 = Neuron.random_i8
    RANDOMI16 = Neuron.random_i16
    RANDOMI32 = Neuron.random_i32
    RANDOMI64 = Neuron.random_i64
