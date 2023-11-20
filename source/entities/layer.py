from __future__ import annotations

import random
from numbers import Number
from typing import Callable, List, Type

from .neuron import Neuron


class Layer:
    __slots__ = "_neurons"

    def __init__(self, neurons: List[Neuron] = []):
        self.neurons = neurons

    @property
    def neurons(self):
        return self._neurons

    @neurons.setter
    def neurons(self, neurons: List[Neuron]):
        if isinstance(neurons, list):
            for neuron in neurons:
                if not isinstance(neuron, Neuron):
                    raise TypeError(
                        "Neurons must be a list of Neuron objects, not {}".format(
                            type(neuron)
                        )
                    )
            self._neurons = neurons
        else:
            raise TypeError(
                "Neurons must be a list of Neruon objects, not {}".format(type(neurons))
            )

    @property
    def size(self):
        return len(self.neurons)

    @property
    def data(self):
        return [neuron.data for neuron in self.neurons]

    @property
    def bias_mutable_neurons(self):
        return [neuron for neuron in self.neurons if neuron.rand_bias is not None]

    @property
    def weight_mutable_neurons(self) -> List[Neuron]:
        return [neuron for neuron in self.neurons if neuron.rand_weight is not None]

    @classmethod
    def from_data(cls, data, NeuronClass: Type[Neuron] = Neuron) -> Layer:
        return cls([NeuronClass.from_data(datum) for datum in data])

    @classmethod
    def from_random(
        cls,
        size: int,
        activation: Callable[[Number], Number],
        weights: int,
        rand_weight: Callable[[], Number],
        second_step: Callable[[Number], Number] | None = None,
        rand_bias: Callable[[], Number] | None = None,
        derivative: Callable[[Number], Number] | None = None,
        NeuronClass: Type[Neuron] = Neuron,
    ) -> Layer:
        return cls(
            [
                NeuronClass.from_random(
                    weights, activation, rand_weight, second_step, rand_bias, derivative
                )
                for _ in range(size)
            ]
        )

    def _attune_weights(self, weights: int):
        for neuron in self.neurons:
            neuron._attune_weights(weights)

    def evaluate(self, inputs: List[Number]) -> List[Number]:
        return [neuron.evaluate(inputs) for neuron in self.neurons]

    def mutate_weights(self, neurons: int = 1, weights: int = 1):
        if not isinstance(neurons, int):
            raise TypeError("Neurons must be an integer, not {}".format(type(neurons)))
        mutable_neurons = self.weight_mutable_neurons
        if neurons > len(mutable_neurons) or neurons < 1:
            raise ValueError(
                "Invalid number of neurons to mutate. Must be between 1 and the number of mutable neurons."
            )
        for neuron in random.sample(mutable_neurons, neurons):
            neuron.mutate_weights(weights)

    def mutate_biases(self, neurons: int):
        mutable_neurons = self.bias_mutable_neurons
        if not isinstance(neurons, int):
            raise TypeError("Neurons must be an integer, not {}".format(type(neurons)))
        if neurons > len(mutable_neurons) or neurons < 1:
            raise ValueError(
                "Invalid number of neurons to mutate. Must be between 1 and the number of mutable neurons."
            )
        for neuron in random.sample(mutable_neurons, neurons):
            neuron.mutate_bias()

    def __repr__(self) -> str:
        return "Layer({})".format(self.size)

    def __str__(self) -> str:
        return str(self.neurons)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        return self.neurons == __value.neurons
