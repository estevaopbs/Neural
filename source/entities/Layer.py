from __future__ import annotations

import random
from numbers import Number
from typing import Callable, List, Type

from .Neuron import Neuron


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

    @classmethod
    def from_data(cls, data, NeuronClass: Type[Neuron] = Neuron) -> Layer:
        return cls([NeuronClass.from_data(datum) for datum in data])

    @classmethod
    def from_random(
        cls,
        size: int,
        weights: int,
        first_step: Callable[[Number], Number],
        rand_weights_generator: Callable[[], Number],
        second_step: Callable[[Number], Number] | None = None,
        rand_bias_generator: Callable[[], Number] | None = None,
        derivative: Callable[[Number], Number] | None = None,
        NeuronClass: Type[Neuron] = Neuron,
    ) -> Layer:
        return cls(
            [
                NeuronClass.from_random(
                    weights,
                    first_step,
                    rand_weights_generator,
                    second_step,
                    rand_bias_generator,
                    derivative,
                )
                for _ in range(size)
            ]
        )

    def enforce_weights(self, weights: int):
        for neuron in self.neurons:
            neuron.enforce_weights(weights)

    def evaluate(self, inputs: List[Number]) -> List[Number]:
        return [neuron.evaluate(inputs) for neuron in self.neurons]

    def mutate_weight(self):
        random.choice(self.neurons).mutate_weight()

    def mutate_bias(self):
        random.choice(self.neurons).mutate_bias()

    def mutate_weights(self, neurons: int):
        for _ in range(neurons):
            self.mutate_weight()

    def mutate_biases(self, neurons: int):
        for _ in range(neurons):
            self.mutate_bias()

    def __repr__(self) -> str:
        return "Layer({})".format(self.size)

    def __str__(self) -> str:
        return str(self.neurons)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        return self.neurons == __value.neurons
