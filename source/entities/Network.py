from __future__ import annotations

import json
import pickle
import random
from numbers import Number
from pathlib import Path
from typing import Callable, List, Type

from .Layer import Layer
from .Neuron import Neuron


class Network:
    __slots__ = "_layers"

    def __init__(self, layers: List[Layer] = []):
        self.layers = layers

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, layers):
        if isinstance(layers, list):
            for layer in layers:
                if not isinstance(layer, Layer):
                    raise TypeError(
                        "Layers must be a list of Layer objects, not {}".format(
                            type(layer)
                        )
                    )
            self._layers = layers
        else:
            raise TypeError(
                "Layers must be a list of Layer objects, not {}".format(type(layers))
            )

    @property
    def data(self):
        return [layer.data for layer in self.layers]

    @classmethod
    def from_data(
        cls, data, LayerClass: Type[Layer] = Layer, NeuronClass: Type[Neuron] = Neuron
    ) -> Network:
        return cls([LayerClass.from_data(datum, NeuronClass) for datum in data])

    @classmethod
    def from_random(
        cls,
        sizes: List[int],
        activation: Callable[[Number], Number],
        rand_weights_generator: Callable[[], Number],
        second_step: Callable[[Number], Number] | None = None,
        rand_bias_generator: Callable[[], Number] | None = None,
        derivative: Callable[[Number], Number] | None = None,
        LayerClass: Type[Layer] = Layer,
        NeuronClass: Type[Neuron] = Neuron,
    ) -> Network:
        return cls(
            [
                LayerClass.from_random(
                    size,
                    weights,
                    activation,
                    rand_weights_generator,
                    second_step,
                    rand_bias_generator,
                    derivative,
                    NeuronClass,
                )
                for size, weights in zip(sizes, [1] + sizes[:-1])
            ]
        )

    def evaluate(self, inputs: List[Number]) -> List[Number]:
        for layer in self.layers:
            inputs = layer.evaluate(inputs)
        return inputs

    def enforce_weights(self):
        for layer, weights in zip(
            self.layers, [1] + [layer.size for layer in self.layers[:-1]]
        ):
            layer.enforce_weights(weights)

    def mutate_weight(self):
        random.choice(self.layers).mutate_weight()

    def mutate_bias(self):
        random.choice(self.layers).mutate_bias()

    def mutate_layer_weights(self):
        random.choice(self.layers).mutate_weights()

    def mutate_layer_biases(self):
        random.choice(self.layers).mutate_biases()

    def mutate_weights(self):
        for layer in self.layers:
            layer.mutate_weights()

    def mutate_biases(self):
        for layer in self.layers:
            layer.mutate_biases()

    def to_json(self, filename: Path):
        with open(filename, "w") as f:
            json.dump(self.data, f)

    @classmethod
    def from_json(cls, filename: Path):
        with open(filename, "r") as f:
            return cls.from_data(json.load(f))

    def to_pickle(self, filename: Path):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filename: Path):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @property
    def size(self):
        return len(self.layers)

    def __repr__(self) -> str:
        return "Network({})".format(self.size)

    def __str__(self) -> str:
        return str(self.data)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        return self.layers == __value.layers
