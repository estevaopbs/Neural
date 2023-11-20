from __future__ import annotations

import json
import pickle
import random
from numbers import Number
from pathlib import Path
from typing import Callable, List, Type

from .layer import Layer
from .neuron import Neuron
from ..cases.crossover import layer_crossover, neuron_crossover


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
        rand_weight: Callable[[], Number],
        second_step: Callable[[Number], Number] | None = None,
        rand_bias: Callable[[], Number] | None = None,
        derivative: Callable[[Number], Number] | None = None,
        LayerClass: Type[Layer] = Layer,
        NeuronClass: Type[Neuron] = Neuron,
    ) -> Network:
        return cls(
            [
                LayerClass.from_random(
                    size,
                    activation,
                    weights,
                    rand_weight,
                    second_step,
                    rand_bias,
                    derivative,
                    NeuronClass,
                )
                for size, weights in zip(sizes, [1] + sizes[:-1])
            ]
        )

    def feed_forward(self, inputs: List[Number]) -> List[Number]:
        for layer in self.layers:
            inputs = layer.evaluate(inputs)
        return inputs

    def attune_weights(self):
        for layer, weight in zip(
            self.layers, [1] + [layer.size for layer in self.layers[:-1]]
        ):
            layer._enforce_weights(weight)

    def mutate_weights(self, neurons, weights: int = 1):
        mutable_layers_indexes = []
        for n, layer in enumerate(self.layers):
            mutable_neurons = len(layer.weight_mutable_neurons)
            if mutable_neurons:
                mutable_layers_indexes += [n] * mutable_neurons
        if len(mutable_layers_indexes) < neurons:
            raise ValueError(
                "Invalid number of neurons to mutate. Must be less than or equal to the number of mutable neurons."
            )
        layers_to_mutate = random.sample(mutable_layers_indexes, neurons)
        for layer_index in set(layers_to_mutate):
            self.layers[layer_index].mutate_weights(
                layers_to_mutate.count(layer_index), weights
            )

    def mutate_biases(self, neurons: int = 1):
        mutable_layers_indexes = []
        for n, layer in enumerate(self.layers):
            mutable_neurons = len(layer.bias_mutable_neurons)
            if mutable_neurons:
                mutable_layers_indexes += [n] * mutable_neurons
        if len(mutable_layers_indexes) < neurons:
            raise ValueError(
                "Invalid number of neurons to mutate. Must be less than or equal to the number of mutable neurons."
            )
        layers_to_mutate = random.sample(mutable_layers_indexes, neurons)
        for layer_index in set(layers_to_mutate):
            self.layers[layer_index].mutate_biases(layers_to_mutate.count(layer_index))

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

    def layer_crossover(self, donor: Network, layers: int = 1) -> Network:
        return layer_crossover(self, donor, layers)

    def neuron_crossover(self, donor: Network, neurons: int = 1) -> Network:
        return neuron_crossover(self, donor, neurons)

    def __repr__(self) -> str:
        return "Network({})".format(self.size)

    def __str__(self) -> str:
        return str(self.data)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        return self.layers == __value.layers
