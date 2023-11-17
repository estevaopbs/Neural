from __future__ import annotations

import random
from math import e
from numbers import Number
from typing import Callable, List

from ..enums.NeuronDerivative import NeuronDerivative
from ..enums.NeuronFunction import NeuronFunction
from ..enums.NeuronRandomGenerator import NeuronRandomGenerator


class Neuron:
    __slots__ = (
        "_activation",
        "_weights",
        "_bias",
        "_second_step",
        "_rand_weights_generator",
        "_rand_bias_generator",
        "_derivative",
        "_evaluate",
    )

    def __init__(
        self,
        activation: Callable[[Number], Number],
        weights: List[Number] = [],
        bias: Number | None = None,
        second_step: Callable[[Number], Number] | None = None,
        rand_weights_generator: Callable[[], Number] | None = None,
        rand_bias_generator: Callable[[], Number] | None = None,
        derivative: Callable[[Number], Number] | None = None,
    ):
        self.activation = activation
        self.weights = weights
        self.bias = bias
        self.second_step = second_step
        self.rand_weights_generator = rand_weights_generator
        self.rand_bias_generator = rand_bias_generator
        self.derivative = derivative
        self._get_evaluate()

    @property
    def activation(self) -> Callable[[Number], Number]:
        return self._activation

    @activation.setter
    def activation(self, activation: Callable[[Number], Number]):
        if callable(activation):
            self._activation = activation
        else:
            raise TypeError(
                "activation must be a function that receives a number and returns a number, not {}".format(
                    type(activation)
                )
            )

    @property
    def weights(self) -> List[Number]:
        return self._weights

    @weights.setter
    def weights(self, weights: List[Number]):
        if isinstance(weights, list):
            for weight in weights:
                if not isinstance(weight, Number):
                    raise TypeError(
                        "weights must be a list of numbers, not {}".format(
                            type(weight).__name__
                        )
                    )
            self._weights = weights
        else:
            raise TypeError(
                "weights must be a list of numbers, not {}".format(
                    type(weights).__name__
                )
            )

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias: Number | None):
        if isinstance(bias, Number) or bias is None:
            self._bias = bias
        else:
            raise TypeError("bias must be None or a number, not {}".format(type(bias)))
        self._get_evaluate()

    @property
    def second_step(self):
        return self._second_step

    @second_step.setter
    def second_step(self, second_step: Callable[[Number], Number] | None):
        if callable(second_step) or second_step is None:
            self._second_step = second_step
        else:
            raise TypeError(
                "second_step must be None or a function that receives a number and returns a number, not {}".format(
                    type(second_step)
                )
            )
        self._get_evaluate()

    @property
    def rand_weights_generator(self):
        return self._rand_weights_generator

    @rand_weights_generator.setter
    def rand_weights_generator(
        self, rand_weights_generator: Callable[[], Number] | None
    ):
        if rand_weights_generator is None:
            self._rand_weights_generator = None
        elif callable(rand_weights_generator):
            self._rand_weights_generator = rand_weights_generator
        else:
            raise TypeError(
                "rand_weights_generator must be a None or function that receives no arguments and returns a number, not {}".format(
                    type(rand_weights_generator)
                )
            )

    @property
    def rand_bias_generator(self):
        return self._rand_bias_generator

    @rand_bias_generator.setter
    def rand_bias_generator(self, rand_bias_generator: Callable[[], Number] | None):
        if rand_bias_generator is None:
            self._rand_bias_generator = None
        elif callable(rand_bias_generator):
            self._rand_bias_generator = rand_bias_generator
        else:
            raise TypeError(
                "rand_bias_generator must be None or a function that receives no arguments and returns a number, not {}".format(
                    type(rand_bias_generator)
                )
            )

    @property
    def derivative(self):
        return self._derivative

    @derivative.setter
    def derivative(self, derivative: Callable[[Number], Number] | None):
        to_derivate = self.second_step if self.second_step else self.activation
        if derivative is None:
            self._derivative = None
            for f in NeuronDerivative.__members__:
                if f == to_derivate:
                    self._derivative = NeuronDerivative[f]
                    break
        elif callable(derivative):
            self._derivative = derivative
        else:
            raise TypeError(
                "derivative must be None or a function that receives a number and returns a number, not {}".format(
                    type(derivative)
                )
            )

    def _get_evaluate(self):
        if self.second_step is None:
            if self.bias is None:
                self._evaluate = self._eval_no_bias_no_second_step
            else:
                self._evaluate = self._eval_bias_no_second_step
        else:
            if self.bias is None:
                self._evaluate = self._eval_no_bias_second_step
            else:
                self._evaluate = self._eval_bias_second_step

    def _eval_no_bias_no_second_step(self, inputs: List[Number]) -> Number:
        return self.activation(inputs)

    def _eval_bias_no_second_step(self, inputs: List[Number]) -> Number:
        return self.activation(inputs + self.bias)

    def _eval_no_bias_second_step(self, inputs: List[Number]) -> Number:
        return self.second_step(self.activation(inputs))

    def _eval_bias_second_step(self, inputs: List[Number]) -> Number:
        return self.second_step(self.activation(inputs + self.bias))

    def evaluate(self, inputs: List[Number]) -> Number:
        try:
            return self._evaluate(
                sum(weight * inp for weight, inp in zip(self.weights, inputs))
            )
        except TypeError as error:
            if not hasattr(inputs, "__iter__") or not isinstance(all(inputs), Number):
                raise TypeError(
                    *error.args,
                    "inputs must be a list of numbers, not {}".format(type(inputs)),
                )
            elif len(inputs) != len(self.weights):
                raise TypeError(
                    *error.args,
                    "inputs must have the same length as weights. Weights length: {}, inputs length: {}".format(
                        len(self.weights), len(inputs)
                    ),
                )
            else:
                raise TypeError(
                    *error.args,
                    "Neuron activation and second_step functions must receive one number and return one number",
                )

    def mutate_weights(self):
        if not self.rand_weights_generator:
            raise ValueError(
                "rand_weights_generator must be a function that receives no arguments and returns a number, not None"
            )
        self.weights[
            random.choice(range(len(self.weights)))
        ] = self.rand_weights_generator()

    def mutate_bias(self):
        if not self.rand_bias_generator:
            raise ValueError(
                "rand_bias_generator must be a function that receives no arguments and returns a number, not None"
            )
        self.bias = self.rand_bias_generator()

    def mutate_weights_and_bias(self):
        if not self.rand_weights_generator:
            raise ValueError(
                "rand_weights_generator must be a function that receives no arguments and returns a number, not None"
            )
        if not self.rand_bias_generator:
            raise ValueError(
                "rand_bias_generator must be a function that receives no arguments and returns a number, not None"
            )
        self.mutate_weights()
        self.mutate_bias()

    def mutate_weights_or_bias(self):
        if not self.rand_weights_generator:
            raise ValueError(
                "rand_weights_generator must be a function that receives no arguments and returns a number, not None"
            )
        if not self.rand_bias_generator:
            raise ValueError(
                "rand_bias_generator must be a function that receives no arguments and returns a number, not None"
            )
        if random.choice([True, False]):
            self.mutate_weights()
        else:
            self.mutate_bias()

    @property
    def data(self):
        return {
            "activation": self.activation.__name__,
            "weights": self.weights,
            "bias": self.bias,
            "second_step": self.second_step.__name__,
            "rand_weights_generator": self.rand_weights_generator.__name__
            if self.rand_weights_generator
            else None,
            "rand_bias_generator": self.rand_bias_generator.__name__
            if self.rand_bias_generator
            else None,
            "derivative": self.derivative.__name__ if self.derivative else None,
        }

    @classmethod
    def from_random(
        cls,
        weights: int,
        activation: Callable[[Number], Number],
        rand_weights_generator: Callable[[], Number],
        second_step: Callable[[Number], Number] | None = None,
        rand_bias_generator: Callable[[], Number] | None = None,
        derivative: Callable[[Number], Number] | None = None,
    ) -> Neuron:
        return cls(
            activation,
            [rand_weights_generator() for _ in range(weights)],
            rand_bias_generator() if rand_bias_generator else None,
            second_step,
            rand_weights_generator,
            rand_bias_generator,
            derivative,
        )

    @classmethod
    def _function_from_data(cls, function: str) -> Callable[[Number], Number]:
        if hasattr(cls, function):
            return getattr(cls, function)
        else:
            raise ValueError(
                "the function must be one of the following: {}, not {}".format(
                    ", ".join(
                        [cls.__name__ + "." + f.value.__name__ for f in NeuronFunction]
                    ),
                    function,
                ),
                "If you need to save and load Networks that uses custom functions, you must inherit from Neuron and implement the functions in the Neuron child class or use pickle to save and load the Network",
            )

    @classmethod
    def _rand_generator_from_data(cls, function: str) -> Callable[[], Number] | None:
        if function is None:
            return None
        elif hasattr(cls, function):
            return getattr(cls, function)
        else:
            raise ValueError(
                "rand_weights_generator must be one of the following: {}, not {}".format(
                    ", ".join(
                        [
                            cls.__name__ + "." + f.value.__name__
                            for f in NeuronRandomGenerator
                        ]
                    ),
                    function,
                ),
                "If you need to save and load Networks that uses custom functions, you must inherit from Neuron and implement the functions in the Neuron child class or use pickle to save and load the Network",
            )

    @classmethod
    def _derivative_from_data(cls, derivative: str):
        if hasattr(cls, derivative):
            return getattr(cls, derivative)
        else:
            raise ValueError(
                "The derivative must be one of the following: {}, not {}".format(
                    ", ".join(
                        [
                            cls.__name__ + "." + f.value.__name__
                            for f in NeuronDerivative
                        ]
                    ),
                    derivative,
                ),
                "If you need to save and load Networks that uses custom functions, you must inherit from Neuron and implement the functions in the Neuron child class or use pickle to save and load the Network",
            )

    @classmethod
    def from_data(cls, data: dict) -> Neuron:
        activation = cls._step_function_from_data(data["activation"])
        weights = data["weights"]
        bias = data["bias"]
        second_step = cls._step_function_from_data(data["second_step"])
        rand_weights_generator = cls._rand_generator_from_data(
            data["rand_weights_generator"]
        )
        rand_bias_generator = cls._rand_generator_from_data(data["rand_bias_generator"])
        derivative = cls._derivative_from_data(data["derivative"])
        return cls(
            activation,
            weights,
            bias,
            second_step,
            rand_weights_generator,
            rand_bias_generator,
            derivative,
        )

    def enforce_weights(self, weights: int):
        if len(self.weights) > weights:
            self.weights = self.weights[:weights]
        elif len(self.weights) < weights:
            if not self.rand_weights_generator:
                raise ValueError(
                    "rand_weights_generator must be a function that receives no arguments and returns a number, not None"
                )
            self.weights += [
                self.rand_weights_generator()
                for _ in range(weights - len(self.weights))
            ]

    @staticmethod
    def linear(inp):
        return inp

    @staticmethod
    def linear_derivative(inp):
        return 1

    @staticmethod
    def sigmoid(inp):
        return 1 / (1 + e ** (-inp))

    @staticmethod
    def sigmoid_derivative(inp):
        e_minus_inp = e ** (-inp)
        return e_minus_inp / ((1 - e_minus_inp) ** 2)

    @staticmethod
    def binary(inp):
        return 1 if inp >= 0 else 0

    @staticmethod
    def binary_derivative(inp):
        return 0

    @staticmethod
    def tanh(inp):
        e_plus_inp = e**inp
        e_minus_inp = e ** (-inp)
        return (e_plus_inp - e_minus_inp) / (e_plus_inp + e_minus_inp)

    @staticmethod
    def tanh_derivative(inp):
        return 4 / ((e**inp + e ** (-inp)) ** 2)

    @staticmethod
    def relu(inp):
        return max(0, inp)

    @staticmethod
    def relu_derivative(inp):
        return 1 if inp >= 0 else 0

    @staticmethod
    def leaky_relu(inp):
        return 0.01 * inp if inp < 0 else inp

    @staticmethod
    def leaky_relu_derivative(inp):
        return 0.01 if inp < 0 else 1

    @staticmethod
    def random_f():
        return random.uniform(-1, 1)

    @staticmethod
    def random_i8():
        return random.randint(-128, 127)

    @staticmethod
    def random_i16():
        return random.randint(-32768, 32767)

    @staticmethod
    def random_i32():
        return random.randint(-2147483648, 2147483647)

    @staticmethod
    def random_i64():
        return random.randint(-9223372036854775808, 9223372036854775807)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        functions_str = str(self.activation)
        if self.second_step:
            functions_str += ", " + str(self.second_step)
        return "Neuron({})".format(functions_str)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        return (
            self.activation == __value.activation
            and self.weights == __value.weights
            and self.bias == __value.bias
            and self.second_step == __value.second_step
            and self.rand_weights_generator == __value.rand_weights_generator
            and self.rand_bias_generator == __value.rand_bias_generator
        )
