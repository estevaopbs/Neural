import names
import random
import os.path
from math import e
import json
from copy import deepcopy


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.neural = None

    @property
    def data(self):
        return [neuron.data for neuron in self.neurons]

    def mutate(self):
        random.choice(self.neurons).mutate()

    def sign(self, neural):
        self.neural = neural
        for neuron in self.neurons:
            neuron.sign(self)


class Neuron:
    def __init__(self, function, weight=None, bias=None, second_step=None,
                 rand_range=10, custom_function=False, custom_second_step=False):
        self.function = None
        self.function_name = None
        self.weight = weight
        self.bias = bias
        self.second_step = None
        self.second_step_name = None
        self.custom_function = custom_function
        self.custom_second_step = custom_second_step
        self.mutate = None
        self.rand_range = rand_range
        self.evaluate = None
        self.random_number = None
        self.layer = None
        self._get_parameters(function, rand_range, second_step, custom_second_step)

    def sign(self, layer):
        self.layer = layer

    def _get_parameters(self, function, rand_range, second_step, custom_second_step):
        if rand_range is not None:
            self.random_number = lambda: random.uniform(-rand_range, rand_range)
        if not self.custom_function:
            self.function_name = function
            self._get_function(function)
        else:
            self.function_name = function.__name__
            self.function = function
        if second_step is not None:
            if not custom_second_step:
                self.second_step = Neuron._get_second_step(second_step)
            else:
                self.second_step = second_step
            self.second_step_name = self.second_step.__name__
            self.evaluate = self._evaluate_with_second_step
        else:
            self.second_step_name = None
            self.evaluate = self._evaluate
        if self.bias is None and self.weight is None:
            self.mutate = self._try_mutate_another
            self.random_number = None
        elif self.bias is None:
            self.mutate = self._mutate_weight
        elif self.bias is not None and self.weight is not None:
            self.mutate = self._mutate_all

    def _try_mutate_another(self):
        self.layer.neural.mutate()

    @property
    def data(self):
        return {"function": self.function_name, "weight": self.weight, "bias": self.bias,
                "second_step": self.second_step_name, "rand_range": self.rand_range,
                "custom_function": self.custom_function, "custom_second_step": self.custom_second_step}

    def _mutate_weight(self):
        self.weight = self.random_number()

    def _mutate_bias(self):
        self.bias = self.random_number()

    def _mutate_all(self):
        random_parameter = random.choice([self._mutate_weight, self._mutate_bias])
        random_parameter()

    @staticmethod
    def random(function=None, second_step=None, has_weight=False, has_bias=False,
               rand_range=10, custom_function=False, custom_second_step=False):
        weight = random.uniform(-rand_range, rand_range) if has_weight else None
        bias = random.uniform(-rand_range, rand_range) if has_bias else None
        return Neuron(function, weight, bias, second_step, rand_range, custom_function, custom_second_step)

    @staticmethod
    def check_parameters(function, weight, bias):
        if function is Neuron.binary or function is Neuron.ReLU or function is Neuron.Leaky_ReLU:
            if weight or bias is not None:
                raise NameError(f'{function.__name__} do not use weight or bias. Keep it None.')
            else:
                return
        if weight is None:
            raise NameError(f'{function.__name__} needs a weight, but it was not included.')
        else:
            return

    @staticmethod
    def _get_second_step(second_step):
        second_step = second_step.lower()
        if second_step == 'binary':
            function = Neuron.binary
            return function
        elif second_step == 'relu':
            function = Neuron.ReLU
            return function
        elif second_step == 'leaky relu':
            function = Neuron.Leaky_ReLU
            return function
        raise NameError('Neuron\'s second step was not properly selected.')

    def _evaluate(self, _input):
        if self.weight is not None or self.bias is not None:
            return self.function(self, _input)
        else:
            return self.function(_input)

    def _evaluate_with_second_step(self, _input):
        return self.second_step(self.function(self, _input))

    @staticmethod
    def linear(neuron, _input):
        return _input * neuron.weight

    @staticmethod
    def biased_linear(neuron, _input):
        return _input * neuron.weight + neuron.bias

    @staticmethod
    def sigmoid(neuron, _input):
        return 1 / (1 + e ** (- neuron.weight * _input))

    @staticmethod
    def biased_sigmoid(neuron, _input):
        return 1 / (1 + e ** (- neuron.weight * _input + neuron.bias))

    @staticmethod
    def binary(_input):
        return 1 if _input >= 0 else 0

    @staticmethod
    def tanh(neuron, _input):
        return neuron.weight * ((2 / (1 + e ** (- neuron.weight * 2 * _input))) - 1)

    @staticmethod
    def biased_tanh(neuron, _input):
        return neuron.weight * ((2 / (1 + e ** (- neuron.weight * 2 * _input + neuron.bias))) - 1)

    @staticmethod
    def ReLU(_input):
        return max(0, _input)

    @staticmethod
    def Leaky_ReLU(_input):
        return 0.01 * _input if _input < 0 else _input

    def _get_function(self, function):
        function = function.lower()
        if function == 'linear':
            if self.bias is not None:
                self.function = Neuron.biased_linear
            else:
                self.function = Neuron.linear
        elif function == 'sigmoid':
            if self.bias is not None:
                self.function = Neuron.biased_sigmoid
            else:
                self.function = Neuron.sigmoid
        elif function == 'binary':
            self.function = Neuron.binary
        elif function == 'tanh':
            if self.bias is not None:
                self.function = Neuron.biased_tanh
            else:
                self.function = Neuron.tanh
        elif function == 'relu':
            self.function = Neuron.ReLU
        elif function == 'leaky relu':
            self.function = Neuron.Leaky_ReLU
        Neuron.check_parameters(self.function, self.weight, self.bias)


class Neural:
    def __init__(self, inputs=None, layers=None, name=None):
        self.name = None
        self.layers = layers  # receives the layers of the neural network
        self.inputs = inputs  # receives the number of inputs wanted
        self._get_name(name)
        self.sign()

    def sign(self):  # signs it's layers as its own
        for layer in self.layers:
            layer.sign(self)

    def _get_name(self, name):  # identifies the neural network with a name tag
        if name is True:
            self.name = names.get_full_name()
        elif name is False or name is None:
            self.name = str(self.__hash__())
        if name is not True and name is not False and name is not None and name is not True and type(name) is not str:
            raise NameError('name has not an appropriate type.')
        return name

    @property
    def data(self):  # returns all data that determine the neural network as a dict
        return {self.name: {"layers": [layer.data for layer in self.layers], "inputs": self.inputs}}

    @property  # returns the number of neurons in each of the hidden layers in as a list
    def hidden_neurons(self):
        return [len(layer.neurons) for layer in self.layers[0:-1]]

    @property  # returns the number of output neurons in the network
    def outputs(self):
        return len(self.layers[-1].neurons)

    @property  # returns the number of hidden layers in the network
    def hidden_layers(self):
        return len(self.layers) - 1

    def write_data(self, document=None, directory='data'):
        directory = directory + '/'
        if document is None:
            file_name = self.name
        else:
            file_name = document
        if not os.path.exists(directory):
            os.mkdir(directory)
        file_address = f'{directory}{file_name}.json'
        if os.path.exists(file_address):
            with open(file_address, 'r+') as file:
                content = json.load(file)
                content.update(self.data)
                file.seek(0)
                json.dump(content, file, indent=6)
        else:
            with open(file_address, 'a+') as file:
                json.dump(self.data, file, indent=6)

    def evaluate(self, inputs):
        for layer in self.layers:
            layer_output = []
            for neuron in layer.neurons:
                neuron_output = 0
                for _input in inputs:
                    neuron_output += neuron.evaluate(_input)
                layer_output.append(neuron_output)
            inputs = layer_output
        return inputs

    def mutate(self):
        random.choice(self.layers).mutate()


def load_data(document, name=None, keep_name=False, directory='data'):
    directory = directory + '/'
    document_address = directory + document + '.json'
    if not os.path.exists(directory):
        os.mkdir(directory)
        raise NameError(f"There's no '{document}' in '{directory}'.")
    if not os.path.isfile(document_address):
        raise NameError(f"There's no '{document}' in the {directory}.")
    with open(document_address, 'r') as file:
        content = json.load(file)
        if name is not None:
            neural_layers = content[name]['layers']
            layers = _get_layers(neural_layers)
            inputs = content[name]['inputs']
            if not keep_name:
                name = None
            return Neural(inputs, layers, name)
        else:
            neural_names = list(content.keys())
            saved_content = []
            for n, neural in enumerate(content):
                neural_name = neural_names[n]
                neural = content[neural_name]
                neural_layers = neural['layers']
                inputs = neural['inputs']
                layers = _get_layers(neural_layers)
                if not keep_name:
                    neural_name = None
                saved_content.append(Neural(inputs, layers, neural_name))
            return saved_content


def _get_layers(neural_layers):
    layers = []
    for n, layer in enumerate(neural_layers):
        layers.append(Layer([]))
        for neuron in layer:
            function = neuron['function']
            weight = neuron['weight']
            bias = neuron['bias']
            second_step = neuron['second_step']
            rand_range = neuron['rand_range']
            custom_function = neuron['custom_function']
            custom_second_step = neuron['custom_second_step']
            layers[n].neurons.append(Neuron(function, weight, bias, second_step,
                                            rand_range, custom_function, custom_second_step))
    return layers


def random_homogeneous_neural(neurons_in_layer, neurons_function=None, neurons_second_step=None, weight=True,
                              bias=False, rand_range=10, custom_function=False, custom_second_step=False, name=None):
    if not custom_function:
        Neuron.check_parameters(neurons_function, weight, bias)
    inputs = neurons_in_layer[0]
    layers = []
    for _layer in neurons_in_layer[1:]:
        layer = []
        for neuron in range(_layer):
            layer.append(Neuron.random(neurons_function, neurons_second_step, weight, bias, rand_range,
                                       custom_function, custom_second_step))
        layers.append(Layer(layer))
    return Neural(inputs, layers, name)


def crossover(parent, donor, name=False):
    donor_layer_index = random.choice(list(range(len(donor.layers))))
    layers = deepcopy(parent.layers)
    layers[donor_layer_index] = deepcopy(donor.layers[donor_layer_index])
    if name:
        parent_family_name = parent.name.split(' ')[1:]
        donor_family_name = donor.name.split(' ')[1:]
        child_family_name = []
        for family_name in donor_family_name:
            if family_name in parent_family_name:
                continue
            child_family_name.append(family_name)
        child_family_name += parent_family_name
        name = names.get_first_name() + ' ' + ' '.join(child_family_name)
    return Neural(parent.inputs, layers, name)
