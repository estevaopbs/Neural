import copy
import names
import random
import os.path
from math import e
import json
from copy import deepcopy


class Layer:
    """
    Layer([...]) creates a layer of neurons by receiving a iterable of neurons.

    The properties/methods of a layer are
    neurons -> Its iterable of neurons (list, array, tuple, ...);
    neural -> The neural network which the layer belongs;
    data -> The list which contains all the data needed to determine the layer;
    mutate -> The method by which the layer can suffer a mutation by choosing randomly one of its neurons and
    calling its mutate method;
    sign -> The method by which it is signed by a neural network as its own. It also signs its neurons as its own;
    It is automatically called when a neural network with the related layer is created.

    All these properties/methods can be seen by neuron.respective_property or neuron.respective_property() if its
    callable.

    Example of layer:
    layer = Layer([Neuron(...), Neuron(...), Neuron(...), Neuron(...)])
    """

    def __init__(self, neurons):
        self.neurons = neurons
        self.neural = None

    @property
    def data(self):
        """
        Returns all data that determine the layer as a list.
        """
        return [neuron.data for neuron in self.neurons]

    def mutate(self):
        """
        Provokes a mutation in the layer changing one parameter (bias or weight) of a random neuron
        """
        random.choice(self.neurons).mutate()

    def sign(self, neural):
        """
        Receives the signature of the neural network, enabling it to be accessed from the layer. Also signs each
        neuron of the layer as its own.
        """
        self.neural = neural
        for neuron in self.neurons:
            neuron.sign(self)


class Neuron:
    """
    The fundamental component of the neural network. The neurons are objects which has each one its own function,
    weight, bias, random number generator and, possibly, a second step function.

    It generates a neuron by receiving a function, which is supposed to be a string correspondent to one of the built-in
    functions that are 'linear', 'sigmoid', 'tanh', 'binary', 'relu', 'leaky relu' or the own function if it is a custom
    function. It also receives a weight and/or bias if needed for the function. The 'second_step' variable receives a
    function that will be executed on the result of the main function before the neuron give its output. This function
    is not supposed to have a weight or a bias. It must receive a string correspondent to one of the built-in functions
    which does not accept weight or bias. They are: 'binary', 'relu' (rectified linear unity) and 'leaky relu'. But if
    you want to use a custom second step function, as in the main function, it must receive the own function. The
    'rand_range' variable receives the upper limit of the neuron's random number generator which will feed its mutate
    function. The lower limit is the negative of the upper. It means that if some neuron has rand_range equal 10, when
    we call its mutate function, the next generated weight or bias will be in the range (-10, 10). 'custom_function' and
    'custom_second_step' are variables which receives boolean values 'True' or 'False' for enable or disable,
    respectively, the use of a custom main function or custom second step function respectively.

    The properties/methods of a neuron are
    function -> Its main function;
    function_name -> The string which name the main function;
    weight -> Its multiplicative factor. When it's 'None', it will never be changed by a mutate call;
    bias -> Its additive factor. When it's 'None', it will work as zero in the built-in functions, and it will never be
    changed by a mutate call.
    second_step -> Its second step function;
    second_step_name -> The string which name the second step function;
    custom_function -> Boolean value that enables (True) or disable (False) the use of a alternative function;
    mutate -> Mutate method proper of the neuron that can be called by neuron.mutate() causing it to mutate;
    rand_value -> The upper limit of the neuron's random number generator. The lower is its negative. It's 10 by
    default;
    evaluate -> The method which the neuron receives a input and outputs its evaluation;
    self.random_number -> The method the neuron uses to generate random numbers;
    layer -> The layer which the neuron belongs;
    data -> A dictionary with all the information needed to determine the neuron;
    random -> Static method which returns a neuron with random weight and bias by receiving a main function and possibly
    a second step function;
    check_parameters -> Check, for built-in functions, if weight or bias are acceptable, raising error when a neuron
    is incoherent. It is automatically called when the neuron is created;
    sign ->  Method by which it is signed by a layer as its own. It is automatically called when a layer with the
    related neuron is created.

    All these properties/methods can be seen by neuron.respective_property or neuron.respective_property() if its
    callable.

    Built-in functions' explanation:
    linear -> returns weight * input + bias;
    sigmoid -> returns 1 / (1 + e ^ - (weight * input + bias));
    tanh -> returns tanh(weight * input + bias);
    binary -> returns 1 if the input is greater or equal 0 otherwise returns 0;
    relu -> returns maximum value between zero and the input;
    leaky relu -> returns (0.01 * input) if the input is less than zero otherwise returns the input.

    Example of neuron:
    neuron = Neuron(function='linear', weight=1, bias=None, second_step='binary',
                        rand_range=100, custom_function=False, custom_second_step=False)
    """

    def __init__(self, function, weights=None, bias=None, second_step=None, rand_weights_range=10,
                 rand_bias_range=None, custom_function=False, custom_second_step=False):
        self.function = None
        self.function_name = None
        self.weights = weights
        self.bias = bias
        self.second_step = None
        self.second_step_name = None
        self.custom_function = custom_function
        self.custom_second_step = custom_second_step
        self.mutate = None
        self.rand_weights_range = rand_weights_range
        self.rand_bias_range = rand_bias_range
        self.evaluate = None
        self.rand_weight = None
        self.rand_bias = None
        self.layer = None
        self.input_neuron = False
        self._get_parameters(function, second_step)

    def sign(self, layer):
        """
        Receives the signature of the layer it belongs, enabling the neuron to access its layer and consequently the
        network.
        """
        self.layer = layer
        if self.weights is True:
            layer_index = self.layer.neural.layers.index(self.layer)
            if layer_index == 0:
                self.weights = self.rand_weight()
            else:
                self.weights = [self.rand_weight() for _ in range(len(self.layer.neural.layers[layer_index - 1].neurons))]
        if self.bias is None and self.weights is None:
            self.mutate = self._try_mutate_another
        elif self.bias is None and not hasattr(self.weights, '__iter__'):
            self.mutate = self._mutate_weight
        elif self.bias is None and (hasattr(self.weights, '__iter__')):
            self.mutate = self._mutate_weights
        elif self.bias is not None and self.weights is not None and not hasattr(self.weights, '__iter__'):
            self.mutate = self._mutate_weight_and_bias
        elif self.bias is not None and self.weights is not None and hasattr(self.weights, '__iter__'):
            self.mutate = self._mutate_weights_and_bias

    def _get_parameters(self, function, second_step):
        if self.bias is not None and self.bias is not True and self.bias is not int and self.bias is not float:
            raise NameError('Bias\' type is not appropriate.')
        if self.bias is not None and self.rand_bias_range is None:
            raise NameError('Bias needs a int or float value to \'rand_bias_range\' variable to work.')
        if self.rand_weights_range is not None:
            self.rand_weight = lambda: random.uniform(-self.rand_weights_range, self.rand_weights_range)
        if self.rand_bias_range is not None:
            self.rand_bias = lambda: random.uniform(-self.rand_bias_range, self.rand_bias_range)
            if self.bias is True:
                self.bias = self.rand_bias()
        if not self.custom_function:
            self.function_name = function
            self._get_function(function)
        else:
            if type(function) is str:
                function = eval(function)
            self.function_name = function.__name__
            self.function = function
        if second_step is not None:
            if not self.custom_second_step:
                self.second_step = Neuron._get_second_step(second_step)
            else:
                if type(second_step) is str:
                    second_step = eval(second_step)
                self.second_step = second_step
            self.second_step_name = self.second_step.__name__
            if self.bias is None:
                self.evaluate = self._evaluate_with_second_step
            else:
                self.evaluate = self._evaluate_with_bias_with_second_step
        else:
            self.second_step_name = None
            if self.bias is None:
                self.evaluate = self._evaluate
            else:
                self.evaluate = self._evaluate_with_bias

    def _try_mutate_another(self):
        self.layer.neural.mutate()

    @property
    def data(self):
        """
        Returns all data that determine the neuron as a dictionary.
        """
        return {"function": self.function_name, "weights": self.weights, "bias": self.bias,
                "second_step": self.second_step_name, "rand_weights_range": self.rand_weights_range,
                "rand_bias_range": self.rand_bias_range, "custom_function": self.custom_function,
                "custom_second_step": self.custom_second_step}

    def _mutate_weights(self):
        self.weights[random.choice(range(len(self.weights)))] = self.rand_weight()

    def _mutate_weight(self):
        self.weight = self.rand_weight()

    def _mutate_bias(self):
        self.bias = self.rand_bias()

    def _mutate_weight_and_bias(self):
        self._mutate_weight()
        self._mutate_bias()

    def _mutate_weights_and_bias(self):
        self._mutate_weights()
        self._mutate_bias()

    @staticmethod
    def _get_second_step(second_step):
        if second_step == 'binary':
            function = Neuron.binary
        elif second_step == 'relu':
            function = Neuron.relu
        elif second_step == 'leaky relu':
            function = Neuron.leaky_relu
        else:
            raise NameError('Neuron\'s second step was not properly selected.')
        return function

    def _evaluate(self, _input):
        return self.function(_input)

    def _evaluate_with_bias(self, _input):
        return self.function(_input + self.bias)

    def _evaluate_with_bias_with_second_step(self, _input):
        self.second_step(self.function(_input + self.bias))

    def _evaluate_with_second_step(self, _input):
        return self.second_step(self.function(_input))

    @staticmethod
    def linear(_input):
        return _input

    @staticmethod
    def sigmoid(_input):
        return 1 / (1 + e ** (- _input))

    @staticmethod
    def binary(_input):
        return 1 if _input >= 0 else 0

    @staticmethod
    def tanh(_input):
        return 2 * Neuron.sigmoid(2 * _input) - 1

    @staticmethod
    def relu(_input):
        return max(0, _input)

    @staticmethod
    def leaky_relu(_input):
        return 0.01 * _input if _input < 0 else _input

    def _get_function(self, function):
        if function == 'linear':
            self.function = Neuron.linear
        elif function == 'sigmoid':
            self.function = Neuron.sigmoid
        elif function == 'binary':
            self.function = Neuron.binary
        elif function == 'tanh':
            self.function = Neuron.tanh
        elif function == 'relu':
            self.function = Neuron.relu
        elif function == 'leaky relu':
            self.function = Neuron.leaky_relu


class Network:
    """
    The object which contains layers of neurons by which it can passes an argument to evaluate it.

    Produces a neural network by receiving the number of inputs waited (must be an integer value), a iterable of layers
    and a boolean value, 'None' or a string for 'name' variable. The layers it receive in the 'layers' variable exclude
    the input layer, because this one is substituted by the integer value 'inputs' that means the number of input
    neurons it is supposed to have. If name receives 'True', the neural network will be identified by a random human
    name, if it receives 'False', will be identified by its __hash__ number, if it receives a string it will be
    identified by the string itself.

    The properties/methods of a neural network are
    name -> The tag that identifies it;
    layers -> Its iterable of layers (list, array, tuple, ...), but without the input neurons;
    inputs -> Its number of waited inputs, it means, the number of input neurons;
    sign -> The method by which the neural network assigns its layers as its own;
    data -> A dictionary with all the information needed to determine the neural network;
    hidden_neurons -> A list with the number of neurons in each of the hidden layers;
    outputs -> The number of output neurons in the network;
    hidden_layers -> The number of hidden layers in the network;
    write_data -> The method by which neuron's data can be saved in a .json document;
    evaluate -> The method by which the network receives a iterable of inputs and returns its evaluation.;
    mutate -> The method by which the neural network mutate by choosing a random layer and calling its mutate
    function.

    All these properties/methods can be seen by neuron.respective_property or neuron.respective_property() if its
    callable.

    Example of neural network:
    neural_network = Network(5, [Layer([...]), Layer([...]), Layer([...]), name='John')
    """

    def __init__(self, layers=None, name=None):
        self.name = None
        self.layers = layers
        self._get_name(name)
        self.sign()

    def sign(self):
        """
        Signs it's layers as its own.
        """
        for layer in self.layers:
            layer.sign(self)

    def _get_name(self, name):
        """
        Identifies the neural network with a name tag.
        """
        if name is True:
            self.name = names.get_full_name()
        elif name is False or name is None:
            self.name = str(self.__hash__())
        elif type(name) is str:
            self.name = name
        else:
            raise NameError('name has not an appropriate type.')
        return

    @property
    def inputs(self):
        return len(self.layers[0].neurons)

    @property
    def data(self):
        """
        Returns all data that determine the neural network as a dictionary.
        """
        return {self.name: [layer.data for layer in self.layers]}

    @property
    def hidden_neurons(self):
        """
        Returns the number of neurons in each of the hidden layers as a list.
        """
        return [len(layer.neurons) for layer in self.layers[0:-1]]

    @property
    def outputs(self):
        """
        Returns the number of output neurons in the network.
        """
        return len(self.layers[-1].neurons)

    @property
    def hidden_layers(self):
        """
        Returns the number of hidden layers in the network.
        """
        return len(self.layers) - 2

    def write_data(self, document=None, directory='data'):
        """
        Produces a .json document with the neuron's data. The document will be named as the 'document' argument received
        by the function (it must be a string), and it will be saved in the directory named by the 'directory' argument
        received (also a string) that is 'data' by default. If the folder name by directory does not existed it will be
        crated. If the document already exists, the information will be appended, otherwise, the document will be
        created.
        """
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
        """
        Receives a iterable of numeric inputs and returns the neural network's evaluation.
        """
        layer_output = []
        for n, neuron in enumerate(self.layers[0].neurons):
            layer_output.append(neuron.evaluate(inputs[n] * neuron.weights))
        inputs = layer_output
        for layer in self.layers[1:]:
            layer_output = []
            for neuron in layer.neurons:
                neuron_input = 0
                for n, _input in enumerate(inputs):
                    neuron_input += _input * neuron.weights[n]
                layer_output.append(neuron.evaluate(neuron_input))
            inputs = layer_output
        return inputs

    def mutate(self):
        """
        Provokes a mutation in the neural network changing one parameter (bias or weight) of a random neuron.
        """
        random.choice(self.layers).mutate()


def load_data(document, name=None, keep_name=False, directory='data'):
    """
    Reads a .json document and returns the neural networks saved in it. It will search for a document named as the
    'document' argument it received (must be a string) in the directory passed ('data' by default). If it is passed a
    name (string), it return only the neural network with such name, otherwise, it will return a list with all the
    neural networks contained in the document.
    """
    directory = directory + '/'
    document_address = directory + document + '.json'
    if not os.path.exists(directory):
        os.mkdir(directory)
        raise NameError(f"There's no '{document}' in '{directory}'.")
    if not os.path.isfile(document_address):
        raise NameError(f"There's no '{document}.json' in {directory}.")
    with open(document_address, 'r') as file:
        content = json.load(file)
        if name is not None:
            neural_layers = content[name]
            layers = _get_layers(neural_layers)
            if not keep_name:
                name = None
            return Network(layers, name)
        else:
            neural_names = list(content.keys())
            saved_content = []
            for n, neural in enumerate(content):
                layers = _get_layers(content[neural_names[n]])
                if not keep_name:
                    neural_name = None
                saved_content.append(Network(layers, neural_name))
            return saved_content


def _get_layers(neural_layers):
    layers = []
    for n, layer in enumerate(neural_layers):
        layers.append(Layer([]))
        for neuron in layer:
            function = neuron['function']
            weights = neuron['weights']
            bias = neuron['bias']
            second_step = neuron['second_step']
            rand_weights_range = neuron['rand_weights_range']
            rand_bias_range = neuron ['rand_bias_range']
            custom_function = neuron['custom_function']
            custom_second_step = neuron['custom_second_step']
            layers[n].neurons.append(Neuron(function, weights, bias, second_step, rand_weights_range,
                                            rand_bias_range, custom_function, custom_second_step))
    return layers


def random_homogeneous_network(neurons_in_layer, neurons_function=None, neurons_second_step=None, weight=True,
                               bias=False, rand_weight_range=10, rand_bias_range=None, custom_function=False,
                               custom_second_step=False, name=None):
    """
    Returns a neural networks in which all neurons have identical parameters, but weight and bias (if they are allowed)
    will be randomly generated for each individual neuron.

    It receives a iterable of integer numbers describing the number of neurons in each layer, like '[3, 4, 4, 5]', which
    mean it would have three input neurons in the input layer, four hidden neurons in the first hidden layer, four
    neurons in the second hidden layer and five neurons in the output layer. The input layer only means how many inputs
    the network waits, it doesn't do any evaluation on it. The 'neurons_function' variable must receive a string related
    to one of the built-in functions of the Neuron class, or the function itself if the custom function is enabled. The
    'weight' and 'bias' variables must receive 'True' or 'False' to enable or disable each one. By default 'weight' is
    defined as 'True' and bias as 'False'. The 'rand_range' variable determines the range where weight and bias will be
    generated and to where it will can be mutated. It represents the upper limit, and the lower limit the negative of
    the upper. The 'custom_function' and 'custom_second_step' variables enables or disable the use of custom functions
    in the neurons. Set it 'True' to use custom functions, 'False' to built-in ones. By default they are set as 'False',
    it behaviours the same way as if it is 'None'. When it is 'True' it makes the generated network has a random human
    name as name, and when it is a string it makes the generated neural network has the related string as a name.

    Example of use:
    random_network = random_homogeneous_neural(neurons_in_layer=[5, 6, 7, 5], neurons_function='sigmoid',
                                               neurons_second_step=None, weight=True, bias=True, rand_range=10,
                                               custom_function=False, custom_second_step=False, name=True)
    """
    bias = None if bias is False else bias
    layers = []
    for _layer in neurons_in_layer:
        layer = []
        for neuron in range(_layer):
            layer.append(Neuron(neurons_function, weight, bias, neurons_second_step, rand_weight_range,
                                rand_bias_range, custom_function, custom_second_step))
        layers.append(Layer(layer))
    return Network(layers, name)


def _name_crossover(parent_name, donor_name, name):
    parent_family_name = parent_name.split(' ')[1:]
    donor_family_name = donor_name.split(' ')[1:]
    if name and not parent_name.isnumeric() and not donor_name.isnumeric() and type(name) is not str and \
            len(parent_family_name) > 0 and len(donor_family_name) > 0:
        child_family_name = []
        for family_name in donor_family_name:
            if family_name in parent_family_name:
                continue
            child_family_name.append(family_name)
        child_family_name += parent_family_name
        name = names.get_first_name() + ' ' + ' '.join(child_family_name)
    elif name:
        raise NameError('Donor\'s and parent\'s names are not compatible.')
    return name


def layer_crossover(parent, donor, name=False):
    """
    Returns a new neural network by receiving two other networks, one in the 'parent' variable, other in 'donor'
    variable. The network generated will be  identical the parent but with one of its layers substituted by a copy of
    one layer randomly selected of the donor. If the 'name' variable is set 'False' or 'None', the neural network
    generated will has its '__hash__' as name. If name is set 'True' and the parent and donor have human names the
    network generated will inherit the parent's and donor's families names, and will also have a random first name. If
    the 'name' variable is a string the generated network will receive this string as name.
    """
    donor_layer_index = random.choice(list(range(len(donor.layers))))
    layers = deepcopy(parent.layers)
    layers[donor_layer_index] = deepcopy(donor.layers[donor_layer_index])
    name = _name_crossover(parent.name, donor.name, name)
    return Network(layers, name=name)


def n_crossover(parent, donor, name):
    children_layers = deepcopy(parent.layers)
    for n, neuron in enumerate(children_layers[0].neurons):
        neuron.weights = random.choice([neuron.weights, donor.layers[0].neurons[n].weights])
        neuron.bias = random.choice([neuron.bias, donor.layers[0].neurons[n].bias])
    for n, layer in enumerate(children_layers[1:]):
        for m, neuron in enumerate(layer.neurons):
            for o, weight in enumerate(neuron.weights):
                children_layers[n + 1].neurons[m].weights[o] = random.choice([weight, donor.layers[n + 1].neurons[m].weights[o]])
            neuron.bias = random.choice([neuron.bias, donor.layers[n + 1].neurons[m].bias])
    name = _name_crossover(parent.name, donor.name, name)
    return Network(children_layers, name)
