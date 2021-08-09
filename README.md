# Neural
## _Easy fully customisable neural networks with python_
[![N|sbb](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.redd.it%2Fcv563jzwax831.jpg&f=1&nofb=1)](https://github.com/estevaopbs/)

An easy way to 
- implement entirely customisable neural networks;
- produce mutations on they;
- save and load the networks with .json files.

## How does it work?
We have three classes we will use to build our neural network, they are:
- Neuron: A set of weights and functions which represents the minimal unity of calculations of the network;
- Layer: A set of neurons;
- Network: A set of neurons' layers.

We create a neural network by providing its layers of neurons. Each Neuron has a number of weights correspondent to the number of bondings it has with the previous layer. So, as an example, let's imagine a neural network with three layers, the first one, with four neurons, the second with five, the third with two.

The first layer, also known as input layer, doesn't has any neuron before it because it is, as the name suggests, the layer that receive the inputs, what means that we have only four inputs in this hypotetical situation. So in this layer, each neuron has only one weight value which will be applied to the correspondent input. The input will be multiplied by weight of the correspondent neuron, then if the neuron has a bias it will be added, the result will be evaluated by the neuron's function, after this, if the neuron has a second step function the result will be evaluated by it, if not so it doesn't, and the result will be passed to the next layer.

Now, let's take a look to a neuron located in the second layer. The first layer has
four neurons, so this neuron will receive four values to evaluate. Each one of the received values will be multiplied by the neuron's correspondent bonding weight. It means this neuron has one weight which connect him with one neuron of the previous layer. In this context, these weights are analogous to the axons. Then the results will be summed, in sequence if the neuron in question has a bias it will be added and the result will be evaluated by the function, after this, if the neuron has a second step function the result will be evaluated by it, if not so it doesn't, and the result will be passed to the next layer.

Finnaly in the last layer. Here everything will happen the same way it happened in the previous layer, the only difference is that is our last neurons' layer, it can also be named as output layer, consequently the results obtained here will not be passed to other neurons, the two results obained here from the last two neurons will be the final result of the network's evaluation.


## How to use this thing?
To build a neural network we need to give it layers of neurons, and a name that will be the tag by which it  will be identified. So we can create a Network object this way:
```py
neural_network = Neural.Network(layers, name)
```
**layers**: Iiterable of Layers (list, tuple, set, array, ...).

**name**: If name receives 'True', the neural network will be identified by a random human name, if it receives 'False' or none it will be identified by its \_\_hash\_\_ number and if it receives a string it will be identified by the string itself.

To get the layers we will use
```py
layer = Layer(neurons)
```
**neurons**: Iterable of Neurons (list, tuple, set, array, ...).

By the end, to get the neuron we need to provide some parameters this way:
```py
neuron = Neural.Neuron(function, weights, bias, second_step, rand_weights_range, rand_bias_range, 
                       custom_function, custom_second_step)
```
**function**: Its main function. It must receive a string correspondent to one of the built-in functions described in the functions section or the properly function if custom_function is True;

**weights**: Its multiplicative factors for each input. When it's 'None', it will never be changed by a mutate call, When it's True, an adequate number of random weights will be generated by the _sign method when the neuron be inserted in a network. It can als receive a list of ints or floats or receive a single int or float to have its values pre-set;

**bias**: Its additive factor. When it's 'None', it will work as zero in the built-in functions, and it will never be changed by a mutate call. When it's True, a random bias will be generated. It can also receives a int or float tohave its value pre-set;

**second\_step**: Its second step function. It must receive a string correspondent to one of the built-in functions described in the functions section or the properly function if custom\_second\_step is True;

**rand\_weights\_range**: The upper limit of the neuron's random weight generator. The lower is its negative. It's 10 by default;

**rand\_bias\_range**: The upper limit of the neuron's random weight generator. The lower is its negative. It's None bydefault. When it's None, no bias will be added to the inputs and no bias will be able to be generated in a mutate call;

**custom\_function**: Boolean value that enables (True) or disable (False) the use of a alternative function;

**custom\_second\_step**: Boolean value that enables (True) or disable (False) the use of a alternative second step function.

#### Functions
The built-in functions are:
- linear: returns weight * input + bias;
- sigmoid: returns 1 / (1 + e ^ - (weight * input + bias));
- tanh: returns tanh(weight * input + bias);
- binary: returns 1 if the input is greater or equal 0 otherwise returns 0;
- relu: returns maximum value between zero and the input;
- leaky relu: returns (0.01 * input) if the input is less than zero otherwise returns the input.

Using the linear function as an example, to use one of the built-in functions you must do this way when creating a neuron
```py
neuron = Neural.Neuron(function='linear', ..., custom_function=False, ...)
```
But if you want to use other function than one of these, there are two ways to do this. You can define a function and use it as input or use a lambda function, as you can see in the example below.

Using a defined function:

```py
def square_function(x):
    return x ** 2


neuron = Neural.Neuron(function=square_function, ..., custom_function=True, ...)
```

Using a lambda function:

```py
neuron = Neural.Neuron(function=lambda x: x ** 2, ..., custom_function=True, ...)

# or

square_function = lambda x: x ** 2
neuron = Neural.Neuron(function=square_function, ..., custom_function=True, ...)
```

The second step function, works the exact same way. Using a defined function:

```py
def square_function(x):
    return x ** 2


neuron = Neural.Neuron(..., second_step=square_function, 
                       ..., custom_second_step=True)
```

Using a lambda function:

```py
neuron = Neural.Neuron(..., second_step=lambda x: x ** 2, ..., custom_second_step=True)

# or

square_function = lambda x: x ** 2
neuron = Neural.Neuron(..., second_step=square_function, ...,custom_second_step=True)
```
## Properties
### Network
**name**: The tag that identifies it;

**layers**: Its iterable of layers (list, array, tuple, ...), but without the input neurons;

**inputs**: Its number of waited inputs, it means, the number of input neurons;

**neurons**: The list of all neurons

**shape**:  Its number of neurons in each layer;

**data**:A dictionary with all the information needed to determine the neural network;

**hidden_neurons**: A list with the number of neurons in each of the hidden layers;

**outputs**: The number of output neurons in the network;

**hidden_layers**: The number of hidden layers in the network;

### Layer
**neurons**: Its iterable of neurons (list, array, tuple, ...);

**network**: The neural network which the layer belongs;

**data**: The list which contains all the data needed to determine the layer;

### Neuron
**function**: Its main function;

**function_name**: The string which name the main function;

**weights**: Its multiplicative factors for each input;

**bias**: Its additive factor;

**second_step**: Its second step function;

**second_step_name**: The string which name the second step function;

**custom_function**: Boolean value that enables (True) or disable (False) the use of a alternative function;

**custom_second_step**: Boolean value that enables (True) or disable (False) the use of a alternative second step 
function;

**rand_weights_range**: The upper limit of the neuron's random weight generator. The lower is its negative. It's 10 by
default;

**rand_bias_range**: The upper limit of the neuron's random weight generator. The lower is its negative. It's None by default. When it's None, no bias will be added to the inputs and no bias will be able to be generated in a mutatecall;

**evaluate**: The method which the neuron receives a input and outputs its evaluation;

**rand_weight**: The method the neuron uses to generate random weights;

**rand_bias**: The method the neuron uses to generate random biases;

**layer**: The layer which the neuron belongs;

**data**: A dictionary with all the information needed to determine the neuron;

All these properties/methods can be obtained by \<object>.\<property>.

## Methods

### Network
**save_data**: This method produces a .json document with the neuron's data. It must be called this way:
```py
neural_network.save_data(self, document, directory)
```
The document will be named as the 'document' argument received
by the function (it must be a string), and it will be saved in the directory named by the 'directory' argument
received (also a string) that is 'data' by default. If the folder name by directory does not existed it will be
crated. If the document already exists, the information will be appended, otherwise, the document will be
created.

**evaluate**: The method by which the network receives a iterable of inputs and returns its evaluation;

**mutate**: The method by which the network mutates by choosing a random layer and calling its mutate method;

**uniform_mutate**: The method by which the network mutates by a random neuron and calling its mutate method;

### Layer
**mutate**: The method by which the layer can suffer a mutation by choosing randomly one of its neurons and calling its mutate method;

### Neuron
**mutate**: Mutate method proper of the neuron that can be called by neuron.mutate() causing it to mutate;

### \_\_main\_\_
**load_data**: This method reads a .json document and returns the neural networks saved in it. It must be called this way:
```py
neural_network = load_data(document, name, keep_name, directory)
```
It will search for a document named as the 'document' argument it received (must be a string) in the directory passed ('data' by default). If it is passed a name (string), it return only the neural network with such name, otherwise, it will return a list with all the neural networks contained in the document.

**random_homogeneous_network**: Returns a neural networks in which all neurons have identical parameters, but weight and bias (if they are allowed) will be randomly generated for each individual neuron. It must me called this way:
```py
neural_network = random_homogeneous_network(neurons_in_layer, neurons_function, neurons_second_step,
                                            weights, bias, rand_weight_range, rand_bias_range, 
                                            custom_function, custom_second_step, name)
```
It receives a iterable of integer numbers describing the number of neurons in each layer, like '[3, 4, 4, 5]', which
mean it would have three input neurons in the input layer, four hidden neurons in the first hidden layer, four
neurons in the second hidden layer and five neurons in the output layer. The 'neurons_function' variable must
receive a string related to one of the built-in functions of the Neuron class, or the function itself if the custom
function is enabled. The 'weights' and 'bias' variables must receive 'True' or 'False' to enable or disable each
one. By default 'weights' is defined as 'True' and bias as 'False'. The 'rand_weight_range' variable determines the
range where weight will be generated and to where it will can be mutated, by default it is 10. It represents the
upper limit, and the lower limit the negative of the upper. Similarly 'rand_bias_range' does for the bias, but by
the default it is None.  The 'custom_function' and 'custom_second_step' variables enables or disable the use of
custom functions in the neurons. Set it 'True' to use custom functions, 'False' to built-in ones. By default they
are set as 'False', it behaviours the same way as if it is 'None'. When Name is 'True' it makes the generated
network has a random human name as name, when it is a string it makes the generated neural network has the related
string as a name, and when  it is None, the network receives its \_\_hash\_\_ as name.


**layer_crossover**: Returns a new neural network by mixing the layers of two other networks. It must be called like
```py
child_network = layer_crossover(parent, donor, name)
```
The network generated will be  identical the parent but with one of its layers randomly substituted by a
copy of the correspondent layer of the donor. If the 'name' variable is set 'False' or 'None', the neural network
generated will has its '\_\_hash__' as name. If name is set 'True' and the parent and donor have human names the
network generated will inherit the parent's and donor's families names, and will also have a random first name. If
the 'name' variable is a string the generated network will receive this string as name.

**n_crossover**: Returns a new neural network by mixing the neurons of two other networks, one in the 'parent' variable, other in 'donor' variable. It must be called like 
```py
child_network = n_crossover(parent, donor, name)
```
The network generated will be  identical the parent but with one of its neurons randomly substituted by a
copy of the correspondent neuron of the donor. If the 'name' variable is set 'False' or 'None', the neural network
generated will has its '\_\_hash\_\_' as name. If name is set 'True' and the parent and donor have human names the
network generated will inherit the parent's and donor's families names, and will also have a random first name. If
the 'name' variable is a string the generated network will receive this string as name.