# Neural

## Table of Contents
- [Neural](#neural)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Network Class](#network-class)
      - [Attributes](#attributes)
      - [Methods](#methods)
    - [Layer Class](#layer-class)
      - [Attributes](#attributes-1)
      - [Methods](#methods-1)
    - [Neuron Class](#neuron-class)
      - [Attributes](#attributes-2)
      - [Methods](#methods-2)
    - [Crossover Functions](#crossover-functions)
        - [Layer Crossover](#layer-crossover)
      - [Neuron Crossover](#neuron-crossover)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)
  - [Licensing](#licensing)

## Introduction
The Neural project provides classes for building neural networks, including Network, Layer, and Neuron classes. It also includes crossover functions for evolving networks.

## Features
- Flexible and customizable neural network architecture
- Support for various activation functions
- Crossover functions for evolving networks
- Enum classes for activation, random weight generation, and derivative functions

## Requirements
- Python 3.x

## Installation
Clone the repository:
```bash
git clone https://github.com/estevaopbs/Neural.git
```

## Usage
### Network Class
The Network class represents a neural network. It consists of layers, and each layer contains neurons.

#### Attributes
- `layers`: List of Layer objects.

#### Methods
- `evaluate(inputs: List[Number]) -> List[Number]`: Evaluates the network with the given inputs.
- `enforce_weights():` Enforces consistent weights across layers.
- `mutate_weight()`: Mutates a random weight in the network.
- `mutate_bias()`: Mutates a random bias in the network.
- `mutate_layer_weights()`: Mutates weights in a random layer.
- `mutate_layer_biases()`: Mutates biases in a random layer.
- `mutate_weights()`: Mutates weights in all layers.
- `mutate_biases()`: Mutates biases in all layers.
- `to_json(filename: Path)`: Saves the network to a JSON file.
- `from_json(filename: Path)`: Loads the network from a JSON file.
- `to_pickle(filename: Path)`: Saves the network to a pickle file.
- `from_pickle(filename: Path)`: Loads the network from a pickle file.

### Layer Class
The Layer class represents a layer in a neural network.

#### Attributes
- neurons: List of Neuron objects.

#### Methods
- `evaluate(inputs: List[Number]) -> List[Number]`: Evaluates the layer with the given inputs.
- `mutate_weight()`: Mutates a random weight in the layer.
- `mutate_bias()`: Mutates a random bias in the layer.
- `mutate_weights(neurons: int)`: Mutates weights in a specified number of neurons.
- `mutate_biases(neurons: int)`: Mutates biases in a specified number of neurons.
- `enforce_weights(weights: int)`: Enforces a consistent number of weights across neurons.
- `from_data(data, NeuronClass: Type[Neuron] = Neuron) -> Layer`: Creates a layer from data.
- `from_random(...) -> Layer`: Creates a layer with random weights and biases.

### Neuron Class
The Neuron class represents a single neuron in a neural network.

#### Attributes
- `activation`: Activation function.
- `weights`: List of weights.
- `bias`: Bias value.
- `second_step`: Secondary activation function.
- `rand_weights_generator`: Random weight generator function.
- `rand_bias_generator`: Random bias generator function.
- `derivative`: Derivative function.
- `evaluate`: Internal evaluation function.

#### Methods
- `evaluate(inputs: List[Number]) -> Number`: Evaluates the neuron with the given inputs.
- `mutate_weights()`: Mutates a random weight in the neuron.
- `mutate_bias()`: Mutates the bias of the neuron.
- `mutate_weights_and_bias()`: Mutates both weights and bias.
- `mutate_weights_or_bias()`: Mutates either weights or bias.
- `from_random(...) -> Neuron`: Creates a neuron with random parameters.
- `enforce_weights(weights: int)`: Enforces a consistent number of weights.


### Crossover Functions
##### Layer Crossover
```python
layer_crossover(parent: Network, donor: Network, layers: int) -> Network
```
Performs crossover at the layer level between a parent and a donor network.

#### Neuron Crossover
```python
neuron_crossover(parent: Network, donor: Network, neurons: int) -> Network

```
Performs crossover at the neuron level between a parent and a donor network.

## Roadmap
- Implement backpropagation for training networks.
- Implement unit tests for code coverage.

## Contributing
Contributions are welcome! Feel free to open issues or pull requests.

## Licensing
This project is licensed under the MIT License.