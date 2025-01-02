import numpy as np

####### DEFINIMOS LA ESTRUCTURA DE LA RED ######

n = 2                   # imputs
num_hidenn_layers = 2   # capas ocultas
m = [2,2]               # nodos en cada capa oculta
num_nodes_output = 1    # nodos de salida

###### INICIAMOS PESOS Y SESGOS #####

num_nodes_previous = n  # numero de nodos en la capa previa
network = {}    # red vacia

###### RECORREMOS CADA CAPA E INICIALMOS ##########
###### PESOS Y SESGOS ALEATORIAMENTE ##############

for layer in range(num_hidenn_layers + 1):
    if layer == num_hidenn_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output
    else:
        layer_name = 'layer_{}'.format(layer + 1)
        num_nodes = m[layer]
    
    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = 'node_{}'.format(node + 1)
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }
    
    nodos_prev = num_nodes_previous

#print(network)  # print network

#############################################
############## CREAMOS FUNCIÓN ##############
############ PARA INICIAR UNA RED ###########

def initialize_network(num_inputs, num_hidenn_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs

    network = {}
    for layer in range(num_hidenn_layers + 1):
        if layer == num_hidenn_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2)
            }
        num_nodes_previous = num_nodes

    return network

######################################
############## PRACTICE ##############

small_network = initialize_network(5, 3, [3, 2, 3], 1)

#############################################
############## CREAMOS FUNCIÓN ##############
####### PARA HACER LA SUMA PONDERADA ########

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

from random import seed
np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

print('The inputs yo the network are {}'.format(inputs))

node_weight = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']

weight_sum = compute_weighted_sum(inputs, node_weight, node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weight_sum[0]), decimals=4))

#############################################
############## CREAMOS FUNCIÓN ##############
####### SIGMOIDEA PARA LA ACTIVACIÓN ########

def node_activation(weight_sum):
    return 1.0 / (1.0 + np.exp(-1 * weight_sum))

node_output = node_activation(compute_weighted_sum(inputs, node_weight, node_bias))
print ('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0]), decimals=4))
node_output = node_activation(weight_sum)
print ('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0]), decimals=4))


#############################################
############ FORWARD PROPAGATION ############

def forward_propagate(network, inputs):
    layer_inputs = list(inputs) # comience con la capa de entrada como entrada a la primera capa oculta

    for layer in network:
        layer_data = network[layer]
        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
        
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
        
        layer_inputs = layer_outputs
    
    network_predictions = layer_outputs
    return network_predictions

predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))

my_network = initialize_network(5, 3, [2, 3, 2], 3)
inputs = np.around(np.random.uniform(size=5), decimals=2)
predictions = forward_propagate(my_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))