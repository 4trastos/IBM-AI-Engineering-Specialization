import numpy as np

weights = np.around(np.random.uniform(size=6), decimals=2) # inicializamos peso
biases = np.around(np.random.uniform(size=3), decimals=2) # inicializamos sesgo

print(weights)
print(biases)

x1 = 0.5       # input 1
x2 = -0.35      # input 2

print('X_1 is {} and X_2 is {}'.format(x1, x2))

z_11 = x1 * weights[0] + x2 * weights[1] + biases[0]        # realizamos la suma ponderada
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(np.around(z_11, decimals=4)))

##################################################
#################### PRACTICE ####################

z_12 = x2 * weights[2] + x2 * weights[3] + biases[1]
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))

a_11 = 1.0 / (1.0 + np.exp(-z_11))      # activación sigmoide del primer nodo
a_12 = 1.0 / (1.0 + np.exp(-z_12))      # activación sigmoide del segundo nodo

print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs at the third node in the hidden layer is {}'.format(np.around(z_2, decimals=4)))

a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))

##################################################
##################################################

