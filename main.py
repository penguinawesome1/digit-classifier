'''
Purpose: A digit classifier made in Python using a neural network.

Owen Colley
9/3/24
'''

def main():
    inputs = [1.2, 5.1, 2.1]
    weights = [3.1, 2.1, 8.7]
    bias = 3

    output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
    print(output)