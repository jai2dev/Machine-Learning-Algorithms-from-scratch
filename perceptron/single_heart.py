from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

class Perceptron:
    def __init__(self,dataset):
        self.dataset=dataset


    def load_csv(self,filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    
    def str_column_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    def dataset_minmax(self, dataset):
        minmax = list()
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        return stats

    def normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def backward_propagate_error(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    def train_network(self, network, train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            for row in train:
                outputs = self.forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                self.backward_propagate_error(network, expected)
                self.update_weights(network, row, l_rate)

    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network

    def predict(self, network, row):
        outputs = self.forward_propagate(network, row)
        return outputs.index(max(outputs))

    def back_propagation(self, train, test, l_rate, n_epoch, n_hidden):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        # n_outputs=1
        network = self.initialize_network(n_inputs, n_hidden, n_outputs)
        self.train_network(network, train, l_rate, n_epoch, n_outputs)
        predictions = list()
        for row in test:
            prediction = self.predict(network, row)
            predictions.append(prediction)
        return(predictions)


def single_perceptron(filename):

    filename = filename
    perceptron=Perceptron(filename)

    dataset = perceptron.load_csv(filename)

    dataset=dataset[1:]
    # print(dataset)

    for i in range(len(dataset[0])-1):
        perceptron.str_column_to_float(dataset, i)

    
    perceptron.str_column_to_int(dataset, len(dataset[0])-1)
    
    
    minmax = perceptron.dataset_minmax(dataset)
    perceptron.normalize_dataset(dataset, minmax)
    
    n_folds = 15
    l_rate = 0.9
    n_epoch = 1000
    #no of hidden layers as single perceptron so no hidden layer
    n_hidden = 0
    scores = perceptron.evaluate_algorithm(dataset, perceptron.back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    scores=[round(i,3) for i in scores]
    print('Scores over k folds: %s' % scores)
    print('Mean Accuracy over k folds: %.3f%%' % (sum(scores)/float(len(scores))))


if __name__=='__main__':
    seed(1)
    single_perceptron('SPECTF.csv')