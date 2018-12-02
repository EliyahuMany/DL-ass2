import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

# Global vars
LR = 0.005
BATCH_SIZE = 2048
INPUT_SIZE = 250
WIN_SIZE = 5
EPOCHS = 3
EMBEDDING_SIZE = 50
HIDDEN_SIZE_1 = 150
HIDDEN_SIZE_2 = 50
SUFFIX = 3
PREFIX = 3

embeddings_needed = bool(int(sys.argv[2]))
if embeddings_needed:
    import tools2 as tools
else:
    import tools


class NeuralNet(nn.Module):
    def __init__(self, pre_trained, input_size):
        super(NeuralNet, self).__init__()
        self.input_size = WIN_SIZE * EMBEDDING_SIZE
        self.fc0 = nn.Linear(input_size, HIDDEN_SIZE_1)
        self.fc1 = nn.Linear(HIDDEN_SIZE_1, len(tools.TAGS))
        if not pre_trained:
            self.E = nn.Embedding(len(tools.WORDS), EMBEDDING_SIZE)
        else:
            self.E = nn.Embedding(tools.E.shape[0], tools.E.shape[1])
            self.E.weight.data.copy_(torch.from_numpy(tools.E))
        self.pref_index = {suff: i for i, suff in enumerate(self.prefs)}
        self.suff_index = {suff: i for i, suff in enumerate(self.suffs)}
        self.prefs = {word[:PREFIX] for word in tools.WORDS}
        self.suffs = {word[:-SUFFIX] for word in tools.WORDS}
        self.E_pref = nn.Embedding(len(self.prefs), EMBEDDING_SIZE)
        self.E_suff = nn.Embedding(len(self.suffs), EMBEDDING_SIZE)
        self.prefs = list(self.prefs)
        self.suffs = list(self.suffs)

    def forward(self, v):
        pref_var = v.data.numpy().copy()
        suff_var = v.data.numpy().copy()
        pref_var = pref_var.reshape(-1)
        suff_var = suff_var.reshape(-1)
        pref_var = [self.prefs[self.prefix_to_index[tools.I2W[index][:PREFIX]]]
                    for index in pref_var]
        suff_var = [self.suffs[self.suffix_to_index[tools.I2W[index][:-SUFFIX]]]
                    for index in suff_var]
        pref_var = [self.pref_index[pref] for pref in pref_var]
        suff_var = [self.suff_index[suff] for suff in suff_var]
        pref_var = np.asanyarray(pref_var)
        suff_var = np.asanyarray(suff_var)
        pref_var = torch.from_numpy(pref_var.reshape(x.data.shape)).type(torch.LongTensor)
        suff_var = torch.from_numpy(suff_var.reshape(x.data.shape)).type(torch.LongTensor)

        v = (self.E(v) + self.E_pref(pref_var) + self.E_suff(suff_var)).view(-1, self.input_size)
        v = F.tanh(self.fc0(v))
        v = self.fc1(v)
        return F.log_softmax(v, dim=1)


class Trainer(object):
    def __init__(self, train, valid, test, model, optimizer):
        self.valid = valid
        self.train = train
        self.optimizer = optimizer
        self.model = model
        self.test = test

    def run(self, type):
        avg_train_loss = {}
        avg_valid_loss = {}
        valid_accuracy = {}
        for epoch in range(1, EPOCHS + 1):
            print(str(epoch))
            self.train(epoch, avg_train_loss)
            self.valid(epoch, avg_valid_loss,
                       valid_accuracy, type)
            plot_graphs(avg_valid_loss, valid_accuracy)
            self.test(type)

    def test(self, type):
        self.model.eval()
        prediction_list = []
        for data in self.test:
            output = self.model(torch.LongTensor(data))
            # get the predicted class out of output tensor
            prediction = output.data.max(1, keepdim=True)[1]
            # add current prediction to predictions list
            prediction_list.append(prediction.item())

        prediction_list = [tools.I2T[index] for index in prediction_list]
        self.write_results(type + "/test", "test1." + type, prediction_list)

    def write_results(self, test_path, output_path, predictions_list):
        with open(test_path, 'r') as test_file, open(output_path, 'w') as output:
            content = test_file.readlines()
            i = 0
            for line in content:
                if line is tools.NEW_LINE:
                    output.write(line)
                else:
                    output.write(line.strip(tools.NEW_LINE) + " " + predictions_list[i] + tools.NEW_LINE)
                    i += 1

    def train(self, epoch, avg_train_loss):
        self.model.train()
        train_loss = 0
        correct = 0

        for data, labels in self.train:
            self.optimizer.zero_grad()
            output = self.model(data)
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(labels.data.view_as(prediction)).cpu().sum().item()
            loss = F.nll_loss(output, labels)
            train_loss += loss
            loss.backward()
            self.optimizer.step()

        train_loss /= (len(self.train))
        avg_train_loss[epoch] = train_loss
        print("Epoch: {} Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.00f}%)"
              .format(epoch, train_loss, correct, len(self.train) * BATCH_SIZE,
                      100. * correct / (len(self.train) * BATCH_SIZE)))

    def validation(self, epoch_num, avg_valid_loss, valid_accuracy, type):
        total = 0
        correct = 0
        self.model.eval()
        valid_loss = 0
        for data, target in self.valid:
            output = self.model(data)
            valid_loss += F.nll_loss(output, target, size_average=False).item()
            prediction = output.data.max(1, keepdim=True)[1]
            if type is not 'ner':
                total += 1
                correct += prediction.eq(target.data.view_as(prediction)).cpu().sum().item()
            else:
                if tools.I2T[prediction.cpu().sum().item()] != 'O' or tools.I2T[target.cpu().sum().item()] != 'O':
                    correct += prediction.eq(target.data.view_as(prediction)).cpu().sum().item()
                    total += 1
        accuracy = 100. * correct / total
        valid_accuracy[epoch_num] = accuracy
        valid_loss /= len(self.valid)
        avg_valid_loss[epoch_num] = valid_loss
        print('\n Epoch:{} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_num, valid_loss, correct, total,
            accuracy))


def plot_graphs(avg_valid_loss, valid_accuracy):
    line1, = plt.plot(avg_valid_loss.keys(), avg_valid_loss.values(), "purple",
                      label='Validation average loss')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()
    plt.savefig('avg_loss.png')
    line2, = plt.plot(valid_accuracy.keys(), valid_accuracy.values(),
                      label='Validation average accuracy')
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=4)})
    plt.show()
    plt.savefig('accuracy.png')


def load_data(file_name, dev):
    x, y = tools.get_tagged(file_name, dev)
    x, y = np.asarray(x, np.float32), np.asarray(y, np.int32)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.type(torch.LongTensor), y.type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(x, y)
    if dev:
        load = torch.utils.data.DataLoader(dataset, 1, shuffle=True)
    else:
        load = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
    return load


def main(args):
    global LR
    type = args[0]
    if type is 'ner':
        LR = 0.01
    train = load_data(type + '/train', False)
    valid = load_data(type + '/dev', True)
    test = tools.not_tagged(type + '/test')
    model = NeuralNet(embeddings_needed)
    optimizer = optim.Adam(model.parameters(), LR)

    trainer = Trainer(train, valid, test, model, optimizer)
    trainer.run(type)


if __name__ == "__main__":
    main(sys.argv[1:])
