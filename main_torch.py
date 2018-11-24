# encoding: utf-8
from util import data_preprocess
import models
import torch
import torch.optim as optim
import torch.nn.functional as F
batch_size = 64

num_samples = 25000


def train(epoch, x_train, y_train, model, optimizer):
    num_batchs = num_samples // batch_size
    model.train()
    # model.hidden = model.init_hidden()
    for k in range(num_batchs):
        # hidden = repackage_hidden(hidden)
        start, end = k * batch_size, (k + 1) * batch_size
        data = torch.Tensor(x_train[start:end]).long()
        target = torch.Tensor(y_train[start:end]).long()
        optimizer.zero_grad()
        output = model(data)
        # print("output :",output)
        # print("target:",target)
        loss = F.nll_loss(output, target)  # F.binary_cross_entropy(output,target) # F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if k % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, k * len(data), num_samples,
                       100. * k / num_samples, loss.data[0]))


def main():
    x_train, y_train, x_test, y_test = data_preprocess.word_vectors()
    # print(x_train.shape)
    # print(y_train.shape)

    model = getattr(models, 'LSTMNet')()
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train(2, x_train, y_train, model, optimizer)


if __name__ == '__main__':
    main()

