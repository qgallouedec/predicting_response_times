# coding:utf-8

"""Challenge data : predicting Paris Brigade Fire Response
Running this file will read the data of data/*.pickle. It will train
the model and save the reults on testset in submission.csv
"""

import torch  # basic tools

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn  # Module classe
import torch.nn.functional as F  # functionnals like relu...
import torch.optim as optim  # L'optimizer
from torch.distributions import transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pickle
import sklearn.metrics
import time
import math
import csv

import logging
logging.basicConfig(filename='logs/response_time.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(' '.join(("Device used :", str(device))))

# Path to save the model
PATH = './nets/net.pth'


def tensor_to_score(cpu_y, cpu_y_pred):
    return(sklearn.metrics.r2_score(
        cpu_y.detach().numpy(),
        cpu_y_pred.detach().numpy()))


def score_on_testset(net, validation_loader):
    test_scores = []

    for cpu_x, cpu_y in validation_loader:
        logging.debug("data loaded")

        # Set net mode to eval
        net.eval()
        logging.debug("network in train mode")

        # send inputs and output on gpu
        gpu_x, gpu_y = cpu_x.to(device), cpu_y.to(device)
        logging.debug("x and y sent on gpu")

        # compute output
        gpu_y_pred = net(gpu_x)
        logging.debug("output computed")

        # compute score
        cpu_y_pred = gpu_y_pred.cpu()
        test_score = tensor_to_score(cpu_y, cpu_y_pred)
        logging.debug("score computed")

        # store loss
        test_scores.append(test_score)

    return np.mean(test_scores)


def loss_on_testset(net, validation_loader, criterion):
    """ Test the model (on validation_set), return the mean loss"""

    test_losses = []

    for cpu_x, cpu_y in validation_loader:
        logging.debug("data loaded")

        # Set net mode to eval
        net.eval()
        logging.debug("network in train mode")

        # send inputs and output on gpu
        gpu_x, gpu_y = cpu_x.to(device), cpu_y.to(device)
        logging.debug("x and y sent on gpu")

        # compute output
        gpu_y_pred = net(gpu_x)
        logging.debug("output computed")

        # compute loss
        loss = criterion(gpu_y_pred, gpu_y)
        logging.debug("loss computed")

        # store loss
        test_losses.append(loss.item())

    return np.mean(test_losses)


class ResponseTimeDataset(Dataset):
    """Response Time dataset."""

    def __init__(self, train, slice_idx=12, nb_slices=40):
        super(ResponseTimeDataset, self).__init__()

        # load the np arrays
        x_train = pickle.load(open('data/x_train.pickle', 'rb'))
        x_test = pickle.load(open('data/x_test.pickle', 'rb'))
        y = pickle.load(open('data/y_train.pickle', 'rb'))

        x = np.concatenate((x_train, x_test))

        # choose witch part of data to use
        nb_data = x_train.shape[0]
        validation_set_start_idx = int(slice_idx/nb_slices*nb_data)
        validation_set_stop_idx = int((slice_idx+1)/nb_slices*nb_data)

        if not train:  # if validation set
            self.inputs = x_train[validation_set_start_idx:validation_set_stop_idx]
            self.outputs = y[validation_set_start_idx:validation_set_stop_idx]

        else:
            # take the right slice
            self.inputs = np.concatenate(
                (x_train[:validation_set_start_idx], x_train[validation_set_stop_idx:]))
            self.outputs = np.concatenate(
                (y[:validation_set_start_idx], y[validation_set_stop_idx:]))

        # mean reduce x
        self._mean_reduce_input(x)

    def _mean_reduce_input(self, x):
        mean = x.mean(axis=0)
        std = np.std(x, axis=0)
        # Some values are always 0. The std_dev is 0 then.
        # This line is used not to divide by 0
        std = np.array([1 if val == 0 else val for val in std])
        self.inputs = (self.inputs - mean) / std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float), torch.tensor(self.outputs[idx], dtype=torch.float)


class ResponseTimeDatasetTest(Dataset):
    """Response Time dataset."""

    def __init__(self):
        super(ResponseTimeDatasetTest, self).__init__()

        # load the np arrays
        x_train = pickle.load(open('data/x_train.pickle', 'rb'))
        x_test = pickle.load(open('data/x_test.pickle', 'rb'))

        x = np.concatenate((x_train, x_test))

        # take the right slice
        self.inputs = x_test

        # mean reduce x
        mean = x.mean(axis=0)
        std = np.std(x, axis=0)
        std = np.array([1 if val == 0 else val for val in std])
        self.inputs = (self.inputs - mean) / std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float)


class Net(nn.Module):
    """Neural network"""

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1964, 72)
        self.dropout1 = nn.Dropout(p=0.03)
        self.fc2 = nn.Linear(72, 3)

    def forward(self, x):
        # Fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


def train(net, train_loader, validation_loader, optimizer, scheduler, criterion):
    """ Train the model """
    for epoch in range(300):  # loop over the dataset multiple times
        logging.debug("starting epoch %d" % epoch)
        train_losses = []

        for cpu_x, cpu_y in train_loader:
            logging.debug("data loaded")

            net.train()
            logging.debug("network in train mode")

            # zero the optimizer gradient
            optimizer.zero_grad()
            logging.debug("grad zeroed")

            # send inputs and output on gpu
            gpu_x, gpu_y = cpu_x.to(device), cpu_y.to(device)
            logging.debug("x and y sent on gpu")

            # compute output
            gpu_y_pred = net(gpu_x)
            logging.debug("output computed")

            # compute loss
            loss = criterion(gpu_y_pred, gpu_y)
            logging.debug("loss computed")

            # backpropagate
            loss.backward(retain_graph=True)
            logging.debug("loss backwarded")

            # step optimizer
            optimizer.step()
            logging.debug("optimizer steped")

            # store loss
            train_losses.append(loss.item())

        # log some metrics about validation_set
        mean_train_loss = np.mean(train_losses)
        mean_test_loss = loss_on_testset(net, validation_loader, criterion)
        mean_test_score = score_on_testset(net, validation_loader)
        logging.info(
            "Epoch %d   train loss : %.2f  test loss : %.2f  test score : %.2f"
            % (epoch, mean_train_loss, mean_test_loss, mean_test_score))

        # save the model
        torch.save(net.state_dict(), f=PATH)
        logging.debug("model saved")

        # step the scheduler
        scheduler.step()

    logging.info('Finished training')


if __name__ == '__main__':

    # Data set and data loader
    train_set = ResponseTimeDataset(train=True)
    train_loader = DataLoader(train_set, batch_size=2048,
                              shuffle=True, num_workers=8)

    validation_set = ResponseTimeDataset(train=False)
    validation_loader = DataLoader(validation_set, batch_size=2048,
                                   shuffle=True, num_workers=8)

    # neural network
    net = Net()
    net.to(device)
    logging.info('Neural network : {}'.format(str(net)))

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=10e-4)

    # Scheduler : for learning rate decay
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    # criterion to compute loss
    criterion = nn.L1Loss()

    # training
    train(net, train_loader, validation_loader,
          optimizer, scheduler, criterion)

    # to load an existing model and create the submission
    net.load_state_dict(torch.load(PATH))

    test_set = ResponseTimeDatasetTest()
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    with open('./data/submission.csv', 'w') as write_file:
        with open('./data/x_test.csv', 'r') as csvfile:

            # write the first line
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            write_file.writelines(
                ('emergency vehicle selection,delta selection-departure,delta\
                    departure-presentation,delta selection-presentation\n'))
            next(csv_reader)

            # iterate over test data
            for cpu_x in test_loader:
                net.eval()
                gpu_x = cpu_x.to(device)
                gpu_y_pred = net(gpu_x)
                numpy_y_pred = gpu_y_pred.cpu().detach().numpy()[0]
                id_inter = next(csv_reader)[0]

                line = ','.join([str(val) for val in numpy_y_pred])
                line = id_inter + ',' + line + '\n'
                write_file.write(line)
