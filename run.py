__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
import numpy as np
from core.lstm import LSTM
from core.quantile_loss import QuantileLoss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from reshaped_data import reshape_data
from to_surpervised import to_surpervised
from sklearn.preprocessing import MinMaxScaler
import math

quantiles = [0.5, 0.9]
input_size = 20
hidden_size = 100
number_layer = 1
# output_layer = len(quantiles)
batch_size = 32
epoch = 50
input_layer = 48
output_layer = len(quantiles)
output_size = 24
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loss = []
validation_loss = []
loss_function = QuantileLoss(quantiles)


def plot_results(prediction_list, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i in range(len(prediction_list)):
        plt.plot(prediction_list[i], label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


# pytorch lstm train the model
def train(model, x, y, x_v, y_v, optimizer, batch_size, epoch):
    for e in range(1, epoch + 1):
        len_batch = math.ceil(x.size(0) / batch_size)
        losses = []
        for batch_idx in range(len_batch):
            # print(x[batch_idx])
            # output = model(x[batch_idx])
            if batch_size * (batch_idx + 1) > x.size(0):
                output = model(x[batch_idx * batch_size:])
                target = y[batch_idx * batch_size:]
            else:
                output = model(x[batch_idx * batch_size:(batch_idx + 1) * batch_size])
                target = y[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # output = output.view()
            # output = output.view(x.size()).to(device)
            loss = loss_function(output, target)
            losses.append(loss.item())
            # loss = loss_function(output, y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e,
                                                                        batch_idx * batch_size, x.size(0),
                                                                        100. * batch_idx / len_batch, loss.item()))
        train_loss.append(np.mean(losses))
        losses.clear()
        pred = validation(model, x_v)
        # y_v= torch.tensor(y_v, dtype=torch.float32).to(device)
        los = loss_function(pred, y_v)
        validation_loss.append(los)
        print('Epoch:{} train loss:{}, validation loss:{}'.format(e, train_loss[e - 1], validation_loss[e - 1]))
        # print('Epoch:{} train loss:{}'.format(e, train_loss[e-1]))

        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         e, batch_idx * len(x), len(x), 100. * batch_idx / len(x), loss.item()))


# pytorch lstm validate the result
def validation(model, x):
    # loss_function = torch.nn.MSELoss()
    # sum_loss = []
    # for batch_idx in range(len(x)):
    #     prediction = model(x)
    #     sum_loss.append(loss_function(y-prediction))
    # print('the test Loss is {:.6f}'.format(np.mean(sum_loss)))
    predicted = []
    # for batch_idx in range(len(x)):
    # x_ind = x[batch_idx]
    # p = model(x_ind).detach().numpy()
    p = model(x).view(-1, 1, output_layer)
    # predicted.append(p)

    # return np.reshape(np.asarray(predicted), (655,))
    return p


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # build the model
    # model = Model()
    # model.build_model(configs)
    model = LSTM(input_size, hidden_size, number_layer, output_size, output_layer).to(device)
    optimizor = optim.SGD(model.parameters(), lr=0.1, momentum=0.2)
    # base_value = data.data_train[0]
    # print(base_value)
    # x, y = data.get_train_data(
    #     seq_len=configs['data']['sequence_length'],
    #     normalise=configs['data']['normalise']
    # )
    # get the data
    scaler = MinMaxScaler()
    scaler.fit(data.data_train)
    # transform data inside the train or prediction function
    train_data = reshape_data(scaler.transform(data.data_train))
    x, y = to_surpervised(train_data, input_layer, output_size, 'train')
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    validation_data = reshape_data(scaler.transform(data.data_test))
    x_validation, y_validation = to_surpervised(validation_data, input_layer, output_size, 'validation')
    x_validation = torch.tensor(x_validation, dtype=torch.float32).to(device)
    y_validation = y_validation.flatten().reshape(-1, 1)
    y_validation = torch.tensor(y_validation, dtype=torch.float32).to(device)

    # '''	# in-memory training
    # history = model.train(
    #     x,
    #     y,
    #     x_validation,
    #     y_validation,
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     save_dir=configs['model']['save_dir']
    # )
    train(model, x, y, x_validation, y_validation, optimizor, batch_size=batch_size, epoch=epoch)
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # '''
    # out-of memory generative training
    # steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    # model.train_generator(
    #     data_gen=data.generate_train_batch(
    #         seq_len=configs['data']['sequence_length'],
    #         batch_size=configs['training']['batch_size'],
    #         normalise=configs['data']['normalise']
    #     ),
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     steps_per_epoch=steps_per_epoch,
    #     save_dir=configs['model']['save_dir']
    # )

    # x_test, y_test = data.get_test_data(
    #     seq_len=configs['data']['sequence_length'],
    #     normalise=configs['data']['normalise']
    # )
    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_validation, configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x_test)
    # predictions_origin = scaler.inverse_transform(predictions)
    predictions = validation(model, x_validation)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    prediction_list = []
    for i in range(output_layer):
        prediction_list.append(scaler.inverse_transform(predictions[:,:,i].cpu().detach().numpy()))
    # predictions = scaler.inverse_transform(predictions.cpu().detach().numpy())
    y_validation = scaler.inverse_transform(y_validation.cpu().detach().numpy())
    plot_results(prediction_list, y_validation)


if __name__ == '__main__':
    main()
