#!/usr/bin/env python

import os
import torch
import argparse
import numpy as np
import torch.optim as optim

from torch.autograd import Variable

from helpers.grapher import Grapher
from helpers.utils import softmax_accuracy
from datasets.loader import get_loader
from helpers.layers import EarlyStopping

parser = argparse.ArgumentParser(description='ML Baseline repo')


# General parameters
parser.add_argument('--seed', type=int, default=1,
                    help="seed for rng (default: 1)")
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer (default: adam)')

# Task parameters
parser.add_argument('--task', type=str, default='mnist',
                    help='dataset to work with (default: mnist)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')

# Device parameters
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# handle randomness / non-randomness
if args.seed is not None:
    print("setting seed %d" % args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed_all(args.seed)


def build_optimizer(model, args):
    ''' helper to build the optimizer '''
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }

    return optim_map[args.optimizer.lower().strip()](
        model.parameters(), lr=args.lr
    )


def train(epoch, model, optimizer, data_loader, args):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader.train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target.squeeze())

        # zero grads of optimizer
        optimizer.zero_grad()

        # project to the output dimension
        output, _ = model(data)
        loss = model.loss_function(output, target)

        # compute backward pass and optimize
        loss.backward()
        optimizer.step()

        # log every nth interval
        if batch_idx % args.log_interval == 0:
            num_samples = len(data_loader.train_loader.dataset)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_samples,
                100. * batch_idx * len(data) / num_samples,
                loss.data[0]))


def test_or_val(epoch, model, data_loader, args):
    model.eval()
    loss = []
    for data, target in data_loader.test_loader:
        if args.cuda:             # move to GPU
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            # variables are needed for computing gradients
            data, target = Variable(data), Variable(target.squeeze())

            # forward pass
            output = model(data)

            # compute model loss and evaluate accuracy
            # place into a list so that we can tabulate mean
            loss = model.loss_function(output, target)
            loss.append(loss.detach().cpu().data[0])

    # no need to do a running mean, just place in list
    loss = np.mean(loss)
    print('\nTest Epoch: {}\tAverage loss: {:.4f}\n'.format(
        epoch, loss
    ))
    return loss


def run(args):
    # get the dataloader
    loader = get_loader(args.task)

    # build the model and early stopper
    model = # build model
    early = EarlyStopping(model)   # get the early stopper

    # a visdom grapher object to enable matplotlib api
    grapher = Grapher(env=model.get_name(),
                      server=args.visdom_url,
                      port=args.visdom_port)

    for epoch in range(args.epochs + 1):
        train(epoch, model, loader.train, model.train_summary_writer, epoch)
        val_loss = test_or_val(epoch, model, loader.val, args)

        # handle early stopping
        if early(val_loss):
            early.restore()
            test_or_val(epoch, model, loader.test, args)
            break

        # evaluate test metrics
        test_loss, acc = test_or_val(epoch, model, loader.test, args)

        # use test loss / acc here.
        # ....

if __name__ == "__main__":
    # seed our RNGs
    np.random.seed(args.seed)
    torch.manual_seed_all(args.seed)

    # build the dirs for storage of logs and models
    create_dir(args.output_dir)
    create_dir(os.path.join(args.output_dir, "logs"))
    create_dir(os.path.join(args.output_dir, "models"))

    # main entrypoint
    run(args)
