#!/usr/bin/env python

import os
import time
import torch
import pprint
import argparse
import torchvision
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models.resnet import resnet18

from helpers.grapher import Grapher
from datasets.loader import get_loader
from helpers.layers import EarlyStopping, append_save_and_load_fns
from helpers.utils import dummy_context, get_name, \
    append_to_csv, check_or_create_dir
from helpers.metrics import softmax_accuracy, bce_accuracy, \
    softmax_correct, all_or_none_accuracy, calculate_fid, calculate_mssim

parser = argparse.ArgumentParser(description='')


# Task parameters
parser.add_argument('--task', type=str, default="fashion",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='minimum number of epochs to train (default: 10000)')
parser.add_argument('--download', type=int, default=1,
                    help='download dataset from s3 (default: 1)')
parser.add_argument('--image-size-override', type=int, default=224,
                    help='Override and force resizing of images to this specific size (default: 224)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--uid', type=str, default="",
                    help='uid for current session (default: empty-str)')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom URL for graphs, needs http://url (default: None)')
parser.add_argument('--visdom-port', type=int, default=None,
                    help='visdom port for graphs (default: None)')

# Optimization related
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help="specify optimizer (default: adam)")
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping (default: False)')

# Device /debug stuff
parser.add_argument('--debug-step', action='store_true', default=False,
                    help='only does one step of the execute_graph function per call instead of all minibatches')
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--half', action='store_true', default=False,
                    help='enables half precision training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.backends.cudnn.benchmark = True

# add aws job ID to config if it exists
aws_instance_id = get_aws_instance_id()
if aws_instance_id is not None:
    args.instance_id = aws_instance_id

# set a fixed seed for GPUs and CPU
if args.seed is not None:
    print("setting seed %d" % args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


def build_optimizer(model):
    """ helper to build the optimizer and wrap model

    :param model: the model to wrap
    :returns: optimizer wrapping model provided
    :rtype: nn.Optim

    """
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


def build_loader_model_grapher(args):
    """builds a model, a dataloader and a grapher

    :param args: argparse
    :param transform: the dataloader transform
    :returns: a dataloader, a grapher and a model
    :rtype: list

    """
    resize_shape = (args.image_size_override, args.image_size_override)
    transform = [transforms.Resize(resize_shape)] \
        if args.image_size_override else None
    loader = get_loader(args, transform=transform, **vars(args))  # build the loader
    args.input_shape = loader.img_shp if args.image_size_override is None \
        else [loader.img_shp[0], *resize_shape]                   # set the input size

    # build the network; to use your own model import and construct it here
    network = resnet18(num_classes=loader.output_size)
    lazy_generate_modules(network, loader.train_loader)
    network = network.cuda() if args.cuda else network
    network = append_save_and_load_fns(network, prefix="VAE_")
    if args.ngpu > 1:
        print("data-paralleling...")
        network.parallel()

    # build the grapher object
    if args.visdom_url:
        grapher = Grapher('visdom', env=get_name(args),
                          server=args.visdom_url,
                          port=args.visdom_port)
    else:
        grapher = Grapher('tensorboard', comment=get_name(args))

    return loader, network, grapher


def lazy_generate_modules(model, loader):
    """ A helper to build the modules that are lazily compiled

    :param model: the nn.Module
    :param loader: the dataloader
    :returns: None
    :rtype: None

    """
    model.eval()
    for minibatch, labels in loader:
        with torch.no_grad():
            _ = model(minibatch)
            break


def register_plots(loss, grapher, epoch, prefix='train'):
    """ Registers line plots with grapher.

    :param loss: the dict containing '*_mean' or '*_scalar' values
    :param grapher: the grapher object
    :param epoch: the current epoch
    :param prefix: prefix to append to the plot
    :returns: None
    :rtype: None

    """
    for k, v in loss.items():
        if isinstance(v, dict):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k or 'scalar' in k:
            key_name = '-'.join(k.split('_')[0:-1])
            value = v.item() if not isinstance(v, (float, np.float32, np.float64)) else v
            grapher.add_scalar('{}_{}'.format(prefix, key_name), value, epoch)


def register_images(output_map, grapher, prefix='train'):
    """ Registers image with grapher. Overwrites the existing image due to space.

    :param output_map: the dict containing '*_img' of '*_imgs' as keys
    :param grapher: the grapher object
    :param prefix: prefix to attach to images
    :returns: None
    :rtype: None

    """
    for k, v in output_map.items():
        if isinstance(v, dict):
            register_images(output_map[k], grapher, epoch, prefix=prefix)

        if 'img' in k or 'imgs' in k:
            key_name = '-'.join(k.split('_')[0:-1])
            img = torchvision.utils.make_grid(v, normalize=True, scale_each=True)
            grapher.add_image('{}_{}'.format(prefix, key_name),
                              img.detach(),
                              global_step=0) # dont use step

def _add_loss_map(loss_tm1, loss_t):
    """ Adds the current dict _t to the previous running dict _tm1

    :param loss_tm1: a dict of previous losses
    :param loss_t: a dict of the current losses
    :returns: a new dict with added values and updated count
    :rtype: dict

    """
    if not loss_tm1: # base case: empty dict
        resultant = {'count': 1}
        for k, v in loss_t.items():
            if 'mean' in k or 'scalar' in k:
                if not isinstance(v, (float, int, np.float32, np.float64)):
                    resultant[k] = v.detach()
                else:
                    resultant[k] = v

        return resultant

    resultant = {}
    for (k, v) in loss_t.items():
        if 'mean' in k or 'scalar' in k:
            if not isinstance(v, (float, np.float32, np.float64)):
                resultant[k] = loss_tm1[k] + v.detach()
            else:
                resultant[k] = loss_tm1[k] + v

    # increment total count
    resultant['count'] = loss_tm1['count'] + 1
    return resultant


def _mean_map(loss_map):
    """ Simply scales all values in the dict by the count

    :param loss_map: the dict of scalars
    :returns: mean of the dict
    :rtype: dict

    """
    for k in loss_map.keys():
        if k == 'count':
            continue

        loss_map[k] /= loss_map['count']

    return loss_map


def execute_graph(epoch, model, loader, grapher, optimizer=None, prefix='test'):
    """ execute the graph; when 'train' is in the name the model runs the optimizer

    :param epoch: the current epoch number
    :param model: the torch model
    :param loader: the train or **TEST** loader
    :param grapher: the graph writing helper (eg: visdom / tf wrapper)
    :param optimizer: the optimizer
    :param prefix: 'train', 'test' or 'valid'
    :returns: dictionary with scalars
    :rtype: dict

    """
    start_time = time.time()
    model.eval() if prefix == 'test' else model.train()
    assert optimizer is not None if 'train' in prefix or 'valid' in prefix else optimizer is None
    loss_map, num_samples, print_once = {}, 0, False

    # iterate over train and valid data
    for minibatch, labels in loader:
        minibatch = minibatch.cuda() if args.cuda else minibatch
        labels = labels.cuda() if args.cuda else labels
        if args.half:
            minibatch = minibatch.half()

        if 'train' in prefix:
            optimizer.zero_grad()                                                # zero gradients on optimizer

        with torch.no_grad() if prefix == 'test' else dummy_context():
            pred_logits = model(minibatch)                                       # get normal predictions
            loss_t = {
                'loss_mean': F.cross_entropy(input=pred_logits, target=labels),  # change to F.mse_loss for regression
                'accuracy_mean': softmax_accuracy(preds=F.softmax(pred_logits, -1), targets=labels)
            }
            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += minibatch.size(0)

        if 'train' in prefix: # compute bp and optimize
            loss_t['loss_mean'].backward()
            optimizer.step()

        if args.debug_step: # for testing purposes
            break

    # compute the mean of the map
    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    print('{}[Epoch {}][{} samples][{:.2f} sec]: Loss: {:.4f}\tAccuracy: {:.4f}'.format(
        prefix, epoch, num_samples, time.time() - start_time,
        loss_map['loss_mean'].item(),
        loss_map['accuracy_mean'].item() * 100.0))

    # plot the test accuracy, loss and images
    register_plots({**loss_map}, grapher, epoch=epoch, prefix='linear'+prefix)
    register_images({'input_imgs': F.upsample(minibatch, size=(100, 100))}, grapher, prefix=prefix)

    # return this for early stopping
    loss_val = loss_map['loss_mean'].detach().item()
    loss_map.clear()
    return loss_val


def train(epoch, model, optimizer, train_loader, grapher, prefix='train'):
    """ Helper to run execute-graph for the train dataset

    :param epoch: the current epoch
    :param model: the model
    :param test_loader: the train data-loader
    :param grapher: the grapher object
    :param prefix: the default prefix; useful if we have multiple training types
    :returns: mean ELBO scalar
    :rtype: float32

    """
    return execute_graph(epoch, model, train_loader, grapher, optimizer, prefix='train')


def test(epoch, model, test_loader, grapher, prefix='test'):
    """ Helper to run execute-graph for the test dataset

    :param epoch: the current epoch
    :param model: the model
    :param test_loader: the test data-loader
    :param grapher: the grapher object
    :param prefix: the default prefix; useful if we have multiple test types
    :returns: mean ELBO scalar
    :rtype: float32

    """
    return execute_graph(epoch, model, test_loader, grapher, prefix='test')


def run(args):
    """ Main entry-point into the program

    :param args: argparse
    :returns: None
    :rtype: None

    """
    loader, model, grapher = build_loader_model_grapher(args)   # build the model, loader and grapher
    optimizer = build_optimizer(model)                          # the optimizer for the vae
    early = EarlyStopping(model, max_steps=200,                 # the early-stopping object
                          burn_in_interval=int(args.epochs*0.2)) if args.early_stop else None

    # main training loop
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, loader.train_loader, grapher)
        test_loss = test(epoch, model, loader.test_loader, grapher)
        if args.early_stop and early(test_loss):
            early.restore() # restore and test+generate again
            test_loss = test(epoch, model, loader.test_loader, grapher)
            break

        if epoch == 2: # make sure we do at least 1 test and train pass
            grapher.add_text('config', pprint.PrettyPrinter(indent=4).pformat(vars(args)),0)# , append=True)

    # cleanups
    grapher.close()


if __name__ == "__main__":
    print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))
    run(args)
