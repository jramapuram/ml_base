import os
import time
import argparse
import pprint
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from torchvision import transforms

from models.vae.vrnn import VRNN
from models.vae.msg import MSGVAE
from models.vae.simple_vae import SimpleVAE
from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from models.vae.sequentially_reparameterized_vae import SequentiallyReparameterizedVAE

from datasets.loader import get_loader
from helpers.metrics import softmax_accuracy, bce_accuracy, \
    softmax_correct, all_or_none_accuracy, calculate_fid, calculate_mssim
from helpers.grapher import Grapher
from helpers.layers import EarlyStopping, append_save_and_load_fns
from helpers.utils import dummy_context, ones_like, get_name, \
    append_to_csv, check_or_create_dir
from helpers.fid import train_fid_model


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
parser.add_argument('--image-size-override', type=int, default=None,
                    help='Override and force resizing of images to this specific size (default: None)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--uid', type=str, default="",
                    help='uid for current session (default: empty-str)')

# VAE related
parser.add_argument('--vae-type', type=str, default='simple',
                    help='parallel, sequential, vrnn, simple, msg (default: simple)')
parser.add_argument('--nll-type', type=str, default='bernoulli',
                    help='bernoulli or gaussian (default: bernoulli)')
parser.add_argument('--reparam-type', type=str, default='isotropic_gaussian',
                    help='isotropic_gaussian, discrete or mixture [default: isotropic_gaussian]')
parser.add_argument('--continuous-size', type=int, default=32, metavar='L',
                    help='latent size of continuous variable when using mixture or gaussian (default: 32)')
parser.add_argument('--discrete-size', type=int, default=1,
                    help='initial dim of discrete variable when using mixture or gumbel (default: 1)')
parser.add_argument('--continuous-mut-info', type=float, default=0.0,
                    help='-continuous_mut_info * I(z_c; x) is applied (opposite dir of disc)(default: 0.0)')
parser.add_argument('--discrete-mut-info', type=float, default=0.0,
                    help='+discrete_mut_info * I(z_d; x) is applied (default: 0.0)')
parser.add_argument('--generative-scale-var', type=float, default=1.0,
                    help='scale variance of prior in order to capture outliers')
parser.add_argument('--use-aggregate-posterior', action='store_true', default=False,
                    help='uses aggregate posterior for generation (default: False)')

# Model
parser.add_argument('--encoder-layer-type', type=str, default='conv',
                    help='dense / resnet / conv (default: conv)')
parser.add_argument('--decoder-layer-type', type=str, default='conv',
                    help='dense / conv / coordconv (default: conv)')
parser.add_argument('--activation', type=str, default='elu',
                    help='default activation function (default: elu)')
parser.add_argument('--latent-size', type=int, default=512, metavar='N',
                    help='sizing for latent layers (default: 512)')
parser.add_argument('--filter-depth', type=int, default=32,
                    help='number of initial conv filter maps (default: 32)')
parser.add_argument('--disable-gated', action='store_true', default=False,
                    help='disables gated convolutional or dense structure (default: False)')
parser.add_argument('--model-dir', type=str, default='.models',
                    help='directory which contains saved models (default: .models)')

# RNN Related
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping for RNN (default: 0.25)')
parser.add_argument('--use-prior-kl', action='store_true',
                    help='add a kl on the VRNN prior against the true prior (default: False)')
parser.add_argument('--use-noisy-rnn-state', action='store_true',
                    help='uses a noisy initial rnn state instead of zeros (default: False)')
parser.add_argument('--max-time-steps', type=int, default=4,
                    help='max time steps for RNN (default: 4)')
parser.add_argument('--mut-clamp-strategy', type=str, default="clamp",
                    help='clamp mut info by norm / clamp / none (default: clamp)')
parser.add_argument('--mut-clamp-value', type=float, default=100.0,
                    help='max / min clamp value if above strategy is clamp (default: 100.0)')

# Regularizer
parser.add_argument('--kl-beta', type=float, default=1, help='beta-vae kl term (default: 1)')
parser.add_argument('--conv-normalization', type=str, default='groupnorm',
                    help='normalization type: batchnorm/groupnorm/instancenorm/none (default: groupnorm)')
parser.add_argument('--dense-normalization', type=str, default='batchnorm',
                    help='normalization type: batchnorm/instancenorm/none (default: batchnorm)')
parser.add_argument('--add-img-noise', action='store_true', default=False,
                    help='add scattered noise to  images (default: False)')

# Metrics
parser.add_argument('--calculate-msssim', action='store_true', default=False,
                    help='enables FID calc & uses model conv/inceptionv3  (default: None)')
parser.add_argument('--calculate-fid-with', type=str, default=None,
                    help='calculated the multi-scale structural similarity (default: False)')

# Optimization related
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help="specify optimizer (default: adam)")
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping (default: False)')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom URL for graphs, needs http://url (default: None)')
parser.add_argument('--visdom-port', type=int, default=None,
                    help='visdom port for graphs (default: None)')

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
    transform = [torchvision.transforms.Resize(resize_shape)] \
        if args.image_size_override else None
    loader = get_loader(args, transform=transform, **vars(args))  # build the loader
    args.input_shape = loader.img_shp if args.image_size_override is None \
        else [loader.img_shp[0], *resize_shape]                   # set the input size

    # build the network
    vae_dict = {
        'simple': SimpleVAE,
        'msg': MSGVAE,
        'parallel': ParallellyReparameterizedVAE,
        'sequential': SequentiallyReparameterizedVAE,
        'vrnn': VRNN
    }
    network = vae_dict[args.vae_type](loader.img_shp, kwargs=deepcopy(vars(args)))
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
    model.config['half'] = False # disable half here due to CPU weights
    for minibatch, labels in loader:
        with torch.no_grad():
            _ = model(minibatch)
            break

    # reset half tensors if requested since torch.cuda.HalfTensor has impls
    model.config['half'] = args.half



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

    # iterate over data and labels
    for minibatch, labels in loader:
        minibatch = minibatch.cuda() if args.cuda else minibatch
        labels = labels.cuda() if args.cuda else labels
        if args.half:
            minibatch = minibatch.half()

        if 'train' in prefix:
            optimizer.zero_grad()                                                # zero gradients on optimizer

        with torch.no_grad() if prefix == 'test' else dummy_context():
            pred_logits, reparam_map = model(minibatch)                          # get normal predictions
            loss_t = model.loss_function(pred_logits, minibatch, reparam_map)
            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += minibatch.size(0)

        if 'train' in prefix: # compute bp and optimize
            if args.half:
                optimizer.backward(loss_t['loss_mean'])
                # with amp_handle.scale_loss(loss_t['loss_mean'], optimizer,
                #                            dynamic_loss_scale=True) as scaled_loss:
                #     scaled_loss.backward()
            else:
                loss_t['loss_mean'].backward()

            if args.clip > 0:
                # TODO: clip by value or norm? torch.nn.utils.clip_grad_value_
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) \
                nn.utils.clip_grad_value_(model.parameters(), args.clip) \
                    if not args.half else optimizer.clip_master_grads(args.clip)

            optimizer.step()
            del loss_t

        if args.debug_step: # for testing purposes
            break

    # compute the mean of the map
    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    print('{}[Epoch {}][{} samples][{:.2f} sec]:\t Loss: {:.4f}\t-ELBO: {:.4f}\tNLL: {:.4f}\tKLD: {:.4f}\tMI: {:.4f}'.format(
        prefix, epoch, num_samples, time.time() - start_time,
        loss_map['loss_mean'].item(),
        loss_map['elbo_mean'].item(),
        loss_map['nll_mean'].item(),
        loss_map['kld_mean'].item(),
        loss_map['mut_info_mean'].item()))

    # activate the logits of the reconstruction and get the dict
    reconstr_map = model.get_activated_reconstructions(pred_logits)

    # tack on MSSIM information if requested
    if args.calculate_msssim:
        loss_map['ms_ssim_mean'] = compute_mssim(reconstr_image, minibatch)

    # gather scalar values of reparameterizers (if they exist)
    reparam_scalars = model.get_reparameterizer_scalars()

    # plot the test accuracy, loss and images
    register_plots({**loss_map, **reparam_scalars}, grapher, epoch=epoch, prefix=prefix)

    # get some generations, only do once in a while for pixelcnn
    generated = None
    if args.decoder_layer_type == 'pixelcnn' and epoch % 10 != 0:
        generated = model.generate_synthetic_samples(args.batch_size, reset_state=True,
                                                     use_aggregate_posterior=args.use_aggregate_posterior)

    # tack on images to grapher
    image_map = {
        'input_imgs': F.upsample(minibatch, (100, 100)) if args.task == 'image_folder' else minibatch
    }
    if generated:
        image_map['generated_imgs'] = F.upsample(generated, (100, 100)) if args.task == 'image_folder' else generated

    register_images({**image_map, **reconstr_map}, grapher, prefix=prefix)
    grapher.save()

    # cleanups (see https://tinyurl.com/ycjre67m) + return ELBO for early stopping
    loss_val = loss_map['elbo_mean'].detach().item()
    loss_map.clear(); image_map.clear(); reparam_map.clear(); reparam_scalars.clear()
    del minibatch; del labels
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
    fid_model = train_fid_model(args, fid_type=args.calculate_fid_with, batch_size=32) \
        if args.calculate_fid_with is not None else None        # the FID object

    # main training loop
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, loader.train_loader, grapher)
        test_loss = test(epoch, model, loader.test_loader, grapher)
        if args.early_stop and early(test_loss):
            early.restore() # restore and test+generate again
            test_loss = test(epoch, model, loader.test_loader, grapher)
            break

        if epoch == 2: # make sure we do at least 1 test and train pass
            grapher.add_text('config', pprint.PrettyPrinter(indent=4).pformat(vars(args)),0)#, append=True)

    # compute fid if requested
    if fid_model is not None:
        fid_score = calculate_fid(fid_model, model, loader,
                                  grapher, num_samples=1000,
                                  cuda=args.cuda)
        print("FID = ", fid_score)

    # cleanups
    grapher.close()


if __name__ == "__main__":
    print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))
    run(args)
