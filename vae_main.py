import os
import time
import tree
import argparse
import functools
import pprint
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np

from copy import deepcopy
from torchvision import transforms

import helpers.metrics as metrics
import helpers.layers as layers
import helpers.utils as utils
import optimizers.scheduler as scheduler

from models.vae import build_vae
from datasets.loader import get_loader
from helpers.grapher import Grapher
from helpers.fid_client import FIDClient
from optimizers.lars import LARS


parser = argparse.ArgumentParser(description='')

# Task parameters
parser.add_argument('--task', type=str, default="fashion",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='minimum number of epochs to train (default: 1000)')
parser.add_argument('--download', type=int, default=1,
                    help='download dataset from s3 (default: 1)')
parser.add_argument('--image-size-override', type=int, default=None,
                    help='Override and force resizing of images to this specific size (default: None)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--log-dir', type=str, default='./runs',
                    help='directory to store logs to (default: ./runs)')
parser.add_argument('--uid', type=str, default="",
                    help='uid for current session (default: empty-str)')

# VAE related
parser.add_argument('--vae-type', type=str, default='simple',
                    help='parallel, sequential, vrnn, simple, msg (default: simple)')
parser.add_argument('--monte-carlo-posterior-samples', type=int, default=1,
                    help='number of monte carlo samples to use from posterior (default: 1)')
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
parser.add_argument('--aggregate-posterior-ema-decay', type=float, default=0.9,
                    help='decay for the EMA based aggregate posterior (default: 0.9)')

# Model
parser.add_argument('--jit', action='store_true', default=False,
                    help='torch-script the model (default: False)')
parser.add_argument('--encoder-layer-type', type=str, default='conv',
                    help='dense / resnet / conv (default: conv)')
parser.add_argument('--encoder-layer-modifier', type=str, default='none',
                    help='none / spectral_norm / gated (default: none)')
parser.add_argument('--encoder-activation', type=str, default='elu',
                    help='default activation function (default: elu)')
parser.add_argument('--decoder-layer-type', type=str, default='conv',
                    help='dense / conv / coordconv (default: conv)')
parser.add_argument('--decoder-layer-modifier', type=str, default='none',
                    help='none / spectral_norm / gated (default: none)')
parser.add_argument('--decoder-activation', type=str, default='elu',
                    help='default activation function (default: elu)')
parser.add_argument('--weight-initialization', type=str, default=None,
                    help='weight initialization type; None uses default pytorch init. (default: None)')
parser.add_argument('--latent-size', type=int, default=512, metavar='N',
                    help='sizing for latent layers (default: 512)')
parser.add_argument('--encoder-base-channels', type=int, default=32,
                    help='number of initial conv filter maps (default: 32)')
parser.add_argument('--encoder-channel-multiplier', type=int, default=2,
                    help='grow channels by this per layer (default: 2)')
parser.add_argument('--decoder-base-channels', type=int, default=256,
                    help='number of initial conv filter maps (default: 1024)')
parser.add_argument('--decoder-channel-multiplier', type=float, default=0.5,
                    help='shrinks channels by this per layer (default: 0.5)')
parser.add_argument('--model-dir', type=str, default='.models',
                    help='directory which contains saved models (default: .models)')

# RNN Related
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping for RNN (default: 0.25)')
parser.add_argument('--use-prior-kl', action='store_true', default=False,
                    help='add a kl on the VRNN prior against the true prior (default: False)')
parser.add_argument('--use-noisy-rnn-state', action='store_true', default=False,
                    help='uses a noisy initial rnn state instead of zeros (default: False)')
parser.add_argument('--max-time-steps', type=int, default=0,
                    help='max time steps for RNN or MSGVAE (default: 0)')
parser.add_argument('--mut-clamp-strategy', type=str, default="clamp",
                    help='clamp mut info by norm / clamp / none (default: clamp)')
parser.add_argument('--mut-clamp-value', type=float, default=100.0,
                    help='max / min clamp value if above strategy is clamp (default: 100.0)')

# Regularizer
parser.add_argument('--kl-annealing-cycles', type=int, default=None, help='cycles for kl-annealing (default: None)')
parser.add_argument('--kl-beta', type=float, default=1, help='beta-vae kl term (default: 1)')
parser.add_argument('--weight-decay', type=float, default=0, help='weight decay (default: 0)')
parser.add_argument('--polyak-ema', type=float, default=0, help='Polyak weight averaging co-ef (default: 0)')
parser.add_argument('--conv-normalization', type=str, default='groupnorm',
                    help='normalization type: batchnorm/groupnorm/instancenorm/none (default: groupnorm)')
parser.add_argument('--dense-normalization', type=str, default='batchnorm',
                    help='normalization type: batchnorm/instancenorm/none (default: batchnorm)')
parser.add_argument('--add-img-noise', action='store_true', default=False,
                    help='add scattered noise to  images (default: False)')

# Metrics
parser.add_argument('--calculate-msssim', action='store_true', default=False,
                    help='enables MS-SSIM (default: False)')
parser.add_argument('--fid-server', type=str, default=None,
                    help='fid server url with port;  eg: myhost:8000 (default: None)')

# Optimization related
parser.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--lr-update-schedule', type=str, default='fixed',
                    help='learning rate schedule fixed/step/cosine (default: fixed)')
parser.add_argument('--warmup', type=int, default=3,
                    help='warmup epochs (default: 0)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help="specify optimizer (default: adam)")
parser.add_argument('--early-stop', action='store_true', default=False,
                    help='enable early stopping (default: False)')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom URL for graphs, needs http://url (default: None)')
parser.add_argument('--visdom-port', type=int, default=None,
                    help='visdom port for graphs (default: None)')

# Device /debug stuff
parser.add_argument('--num-replicas', type=int, default=1,
                    help='number of compute devices available; 1 means just local (default: 1)')
parser.add_argument('--workers-per-replica', type=int, default=2,
                    help='threads per replica for the data loader (default: 2)')
parser.add_argument('--distributed-master', type=str, default='127.0.0.1',
                    help='hostname or IP to use for distributed master (default: 127.0.0.1)')
parser.add_argument('--distributed-rank', type=int, default=0,
                    help='rank of the current replica in the world (default: 0)')
parser.add_argument('--distributed-port', type=int, default=29300,
                    help='port to use for distributed framework (default: 29300)')
parser.add_argument('--debug-step', action='store_true', default=False,
                    help='only does one step of the execute_graph function per call instead of all minibatches')
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--half', action='store_true', default=False,
                    help='enables half precision training')

args = parser.parse_args()

# import half-precision imports
if args.half:
    from apex.fp16_utils import *
    from apex import amp, optimizers


# add aws job ID to config if it exists
aws_instance_id = utils.get_aws_instance_id()
if aws_instance_id is not None:
    args.instance_id = aws_instance_id


def build_lr_schedule(optimizer, last_epoch=-1):
    """ adds a lr scheduler to the optimizer.

    :param optimizer: nn.Optimizer
    :returns: scheduler
    :rtype: optim.lr_scheduler

    """
    if args.lr_update_schedule == 'fixed':
        sched = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0, last_epoch=last_epoch)
    elif args.lr_update_schedule == 'cosine':
        total_epochs = args.epochs - args.warmup
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, last_epoch=last_epoch)
    else:
        raise NotImplementedError("lr scheduler {} not implemented".format(args.lr_update_schedule))

    # If warmup was requested add it.
    if args.warmup > 0:
        warmup = scheduler.LinearWarmup(optimizer, warmup_steps=args.warmup, last_epoch=last_epoch)
        sched = scheduler.Scheduler(sched, warmup)

    return sched


def build_optimizer(model, last_epoch=-1):
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
        "momentum": functools.partial(optim.SGD, momentum=0.9),
        "lbfgs": optim.LBFGS,
    }

    # Add weight decay (if > 0) and extract the optimizer string
    params_to_optimize = layers.add_weight_decay(model, args.weight_decay)
    full_opt_name = args.optimizer.lower().strip()
    is_lars = 'lars' in full_opt_name
    if full_opt_name == 'lamb':  # Lazy add this.
        assert args.half, "Need fp16 precision to use Apex FusedLAMB."
        optim_map['lamb'] = optimizers.fused_lamb.FusedLAMB

    opt_name = full_opt_name.split('_')[-1] if is_lars else full_opt_name
    print("using {} optimizer {} lars.".format(opt_name, 'with'if is_lars else 'without'))

    # Compute the LR and update according to batch size
    lr = args.lr
    if opt_name in ["momentum", "sgd"]:
        lr = args.lr * (args.batch_size * args.num_replicas / 256)

    # build the actual optimizer
    opt = optim_map[opt_name](params_to_optimize, lr=lr)

    # Wrap it with LARS if requested
    if is_lars:
        opt = LARS(opt, eps=0.0)

    # Build the schedule and return
    sched = build_lr_schedule(opt, last_epoch=last_epoch)
    return opt, sched


def build_train_and_test_transforms():
    """Returns torchvision OR nvidia-dali transforms.

    :returns: train_transforms, test_transforms
    :rtype: list, list

    """
    resize_shape = (args.image_size_override, args.image_size_override)

    if 'dali' in args.task:
        # Lazy import DALI dependencies because debug cpu nodes might not have DALI.
        import nvidia.dali.ops as ops
        import nvidia.dali.types as types

        train_transform, test_transform = None, None
        if args.image_size_override:
            train_transform = [
                ops.Resize(resize_x=resize_shape[0],
                           resize_y=resize_shape[1],
                           device="gpu" if args.cuda else "cpu",
                           image_type=types.RGB,
                           interp_type=types.INTERP_LINEAR)
            ]
            test_transform = [
                ops.Resize(resize_x=resize_shape[0],
                           resize_y=resize_shape[1],
                           device="gpu" if args.cuda else "cpu",
                           image_type=types.RGB,
                           interp_type=types.INTERP_LINEAR)
            ]
    else:
        resize_shape = (args.image_size_override, args.image_size_override)
        resize_xform = transforms.Resize(resize_shape) if args.image_size_override \
            else transforms.Lambda(lambda x: x)
        train_transform = [resize_xform]
        test_transform = [resize_xform]

    return train_transform, test_transform


def build_loader_model_grapher(args):
    """builds a model, a dataloader and a grapher

    :param args: argparse
    :param transform: the dataloader transform
    :returns: a dataloader, a grapher and a model
    :rtype: list

    """
    train_transform, test_transform = build_train_and_test_transforms()
    loader_dict = {'train_transform': train_transform,
                   'test_transform': test_transform, **vars(args)}
    loader = get_loader(**loader_dict)

    # set the input tensor shape (ignoring batch dimension) and related dataset sizing
    args.input_shape = loader.input_shape
    args.output_size = loader.output_size
    args.num_train_samples = loader.num_train_samples // args.num_replicas
    args.num_test_samples = loader.num_test_samples  # Test isn't currently split across devices
    args.num_valid_samples = loader.num_valid_samples // args.num_replicas
    args.steps_per_train_epoch = args.num_train_samples // args.batch_size  # drop-remainder
    args.total_train_steps = args.epochs * args.steps_per_train_epoch

    # build the network
    network = build_vae(args.vae_type)(loader.input_shape, kwargs=deepcopy(vars(args)))
    network = network.cuda() if args.cuda else network
    lazy_generate_modules(network, loader.train_loader)
    network = layers.init_weights(network, init=args.weight_initialization)

    if args.num_replicas > 1:
        print("wrapping model with DDP...")
        network = layers.DistributedDataParallelPassthrough(network,
                                                            device_ids=[0],   # set w/cuda environ var
                                                            output_device=0,  # set w/cuda environ var
                                                            find_unused_parameters=True)

    # Get some info about the structure and number of params.
    print(network)
    print("model has {} million parameters.".format(
        utils.number_of_parameters(network) / 1e6
    ))

    # build the grapher object
    grapher = None
    if args.visdom_url is not None and args.distributed_rank == 0:
        grapher = Grapher('visdom', env=utils.get_name(args),
                          server=args.visdom_url,
                          port=args.visdom_port,
                          log_folder=args.log_dir)
    elif args.distributed_rank == 0:
        grapher = Grapher(
            'tensorboard', logdir=os.path.join(args.log_dir, utils.get_name(args)))

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
            # Some sanity prints on the minibatch and labels
            print("minibatch = {} / {} | labels = {} / {}".format(minibatch.shape,
                                                                  minibatch.dtype,
                                                                  labels.shape,
                                                                  labels.dtype))
            mb_min, mb_max = minibatch.min(), minibatch.max()
            print("minibatch in range [min: {}, max: {}]".format(mb_min, mb_max))
            if mb_max > 1.0 or mb_min < 0:
                raise ValueError("Minibatch max > 1.0 or minibatch min < 0. You probably dont want this.")

            minibatch = minibatch.cuda(non_blocking=True) if args.cuda else minibatch
            labels = labels.cuda(non_blocking=True) if args.cuda else labels
            _ = model(minibatch, labels=labels)
            break

    # initialize the polyak-ema op if it exists
    if args.polyak_ema > 0:
        layers.polyak_ema_parameters(model, args.polyak_ema)


def register_plots(loss, grapher, epoch, prefix='train'):
    """ Registers line plots with grapher.

    :param loss: the dict containing '*_mean' or '*_scalar' values
    :param grapher: the grapher object
    :param epoch: the current epoch
    :param prefix: prefix to append to the plot
    :returns: None
    :rtype: None

    """
    if args.distributed_rank == 0 and grapher is not None:  # Only send stuff to visdom once.
        for k, v in loss.items():
            if isinstance(v, dict):
                register_plots(loss[k], grapher, epoch, prefix=prefix)

            if 'mean' in k or 'scalar' in k:
                key_name = '-'.join(k.split('_')[0:-1])
                value = v.item() if not isinstance(v, (float, np.float32, np.float64)) else v
                grapher.add_scalar('{}_{}'.format(prefix, key_name), value, epoch)


def register_images(output_map, grapher, epoch, prefix='train'):
    """ Registers image with grapher. Overwrites the existing image due to space.

    :param output_map: the dict containing '*_img' of '*_imgs' as keys
    :param grapher: the grapher object
    :param epoch: the current epoch
    :param prefix: prefix to attach to images
    :returns: None
    :rtype: None

    """
    if args.distributed_rank == 0 and grapher is not None:  # Only send stuff to visdom once.
        for k, v in output_map.items():
            if isinstance(v, dict):
                register_images(output_map[k], grapher, epoch=epoch, prefix=prefix)

            if 'img' in k or 'imgs' in k:
                key_name = '-'.join(k.split('_')[0:-1])
                img = torchvision.utils.make_grid(v, normalize=True, scale_each=True)

                # specify the keyword-args to the image plotter
                kwargs = {'global_step': epoch}
                if args.visdom_url is not None:
                    kwargs['store_history'] = 'reconstruction' in key_name or 'generated' in key_name

                grapher.add_image('{}_{}'.format(prefix, key_name),
                                  img.detach().cpu(), **kwargs)


def _extract_sum_scalars(v1, v2):
    """Simple helper to sum values in a struct using dm_tree."""

    def chk(c):
        """Helper to check if we have a primitive or tensor"""
        return not isinstance(c, (int, float, np.int32, np.int64, np.float32, np.float64))

    v1_detached = v1.detach() if chk(v1) else v1
    v2_detached = v2.detach() if chk(v2) else v2
    return v1_detached + v2_detached


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
    is_eval = 'train' not in prefix
    model.eval() if is_eval else model.train()
    assert optimizer is None if is_eval else optimizer is not None
    loss_map, num_samples = {}, 0

    # iterate over data and labels
    for num_minibatches, (minibatch, labels) in enumerate(loader):
        minibatch = minibatch.cuda(non_blocking=True) if args.cuda else minibatch
        labels = labels.cuda(non_blocking=True) if args.cuda else labels

        with torch.no_grad() if is_eval else utils.dummy_context():
            if is_eval and args.polyak_ema > 0:                                # use the Polyak model for predictions
                pred_logits, reparam_map = layers.get_polyak_prediction(
                    model, pred_fn=functools.partial(model, minibatch, labels=labels))
            else:
                pred_logits, reparam_map = model(minibatch, labels=labels)     # get normal predictions

            # compute loss
            loss_t = model.loss_function(recon_x=pred_logits,
                                         x=minibatch,
                                         params=reparam_map,
                                         K=args.monte_carlo_posterior_samples)
            loss_map = loss_t if not loss_map else tree.map_structure(         # aggregate loss
                _extract_sum_scalars, loss_map, loss_t)
            num_samples += minibatch.size(0)                                   # count minibatch samples

        if not is_eval:                                                        # compute bp and optimize
            optimizer.zero_grad()                                              # zero gradients on optimizer
            if args.half:
                with amp.scale_loss(loss_t['loss_mean'], optimizer) as scaled_loss:
                    scaled_loss.backward()                                     # compute grads (fp16+fp32)
            else:
                loss_t['loss_mean'].backward()                                 # compute grads (fp32)

            if args.clip > 0:
                # TODO: clip by value or norm? torch.nn.utils.clip_grad_value_
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                nn.utils.clip_grad_value_(model.parameters(), args.clip)

            optimizer.step()                                                   # update the parameters
            if args.polyak_ema > 0:                                            # update Polyak EMA if requested
                layers.polyak_ema_parameters(model, args.polyak_ema)

            del loss_t

        if args.debug_step and num_minibatches > 1:  # for testing purposes
            break

    # compute the mean of the dict
    loss_map = tree.map_structure(
        lambda v: v / (num_minibatches + 1), loss_map)                          # reduce the map to get actual means

    # log some stuff
    def tensor2item(t): return t.detach().item() if isinstance(t, torch.Tensor) else t
    to_log = '{}-{}[Epoch {}][{} samples][{:.2f} sec]:\t Loss: {:.4f}\t-ELBO: {:.4f}\tNLL: {:.4f}\tKLD: {:.4f}\tMI: {:.4f}'
    print(to_log.format(
        prefix, args.distributed_rank, epoch, num_samples, time.time() - start_time,
        tensor2item(loss_map['loss_mean']),
        tensor2item(loss_map['elbo_mean']),
        tensor2item(loss_map['nll_mean']),
        tensor2item(loss_map['kld_mean']),
        tensor2item(loss_map['mut_info_mean'])))

    # activate the logits of the reconstruction and get the dict
    reconstr_map = model.get_activated_reconstructions(pred_logits)

    # tack on MSSIM information if requested
    if args.calculate_msssim:
        loss_map['ms_ssim_mean'] = metrics.compute_mssim(
            minibatch, reconstr_map['reconstruction_imgs'])

    # gather scalar values of reparameterizers (if they exist)
    reparam_scalars = model.get_reparameterizer_scalars()

    # plot the test accuracy, loss and images
    register_plots({**loss_map, **reparam_scalars}, grapher, epoch=epoch, prefix=prefix)

    # tack on images to grapher
    image_map = {'input_imgs': minibatch}

    # Add generations to our image dict
    with torch.no_grad():
        prior_generated = model.generate_synthetic_samples(args.batch_size, reset_state=True,
                                                           use_aggregate_posterior=False)
        ema_generated = model.generate_synthetic_samples(args.batch_size, reset_state=True,
                                                         use_aggregate_posterior=True)
        image_map['prior_generated_imgs'] = prior_generated
        image_map['ema_generated_imgs'] = ema_generated

    register_images({**image_map, **reconstr_map}, grapher, epoch=epoch, prefix=prefix)
    if grapher is not None:
        grapher.save()

    # cleanups (see https://tinyurl.com/ycjre67m) + return ELBO for early stopping
    loss_val = tensor2item(loss_map['elbo_mean']) if args.vae_type != 'autoencoder' \
        else tensor2item(loss_map['nll_mean'])
    for d in [loss_map, image_map, reparam_map, reparam_scalars]:
        d.clear()

    del minibatch
    del labels

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
    :param test_loader: the test data-loaderpp
    :param grapher: the grapher object
    :param prefix: the default prefix; useful if we have multiple test types
    :returns: mean ELBO scalar
    :rtype: float32

    """
    return execute_graph(epoch, model, test_loader, grapher, prefix='test')


def init_multiprocessing_and_cuda(rank, args_from_spawn):
    """Sets the appropriate flags for multi-process jobs."""
    if args_from_spawn.multi_gpu_distributed:
        # Force set the GPU device in the case where a single node has >1 GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        args_from_spawn.distributed_rank = rank

    # Set the cuda flag appropriately
    args_from_spawn.cuda = not args_from_spawn.no_cuda and torch.cuda.is_available()
    if args_from_spawn.cuda:
        torch.backends.cudnn.benchmark = True
        print("Replica {} / {} using GPU: {}".format(
            rank + 1, args_from_spawn.num_replicas, torch.cuda.get_device_name(0)))

    # set a fixed seed for GPUs and CPU
    if args_from_spawn.seed is not None:
        print("setting seed %d" % args_from_spawn.seed)
        np.random.seed(args_from_spawn.seed)
        torch.manual_seed(args_from_spawn.seed)
        if args_from_spawn.cuda:
            torch.cuda.manual_seed_all(args_from_spawn.seed)

    if args_from_spawn.num_replicas > 1:
        torch.distributed.init_process_group(
            backend='nccl', init_method=os.environ['MASTER_ADDR'],
            world_size=args_from_spawn.num_replicas, rank=rank
        )
        print("Successfully created DDP process group!")

        # Update batch size appropriately
        args_from_spawn.batch_size = args_from_spawn.batch_size // args_from_spawn.num_replicas

    # set the global argparse
    global args
    args = args_from_spawn


def run(rank, args):
    """ Main entry-point into the program

    :param args: argparse
    :returns: None
    :rtype: None

    """
    init_multiprocessing_and_cuda(rank, args)                   # handle multi-process + cuda init logic
    loader, model, grapher = build_loader_model_grapher(args)   # build the model, loader and grapher
    print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))   # print the config to stdout (after ddp changes)
    optimizer, scheduler = build_optimizer(model)               # the optimizer for the vae
    if args.half:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # build the early-stopping (or best-saver) objects and restore if we had a previous model
    model = layers.append_save_and_load_fns(model, optimizer, scheduler, grapher, args)
    saver = layers.ModelSaver(model, early_stop=args.early_stop, rank=rank,
                              burn_in_interval=int(0.1 * args.epochs),  # Avoid tons of saving early on.
                              larger_is_better=False, max_early_stop_steps=10)
    restore_dict = saver.restore()
    init_epoch = restore_dict['epoch']

    # add the the fid model if requested
    if args.fid_server is not None:
        model.fid_client = FIDClient(host=args.fid_server.split(':')[0],
                                     port=args.fid_server.split(':')[1])

    # main training loop
    for epoch in range(init_epoch, args.epochs + 1):
        train(epoch, model, optimizer, loader.train_loader, grapher)
        test_loss = test(epoch, model, loader.test_loader, grapher)
        loader.set_all_epochs(epoch)  # set the epoch for distributed-multiprocessing

        # update the learning rate and plot it
        scheduler.step()
        # register_plots({'learning_rate_scalar': scheduler.get_last_lr()[0]}, grapher, epoch)
        register_plots({'learning_rate_scalar': optimizer.param_groups[0]['lr']}, grapher, epoch)

        if saver(test_loss):  # do one more test if we are early stopping
            saver.restore()
            test_loss = test(epoch, model, loader.test_loader, grapher)
            break

        if epoch == 2 and args.distributed_rank == 0:  # make sure we do at least 1 test and train pass
            config_to_post = vars(args)
            slurm_id = utils.get_slurm_id()
            if slurm_id is not None:
                config_to_post['slurm_job_id'] = slurm_id

            grapher.add_text('config', pprint.PrettyPrinter(indent=4).pformat(config_to_post), 0)


if __name__ == "__main__":
    args.multi_gpu_distributed = False  # automagically detected and set below

    if args.num_replicas > 1:
        # Distributed launch
        assert args.distributed_master is not None, "Specify --distributed-master for DDP."

        # Set some environment flags
        endpoint = '{}{}:{}'.format('tcp://' if 'tcp' not in args.distributed_master else '',
                                    args.distributed_master, args.distributed_port)
        os.environ['MASTER_ADDR'] = endpoint
        os.environ['MASTER_PORT'] = str(args.distributed_port)

        # Spawn processes if we have a special case of big node with 4 or 8 GPUs.
        num_gpus = utils.number_of_gpus()
        if num_gpus == args.num_replicas:  # Special case
            # Multiple devices in this process, convert to single processed
            print("detected single node - multi gpu setup: spawning processes")
            args.multi_gpu_distributed = True
            mp.spawn(run, nprocs=args.num_replicas, args=(args,))
        else:
            # Single device in this entire process
            print("detected distributed with 1 gpu - 1 process setup")
            assert num_gpus == 1, "Only 1 GPU per process supported; filter with CUDA_VISIBLE_DEVICES."
            run(rank=args.distributed_rank, args=args)

    else:
        # Non-distributed launch
        run(rank=0, args=args)
