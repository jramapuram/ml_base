import os
import time
import tree
import functools
import pprint
import torch
import torch.multiprocessing as mp
import numpy as np
import torchvision

from torchvision import transforms

import helpers.metrics as metrics
import helpers.layers as layers
import helpers.utils as utils

from helpers.metrics_client import MetricsClient


from vae_main import args, _extract_sum_scalars, init_multiprocessing_and_cuda, \
    build_optimizer, build_loader_model_grapher


COUNTER = 0


def request_remote_metrics_calc(epoch, model, grapher, prefix):
    """Helper to request remote server to compute metrics every post_every epoch.

    :param epoch: the current epoch
    :param model: the model
    :param grapher: the grapher object
    :param prefix: train / test / val
    :returns: nothing, asynchronously calls back the lbda function when complete
    :rtype: None

    """
    assert hasattr(model, 'metrics_client') and model.metrics_client is not None, "Metrics client not created."
    assert hasattr(model, 'test_images') and model.test_images is not None, "Metrics test images not setup."

    # Generate samples and post to the remote server
    with torch.no_grad():
        generated_imgs = model.generate_synthetic_samples(batch_size=10000,
                                                          reset_state=False,
                                                          use_aggregate_posterior=False).transpose(1, -1)
        assert generated_imgs.shape[0] == 10000, "need 10k generations for metrics, got {}.".format(
            generated_imgs.shape)

        # Build the lambda to post the images
        def loss_lbda(metrics_map, epoch, prefix):
            print("{}[Epoch {}] : {}".format(prefix, epoch, metrics_map))
            global COUNTER
            COUNTER += 1

        # POST the true data and the fake data.
        lbda = functools.partial(loss_lbda, epoch=epoch, prefix=prefix)
        fake_images = generated_imgs.detach().cpu().numpy() if 'binarized' not in args.task \
            else np.round(generated_imgs.detach().cpu().numpy())
        model.metrics_client.post_with_images(fake_images=fake_images,
                                              real_images=model.test_images,
                                              lbda=lbda)


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
    loss_map, num_samples = {}, 0

    # iterate over data and labels
    for num_minibatches, (minibatch, labels) in enumerate(loader):
        minibatch = minibatch.cuda(non_blocking=True) if args.cuda else minibatch
        labels = labels.cuda(non_blocking=True) if args.cuda else labels

        with torch.no_grad():
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

    # build the image map
    image_map = {'input_imgs': minibatch}

    # activate the logits of the reconstruction and get the dict
    image_map = {**image_map, **model.get_activated_reconstructions(pred_logits)}

    # tack on remote metrics information if requested, do it in-frequently.
    if args.metrics_server is not None:
        request_remote_metrics_calc(epoch, model, grapher, prefix)

    # Add generations to our image dict
    with torch.no_grad():
        prior_generated = model.generate_synthetic_samples(10000, reset_state=True,
                                                           use_aggregate_posterior=False)
        ema_generated = model.generate_synthetic_samples(10000, reset_state=True,
                                                         use_aggregate_posterior=True)
        image_map['prior'] = prior_generated
        image_map['ema'] = ema_generated

        # tack on MSSIM information if requested
        if args.calculate_msssim:
            loss_map['prior_gen_msssim_mean'] = metrics.calculate_mssim(
                minibatch, prior_generated[0:minibatch.shape[0]])
            loss_map['ema_gen_msssim_mean'] = metrics.calculate_mssim(
                minibatch, ema_generated[0:minibatch.shape[0]])

    # save all the images
    image_dir = os.path.join(args.log_dir, utils.get_name(args), 'images')
    os.makedirs(image_dir, exist_ok=True)

    for k, v in image_map.items():
        grid = torchvision.utils.make_grid(v, normalize=True, scale_each=True)
        grid_filename = os.path.join(image_dir, "{}.png".format(k))
        transforms.ToPILImage()(grid.cpu()).save(grid_filename)

        for idx, sample in enumerate(v):
            current_filename = os.path.join(image_dir, "{}_{}.png".format(idx, k))
            transforms.ToPILImage()(sample.cpu()).save(current_filename)

    # cleanups (see https://tinyurl.com/ycjre67m) + return ELBO for early stopping
    for d in [loss_map, image_map, reparam_map]:
        d.clear()

    del minibatch
    del labels


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

    # build the early-stopping (or best-saver) objects and restore if we had a previous model
    model = layers.append_save_and_load_fns(model, optimizer, scheduler, grapher, args)
    saver = layers.ModelSaver(model, early_stop=args.early_stop, rank=rank,
                              burn_in_interval=int(0.1 * args.epochs),  # Avoid tons of saving early on.
                              larger_is_better=False, max_early_stop_steps=10)
    restore_dict = saver.restore()
    init_epoch = restore_dict['epoch']

    # add the the metrics client as a member if requested
    if args.metrics_server is not None:
        model.metrics_client = MetricsClient(host=args.metrics_server.split(':')[0],
                                             port=args.metrics_server.split(':')[1])

    # main training loop
    for epoch in range(init_epoch, init_epoch + 1):
        loader.set_all_epochs(epoch)  # set the epoch for distributed-multiprocessing
        train(epoch, model, None, loader.train_loader, grapher)  # no optimizer
        test(epoch, model, loader.test_loader, grapher)


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

    if args.metrics_server is not None:
        print("training complete! sleeping for 60 min for remote metrics to complete...")
        init_time, max_timeout = [time.time(), 60 * 60]
        while COUNTER < 2:
            if time.time() - init_time > max_timeout:
                break

            time.sleep(60)
