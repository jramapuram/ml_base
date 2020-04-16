"""Launches a distributed pipeline."""
#!/usr/bin/env python

import os
import pprint
import functools
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.multiprocessing import Process

from vae_main import args, run


def init_process(start_fn, args, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29300'
    mp.spawn(start_fn, nprocs=args.num_replicas, args=(args.num_replicas, args))


if __name__ == "__main__":
    print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))
    init_process(start_fn=run, args=args)


# def init_process(rank, size, fn, backend='gloo'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29300'
#     dist.init_process_group(backend, rank=rank, world_size=size)

#     fn(rank, size)


# if __name__ == "__main__":
#     print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))

#     # Spawn up our multi-process workers
#     processes = []
#     for rank in range(args.num_replicas):
#         p = Process(target=init_process, args=(rank, args.num_replicas, functools.partial(run, args=args)))
#         p.start()
#         processes.append(p)

#     # Join in order to prevent processes from dangling
#     for p in processes:
#         p.join()
