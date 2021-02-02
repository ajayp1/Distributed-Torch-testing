# DEFINE TRAINING LOOP
import torch.distributed.autograd as dist_autograd
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from DummyAE_rpc import *
import socket
import os
import time
import torch.multiprocessing as mp


def run_master(num_split, worker_list):
    # put the two model parts on worker1 and worker2 respectively
    model = Dummy_AE(num_split, worker_list)
    x = torch.randn(1, 10)
    out = model(x)
    print("output: " + str(out))


################# STEP 4
# Launch RPC

def run_worker(rank, world_size, num_split):

    f = open("mynodes.txt","r")
    nodes = f.read()
    nodes_list = nodes.split("\n")

    os.environ['MASTER_ADDR'] = str(os.system(nodes_list[0] + ' -I | awk {print$1}'))
    os.environ['MASTER_PORT'] = '8080'                #probably this
    # options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)   #change....., set omp_numthreads, cpus per test, just try gloo

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split, nodes_list[1:-1])
    else:
        rpc.init_rpc(
            nodes_list[rank],
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":

    f = open("mynodes.txt","r")
    nodes = f.read()
    nodes_list = nodes.split("\n")
    world_size = 4
    num_split = 1

    mp.spawn(run_worker, args=(nodes_list.index(socket.gethostname()), world_size, num_split), nprocs=world_size, join=True)