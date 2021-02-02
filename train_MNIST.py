import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import socket


def run(rank, size):
    """ Distributed function to be implemented later. """
    out = torch.randn(1, 1)
    print('out: ' + str(out))


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


""" Dataset partitioning helper """


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


""" Partitioning MNIST """


def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 1  # 128 / float(size)
    # partition_sizes = [.001 / size for _ in range(size)]
    partition_sizes = [1.6666666666666667e-05 for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)
    return train_set, bsz


'''
""" Distributed Synchronous SGD Example """
def run(rank, size):
    print('RUNNING!')
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)
'''

""" Gradient averaging. """


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


if __name__ == "__main__":
    size = 2
    processes = []
    print('yo!')

    f = open("mynodes.txt", "r")
    nodes = f.read()
    nodes_list = nodes.split("\n")

    if nodes_list.index(socket.gethostname()) == 0:
        addr = str(os.system(
            'hostname -I | awk {print$1}'))
        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = '8080'

    for rank in range(size):
        if rank % size != nodes_list.index(socket.gethostname()): continue
        print('start')
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
