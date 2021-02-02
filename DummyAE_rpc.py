import torch
from torch import nn
import torch.distributed.rpc as rpc
import threading

class AE_part1(nn.Module):

    def __init__(self):
        super(AE_part1, self).__init__()
        self.input = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    #@rpc.functions.async_execution
    def forward(self, x):
        x = x.to_here()
        with threading.Lock():
            x = self.relu(self.input(x))
        return x


class AE_part2(nn.Module):

    def __init__(self):
        super(AE_part2, self).__init__()
        self.hidden = torch.nn.Linear(10, 5)
        self.relu = torch.nn.ReLU()

    #@rpc.functions.async_execution
    def forward(self, x):
        x = x.to_here()
        with threading.Lock():
            x = self.relu(self.hidden(x))
        return x


class AE_part3(nn.Module):

    def __init__(self):
        super(AE_part3, self).__init__()
        self.hidden2 = torch.nn.Linear(5, 10)  #the actual output
        self.output = torch.nn.Linear(10,1)     #just to check output when testing w/ mpi, will delete
        self.relu = torch.nn.ReLU()

    #@rpc.functions.async_execution
    def forward(self, x):
        x = x.to_here()
        with threading.Lock():
            x = self.relu(self.hidden2(x))
            x = self.relu(self.output(x))
        return x


class Dummy_AE(nn.Module):

    def __init__(self, num_split, workers):
        super(Dummy_AE, self).__init__()

        self.num_split = num_split

        # Put the first part of the Dummy_AE on workers[0]
        self.p1_AE = rpc.remote(
            workers[0],
            AE_part1,
        )

        # Put the second part of the Dummy_AE on workers[1]
        self.p2_AE = rpc.remote(
            workers[1],
            AE_part2,
        )

        # Put the third part of the Dummy_AE on workers[2]
        self.p3_AE = rpc.remote(
            workers[2],
            AE_part3,
        )

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            x_rref = rpc.RRef(x)
            w_rref = self.p1_AE.remote().forward(x_rref)
            y_rref = self.p2_AE.remote().forward(w_rref)
            z_fut = self.p3_AE.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        return torch.cat(torch.futures.wait_all(out_futures))

