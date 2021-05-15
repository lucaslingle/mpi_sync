import torch as tc
from mpi4py import MPI


class LinearRegression(tc.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = tc.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)


comm = MPI.COMM_WORLD
tc.manual_seed(comm.Get_rank())
IN_DIM = 10
OUT_DIM = 1
model = LinearRegression(IN_DIM, OUT_DIM)

for p in model.parameters():
    print(f"process {comm.Get_rank()}, data: {p.data}")
