import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.nn import GraphConv, Linear
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import FakeDataset
import torch_geometric
torch_geometric.seed.seed_everything(42)
data = FakeDataset(avg_num_nodes=20000, num_channels=128).generate_data()
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(data.x.size()[-1], 64).jittable()
        self.l1 = Linear(64, 32)
        self.conv2 = GraphConv(32, 16).jittable()
        self.l2 = Linear(16, torch.numel(torch.unique(data.y)))

    def forward(self, x, edge_index):
        x = F.relu(self.l1(self.conv1(x, edge_index)))
        x = self.l2(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = GCN().to(device)
def run_epoch(model, loader, optimizer):
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        batch_size = batch.batch_size
        loss = F.nll_loss(out[:batch_size], batch.y[:batch_size])
        loss.backward()
        optimizer.step()


import time

def benchmark_epoch_time(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    loader = NeighborLoader(data, [50, 50], transform=T.ToDevice(device), batch_size=128)
    for epoch in range(1, 21):
        if epoch==6:
            since=time.time()
        run_epoch(model, loader, optimizer)
    print("Average time per epoch:", (time.time()-since)/15.0)

print("Timing eager mode:")
benchmark_epoch_time(gcn)
print("compiling openAI GCN...")
openAI_gcn = torch.compile(gcn, dynamic=True, mode="reduce-overhead")
print("Timing default torch.compile w/ openAI triton:")
benchmark_epoch_time(openAI_gcn)
print("compiling GCN w/ NVIDIA libs...")
torch._dynamo.reset()
import thunder
print(thunder.examine(gcn))
nvfuser_gcn = thunder.compile(gcn)
print("Timing compile w/ nvfuser:")
benchmark_epoch_time(nvfuser_gcn)
