import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphConv
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import FakeDataset
import torch_geometric
torch_geometric.seed.seed_everything(42)
data = FakeDataset(avg_num_nodes=20000).generate_data()
print(data)
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(data.x.size()[-1], 16).jittable()
        self.conv2 = GraphConv(16, torch.numel(torch.unique(data.y))).jittable()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
loader = NeighborLoader(data, [10, 10], transform=T.ToDevice(device), batch_size=64)
def run_epoch():
    
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        batch_size = batch.batch_size
        loss = F.nll_loss(out[:batch_size], batch.y[:batch_size])
        loss.backward()
        optimizer.step()


import time
model.train()
for epoch in range(1, 21):
    if epoch==6:
        since=time.time()
    run_epoch()
print("Average per epoch time:", (time.time()-since)/15.0)
