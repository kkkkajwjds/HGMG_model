from __future__ import division, print_function

import time
import random
import argparse
import warnings
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score

from utils import (
    accuracy,
    train_val_test_split,
    kmeans_test,
    build_global_split_adj,
load_instances,build_instances_dict,build_instance_batch
)

from HGMG import HGMGModel

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=72)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--hidden', type=int, default=128)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ============================================================
# Load features / labels
# ============================================================
features = torch.load("IMDB/ft_all.pt").float()
ft_c = torch.load("IMDB/ft_c.pt").float()
ft_o = torch.load("IMDB/ft_o.pt").float()
target_node=[]
movie_id=[]
with open('IMDB/data/movie_labels.txt') as f:
    for line in f:
        temp = list(line.strip('\n').split())
        movie_id.append(temp[1])
        target_node.append(int(temp[0]))

data_list = [int(item) for item in movie_id]
labels = torch.tensor(data_list)

# ============================================================
# Meta-graph
# ============================================================
edge_str = "E	0	1	1	0	1	2	2	1	2	3	3	2"
node_file = "IMDB/instance/1"
low_threshold = 3228

core_node_adj, other_node_adj = build_global_split_adj(
    edge_str=edge_str,
    node_file_path=node_file,
    low_threshold=low_threshold,
    directed=False,
    sparse=True
)
print(core_node_adj.shape,other_node_adj.shape)


idx_train, idx_val, idx_test = train_val_test_split(movie_id)

# ============================================================
# Build instances_batch + target_nodes
# ============================================================
instances=load_instances(node_file)
instances_dict=build_instances_dict(instances,target_node)
instances_batch=build_instance_batch(target_node,
    instances_dict)
instances_batch_list=[instances_batch]

model = HGMGModel(
    feature_dim=features.shape[1],
    hidden_dim=args.hidden,
    num_classes=int(labels.max()) + 1
)

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

# ============================================================
# CUDA
# ============================================================
if args.cuda:
    model.cuda()
    features = features.cuda()
    ft_c=ft_c.cuda()
    ft_c = ft_c.cuda()
    labels = labels.cuda()
    core_node_adj = core_node_adj.cuda()
    other_node_adj = other_node_adj.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# ============================================================
# Variable
# ============================================================
features = Variable(features)
labels = Variable(labels)
ft_c = Variable(ft_c)
ft_o = Variable(ft_o)



def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    logits, F_emb, alpha, _, _ = model(
        X_u=features,
        adj_t_list=[core_node_adj],
        adj_ot_list=[other_node_adj],
        X_t_list=ft_c,
        X_ot_list=ft_o,
        instances_batch_list=instances_batch_list,
        target_nodes=target_node,
        k3=50
    )
    acc_train = accuracy(logits[idx_train], labels[idx_train])
    x_train=logits[idx_train]
    y_train=labels[idx_train]
    loss_train = F.nll_loss(x_train, y_train)
    f1_micro_train = f1_score(y_train.data.cpu(), x_train.data.cpu().argmax(1), average='micro')
    f1_macro_train = f1_score(y_train.data.cpu(), x_train.data.cpu().argmax(1), average='macro')

    loss_train.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        logits, F_emb, alpha, _, _ = model(
            X_u=features,
            adj_t_list=[core_node_adj],
            adj_ot_list=[other_node_adj],
            X_t_list=ft_c,
            X_ot_list=ft_o,
            instances_batch_list=instances_batch_list,
            target_nodes=target_node,
            k3=50
        )

        x_val = logits[idx_val]
        y_val = labels[idx_val]
        loss_val = F.nll_loss(x_val, y_val)
        f1_micro_val = f1_score(y_val.data.cpu(), x_val.data.cpu().argmax(1), average='micro')
        f1_macro_val = f1_score(y_val.data.cpu(), x_val.data.cpu().argmax(1), average='macro')

        best_f1_micro_val = 0
        best_f1_macro_val = 0

        print(
            'epoch: {:3d}'.format(epoch),
            'train loss: {:.4f}'.format(loss_train),
            'train micro f1: {:.4f}'.format(f1_micro_train),
            'train macro f1: {:.4f}'.format(f1_macro_train),
            'val micro f1: {:.4f}'.format(f1_micro_val),
            'val macro f1: {:.4f}'.format(f1_macro_val),
        )

        if f1_micro_val > best_f1_micro_val and f1_macro_val > best_f1_macro_val:
            best_f1_micro_val = f1_micro_val
            best_f1_macro_val = f1_macro_val
        print(" best_f1_micro_val = {}, best_f1_macro_val = {}".format(best_f1_micro_val, best_f1_macro_val))
def compute_test():
    model.eval()
    logits, F_emb, alpha, _, _ = model(
        X_u=features,
        adj_t_list=[core_node_adj],
        adj_ot_list=[other_node_adj],
        X_t_list=ft_c,
        X_ot_list=ft_o,
        instances_batch_list=instances_batch_list,
        target_nodes=target_node,
        k3=50
    )


    x_test = logits[idx_test]
    y_test = labels[idx_test]
    loss_test = F.nll_loss(x_test, y_test)
    acc_test = accuracy(x_test, y_test)

    f1_micro_test = f1_score(y_test.data.cpu(), x_test.data.cpu().argmax(1), average='micro')
    f1_macro_test = f1_score(y_test.data.cpu(), x_test.data.cpu().argmax(1), average='macro')

    x_test = logits[idx_test]
    y_test = labels[idx_test]
    loss_test = F.nll_loss(x_test, y_test)
    acc_test = accuracy(x_test, y_test)

    print(
        'test micro f1: {:.4f}'.format(f1_micro_test.item()),
        'test macro f1: {:.4f}'.format(f1_macro_test.item()),
    )

# ============================================================
# RUN
# ============================================================


for epoch in range(args.epochs):
    train(epoch)

compute_test()