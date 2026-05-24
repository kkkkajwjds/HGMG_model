import numpy as np
import scipy.sparse as sp
import random
import pandas as pd
from scipy.sparse import coo_matrix

from collections import defaultdict
from sklearn.metrics import  normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch

from typing import List, Set, Tuple, Optional

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)

    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_val_test_split(labels):
    class_counts = {}
    for label in labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    train_ratio = 0.8
    val_ratio = 0.1
    train_samples = {}
    val_samples = {}
    test_samples = {}
    for label, count in class_counts.items():
        train_count = int(count * train_ratio)
        val_count = int(count * val_ratio)
        test_count = count - train_count - val_count

        # 随机打乱索引，以便随机选择样本
        indices = list(range(count))
        random.shuffle(indices)

        # 分配样本到各个数据集
        train_samples[label] = [indices.pop() for _ in range(train_count)]
        val_samples[label] = [indices.pop() for _ in range(val_count)]
        test_samples[label] = [indices.pop() for _ in range(test_count)]
    train_idx = train_samples['0'] + train_samples['1'] + train_samples['2']+train_samples['3']
    val_idx = val_samples['0'] + val_samples['1'] + val_samples['2']+val_samples['3']
    test_idx = test_samples['0'] + test_samples['1'] + test_samples['2']+test_samples['3']
    train_idx = random.sample(train_idx, len(train_idx))
    val_idx = random.sample(val_idx, len(val_idx))
    test_idx = random.sample(test_idx, len(test_idx))
    test_idx = torch.LongTensor(test_idx)
    val_idx = torch.LongTensor(val_idx)
    train_idx = torch.LongTensor(train_idx)
    return train_idx,val_idx,test_idx

def kmeans_test(X, y, n_clusters):
    nmi_list = []
    ari_list = []
    X=X.cpu()
    y=y.cpu()
    kmeans = KMeans(n_clusters=n_clusters)
    tsne = TSNE(n_components=2)
    reduced_features = tsne.fit_transform(X.detach().numpy())
    y_pred = kmeans.fit_predict(reduced_features)
    nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
    ari_score = adjusted_rand_score(y, y_pred,)
    nmi_list.append(nmi_score)
    ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def build_global_split_adj(
    edge_str: str,
    node_file_path: str,
    low_threshold: int = 3228,
    directed: bool = False,
    sparse: bool = True
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    parts = edge_str.strip().split()
    if parts and parts[0].upper() == 'E':
        parts = parts[1:]
    local_edges = []
    idx_list = [int(x) for x in parts]
    for i in range(0, len(idx_list), 2):
        local_edges.append((idx_list[i], idx_list[i+1]))
    if not directed:
        local_edges = local_edges + [(v, u) for (u, v) in local_edges]

    # 读取所有行，收集全局节点和边
    global_nodes: Set[int] = set()
    global_edges: Set[Tuple[int, int]] = set()

    with open(node_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            node_ids = [int(x) for x in line.split()]
            # 每行应该至少有4个节点（对应索引0,1,data,3）
            if len(node_ids) < 4:
                print(f"警告：某行节点数不足4，跳过该行")
                continue
            # 添加所有节点到全局集合
            global_nodes.update(node_ids)
            # 根据局部边结构，生成实际节点ID之间的边
            for u_idx, v_idx in local_edges:
                # 确保索引在范围内
                if u_idx < len(node_ids) and v_idx < len(node_ids):
                    u_real = node_ids[u_idx]
                    v_real = node_ids[v_idx]
                    # 无向图已经包含反向边，但集合会去重
                    global_edges.add((u_real, v_real))
                else:
                    print(f"警告：局部索引 ({u_idx},{v_idx}) 超出该行节点范围")

    # 分离低ID和高ID节点
    low_nodes = sorted([n for n in global_nodes if n <= low_threshold])
    high_nodes = sorted([n for n in global_nodes if n > low_threshold])
    n_low = 3228
    n_high = 37714

    # 建立映射：实际节点ID -> 新索引（低ID方阵的行/列索引，或二分矩阵的行/列索引）
    low_id_to_idx = {nid: i for i, nid in enumerate(low_nodes)}
    high_id_to_idx = {nid: i for i, nid in enumerate(high_nodes)}

    # ---------- 构建低ID方阵 ----------
    low_edges = []
    for u, v in global_edges:
        if u in low_id_to_idx and v in low_id_to_idx:
            low_edges.append((low_id_to_idx[u], low_id_to_idx[v]))
    if n_low > 0:
        if sparse and low_edges:
            indices = torch.tensor([[u, v] for u, v in low_edges], dtype=torch.long).t()
            values = torch.ones(len(low_edges), dtype=torch.float32)
            low_adj = torch.sparse_coo_tensor(indices, values, (n_low, n_low)).coalesce()
        elif low_edges:
            low_adj = torch.zeros(n_low, n_low, dtype=torch.float32)
            for u, v in low_edges:
                low_adj[u, v] = 1.0
        else:
            # 有节点无边
            if sparse:
                low_adj = torch.sparse_coo_tensor(torch.empty((2,0)), torch.empty(0), (n_low, n_low)).coalesce()
            else:
                low_adj = torch.zeros(n_low, n_low)
    else:
        low_adj = None

    # ---------- 构建二分矩阵（低->高） ----------
    bip_edges = []
    for u, v in global_edges:
        if u in low_id_to_idx and v in high_id_to_idx:
            bip_edges.append((low_id_to_idx[u], high_id_to_idx[v]))
        if u in high_id_to_idx and v in low_id_to_idx:
            bip_edges.append((low_id_to_idx[v], high_id_to_idx[u]))
    if n_low > 0 and n_high > 0:
        if sparse and bip_edges:
            indices = torch.tensor([[r, c] for r, c in bip_edges], dtype=torch.long).t()
            values = torch.ones(len(bip_edges), dtype=torch.float32)
            bip_adj = torch.sparse_coo_tensor(indices, values, (n_low, n_high)).coalesce()
        elif bip_edges:
            bip_adj = torch.zeros(n_low, n_high, dtype=torch.float32)
            for r, c in bip_edges:
                bip_adj[r, c] = 1.0
        else:
            # 有节点但无跨边
            if sparse:
                bip_adj = torch.sparse_coo_tensor(torch.empty((2,0)), torch.empty(0), (n_low, n_high)).coalesce()
            else:
                bip_adj = torch.zeros(n_low, n_high)
    else:
        bip_adj = None

    return low_adj, bip_adj



def load_instances(path):
    instances = []
    with open(path, "r") as f:
        for line in f:
            inst = list(map(int, line.strip().split()))
            instances.append(inst)
    return instances



def build_instances_dict(instances, target_nodes):
    target_set = set(int(x) for x in target_nodes)
    instances_dict = defaultdict(list)
    for inst in instances:
        inst_set = set(inst)
        for t in target_set:
            if t in inst_set:
                instances_dict[t].append(inst)
    return instances_dict



def build_instance_batch(target_list, instances_dict):

    instances_batch = []
    target_nodes = []

    for t in target_list:

        t = int(t)

        if t in instances_dict:
            instances_batch.append(instances_dict[t])
        else:
            instances_batch.append([])

        target_nodes.append(t)

    return instances_batch