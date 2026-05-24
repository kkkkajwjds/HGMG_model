import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


class InstanceEncoder(nn.Module):


    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, node_embeds):
        x = torch.stack(node_embeds, dim=0)  # [L, D]
        x = x.mean(dim=0)  # pooling
        return self.mlp(x)


class InstanceAttention(nn.Module):


    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, instance_feats, target_embed):

        K = instance_feats.size(0)

        target_expand = target_embed.unsqueeze(0).repeat(K, 1)

        h = torch.cat([instance_feats, target_expand], dim=-1)

        scores = self.score(h).squeeze(-1)  # [K]

        alpha = F.softmax(scores, dim=0)

        z = torch.sum(alpha.unsqueeze(-1) * instance_feats, dim=0)

        return z, alpha


class InstanceEmbModel(nn.Module):

    def __init__(self, embed_dim, k2=40):
        super().__init__()
        self.encoder = InstanceEncoder(embed_dim)
        self.attention = InstanceAttention(embed_dim)
        self.k2 = k2

    def forward(self, instances_batch, node_embedding, target_nodes):
        if not torch.is_tensor(target_nodes):
            target_nodes = torch.tensor(target_nodes)
        device = target_nodes.device
        Z_list = []
        for i in range(len(instances_batch)):
            instances = instances_batch[i][:self.k2]
            if not instances:
                z = node_embedding[target_nodes[i]]
                Z_list.append(z)
                continue
            instance_feats = []
            for inst in instances:
                node_embeds = [node_embedding[nid] for nid in inst]
                z_i = self.encoder(node_embeds)
                instance_feats.append(z_i)
            instance_feats = torch.stack(instance_feats, dim=0)
            target_embed = node_embedding[target_nodes[i]]
            z, _ = self.attention(instance_feats, target_embed)
            Z_list.append(z)
        Z = torch.stack(Z_list, dim=0)

        return Z