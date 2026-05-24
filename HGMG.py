import torch
import torch.nn as nn

from Metagraph_fusion import MetaGraphFusion
from Node_emb_model import node_embedding
from SimNode_emb_model import SimGNN, build_similarity_adj
from Instance_emb_model import InstanceEmbModel
class HGMGModel(nn.Module):

    def __init__(self,feature_dim,hidden_dim,num_classes,alpha=0.6):
        super().__init__()
        # ======================
        # Node encoder
        # ======================
        self.node_encoder = node_embedding(
            in_dim=feature_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            alpha=alpha
        )
        # ======================
        # Instance encoder
        # ======================
        self.instance_encoder = InstanceEmbModel(
            embed_dim=hidden_dim,
            k2=50
        )
        # ======================
        # MetaGraph fusion
        # ======================
        self.meta_fusion = MetaGraphFusion(embed_dim=hidden_dim)

        # ======================
        # Similarity branch
        # ======================
        self.sim_gnn = SimGNN(
            in_dim=feature_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim
        )
        # ======================
        # Final fusion
        # ======================
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _encode_meta_graphs(
        self,
        adj_t_list,
        adj_ot_list,
        X_t_list,
        X_ot_list,
        Z_list  # [B, D]
    ):
        H_list = []
        for i in range(len(adj_t_list)):

            X_m = self.node_encoder(
                adj_t_list[i],
                adj_ot_list[i],
                X_t_list,
                X_ot_list
            )
            Z_m = Z_list[i]
            H_m = torch.cat([X_m, Z_m], dim=-1)
            proj = nn.Linear(H_m.size(-1), X_m.size(-1)).to(H_m.device)
            H_m = proj(H_m)
            H_list.append(H_m)
        return H_list
    def forward(self,X_u,adj_t_list,adj_ot_list,X_t_list,X_ot_list,instances_batch_list,target_nodes,k3=50
    ):
        Z_list=[]
        for i in range(len(adj_t_list)):
            Z = self.instance_encoder(instances_batch_list[i],X_u,target_nodes)
            Z_list.append(Z)
        H_list = self._encode_meta_graphs(adj_t_list,adj_ot_list,X_t_list,X_ot_list,Z_list)
        X_mg, alpha = self.meta_fusion(H_list)
        A_sim = build_similarity_adj(X_t_list, k=k3)
        X_sim = self.sim_gnn(X_t_list, A_sim)
        F = self.fusion(torch.cat([X_mg, X_sim], dim=-1))
        logits = self.classifier(F)
        return {logits, F,X_mg,Z,alpha}