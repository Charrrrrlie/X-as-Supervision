import math

import torch
import torch.nn as nn

from modules.gcn import GCN_simple, GCN_residual, GCN_SAGE_residual, my_batched_dense_to_sparse

class FFNHeader(nn.Module):
    def __init__(self, in_channels, hidden_channels, p_dropout=0.2):
        super(FFNHeader, self).__init__()
        self.layer1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class GCNDiscriminator_base(nn.Module):
    def __init__(self, cfg):
        super(GCNDiscriminator_base, self).__init__()

        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.output_dim = cfg['output_dim']
        self.disc_sup_dim = cfg['disc_sup_dim']

        self.num_nodes = cfg['num_node']

        self.use_self_loop = cfg['use_self_loop']

        self.input_layer = nn.Identity()
        self.gcn = nn.Identity()

        # squeeze gcn features [B, J, N] into [B, 1] label
        self.header = nn.Linear(self.output_dim * self.num_nodes, 1)

    def cal_positional_encoding(self, keypoints):
        B, J, C = keypoints.shape
        positional_encoding = torch.zeros(B, J, C).to(keypoints.device)
        for i in range(J):
            for j in range(C):
                if j % 2 == 0:
                    positional_encoding[:, i, j] = math.sin(i / 10000 ** (2 * j / C))
                else:
                    positional_encoding[:, i, j] = math.cos(i / 10000 ** (2 * j / C))
        return positional_encoding

    def compute_graph_matrix(self, keypoints, parent_ids, child_ids, num_nodes):
        B, N = keypoints.shape[:2]

        weight_matrix = torch.zeros(B, num_nodes, num_nodes).to(keypoints.device)
        if self.use_self_loop:
            # NOTE(xuanyi): to acclerate sparse computing, we use self-loop trick in "https://arxiv.org/abs/1609.02907",
            identity_matrix = torch.eye(num_nodes, num_nodes)
            identity_matrix =  identity_matrix.repeat(B, 1, 1).to(keypoints.device)
            weight_matrix = weight_matrix + identity_matrix

        weight_matrix[:, parent_ids, child_ids] = 1.
        weight_matrix[:, child_ids, parent_ids] = 1.

        # (2, B*E) , (B*E, )
        edge_index, edge_weight = my_batched_dense_to_sparse(weight_matrix)
        return edge_index, edge_weight

    def header_forward(self, graph_features, batch_size):
        graph_features = graph_features.view(batch_size, -1)
        graph_features = self.header(graph_features)

        return graph_features

    def forward(self, keypoints):
        raise NotImplementedError


class GCNDiscriminator(GCNDiscriminator_base):
    def __init__(self, cfg):
        super(GCNDiscriminator, self).__init__(cfg)

        if cfg['name'] == 'simple_gcn':
            self.name = 'SimpleGCN'
            self.gcn = nn.Sequential(
                GCN_simple(self.input_dim, self.hidden_dim, self_loop=self.use_self_loop),
                GCN_simple(self.input_dim, self.hidden_dim, self_loop=self.use_self_loop)
            )

        elif cfg['name'] == 'res_gcn':
            self.name = 'ResGCN'
            self.num_layers = cfg['num_layers']
            self.gcn = nn.Sequential(
                GCN_simple(self.input_dim, self.hidden_dim, self_loop=self.use_self_loop),
                *[GCN_residual(self.hidden_dim, self.hidden_dim, self.hidden_dim, self_loop=self.use_self_loop, use_bn=cfg['use_bn']) for _ in range(self.num_layers)],
                GCN_simple(self.hidden_dim, self.output_dim, self_loop=self.use_self_loop)
            )
        else:
            raise NotImplementedError

        self.input_layer = nn.Linear(self.disc_sup_dim, self.input_dim)

        self.parent_ids, self.child_ids = None, None

    def compute_graph_matrix(self, keypoints, parent_ids, child_ids, num_nodes):
        B, N = keypoints.shape[:2]

        start = keypoints[:, child_ids, :].clone()
        end = keypoints[:, parent_ids, :].clone()
        paired_diff = end - start # bone length as weight matrix
        l1_distances = torch.sqrt(torch.sum(paired_diff ** 2, dim=-1)) # compute the distances

        weight_matrix = torch.zeros(B, num_nodes, num_nodes).to(keypoints.device)
        if self.use_self_loop:
            identity_matrix = torch.eye(num_nodes, num_nodes)
            identity_matrix =  identity_matrix.repeat(B, 1, 1).to(keypoints.device)
            weight_matrix = weight_matrix + identity_matrix

        # NOTE(xuanyi): The library's default sparse adjacent matrix is for directed graph, so its shape is (2,E),
        # but our graph is undirected, so the sparse adj_matrix should be (2,2*E)
        weight_matrix[:, parent_ids, child_ids] = 1./l1_distances
        weight_matrix[:, child_ids, parent_ids] = 1./l1_distances

        # (2, B*E) , (B*E, )
        edge_index, edge_weight = my_batched_dense_to_sparse(weight_matrix)
        return edge_index, edge_weight

    def forward(self, keypoints):
        batch_size = keypoints.shape[0]

        edge_index, edge_weight = self.compute_graph_matrix(keypoints, self.parent_ids, self.child_ids, self.num_nodes)

        keypoints = self.input_layer(keypoints).view(batch_size * self.num_nodes, -1)

        node_features = self.gcn((keypoints, edge_index, edge_weight))
        output_features = self.header_forward(node_features[0], batch_size)

        return output_features


class GCNSAGEDiscriminator(GCNDiscriminator_base):
    def __init__(self, cfg):
        super(GCNSAGEDiscriminator, self).__init__(cfg)

        self.name = 'ResSAGEGCN'
        self.num_layers = cfg['num_layers']
        self.gcn = nn.Sequential(
            *[GCN_SAGE_residual(self.hidden_dim, self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)],
            GCN_SAGE_residual(self.hidden_dim, -1, self.output_dim, single_layer=True)
        )

        # use positional encoding
        self.use_pe = cfg['use_pe'] if 'use_pe' in cfg else False
        if self.use_pe:
            self.input_layer = nn.Linear(self.disc_sup_dim * 2, self.input_dim)
        else:
            self.input_layer = nn.Linear(self.disc_sup_dim, self.input_dim)

        self.parent_ids, self.child_ids = None, None

    def forward(self, keypoints):
        batch_size = keypoints.shape[0]
        device = keypoints.device

        edge_index, _ = self.compute_graph_matrix(keypoints, self.parent_ids, self.child_ids, self.num_nodes)

        if self.use_pe:
            positional_encoding = self.cal_positional_encoding(keypoints)
            keypoints = torch.cat([keypoints, positional_encoding], dim=-1)

        keypoints = self.input_layer(keypoints).view(batch_size * self.num_nodes, -1)

        node_features = self.gcn((keypoints, edge_index))
        output_features = self.header_forward(node_features[0], batch_size)

        return output_features


class GCNDiscriminatorDecouple(GCNDiscriminator_base):
    def __init__(self, cfg):
        super(GCNDiscriminatorDecouple, self).__init__(cfg)

        # use positional encoding
        self.use_pe = cfg['use_pe'] if 'use_pe' in cfg else False
        if self.use_pe:
            self.joint_input_layer = nn.Linear(self.disc_sup_dim * 2, self.input_dim)
            self.bone_input_layer = nn.Linear(self.disc_sup_dim * 2, self.input_dim)
        else:
            self.joint_input_layer = nn.Linear(self.disc_sup_dim, self.input_dim)
            self.bone_input_layer = nn.Linear(self.disc_sup_dim, self.input_dim)

        self.name = 'ResGCNDecouple'
        self.num_layers = cfg['num_layers']
        self.joint_gcn = nn.Sequential(
            *[GCN_SAGE_residual(self.hidden_dim, self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)],
            GCN_SAGE_residual(self.hidden_dim, -1, self.output_dim, single_layer=True)
        )

        self.bone_gcn = nn.Sequential(
            *[GCN_SAGE_residual(self.hidden_dim, self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)],
            GCN_SAGE_residual(self.hidden_dim, -1, self.output_dim, single_layer=True)
        )

        self.header = FFNHeader(self.output_dim * self.num_nodes * 2, 512)

        self.parent_ids, self.child_ids = None, None

    def forward(self, keypoints):
        batch_size, _, keypoints_dim = keypoints.shape

        start = keypoints[:, self.child_ids, :].clone()
        end = keypoints[:, self.parent_ids, :].clone()
        bone_vec = end - start

        bone_vec = torch.cat([torch.zeros(batch_size, 1, keypoints_dim).to(keypoints.device), bone_vec], dim=1)

        # ignore edge_weight
        edge_index, _ = self.compute_graph_matrix(keypoints, self.parent_ids, self.child_ids, self.num_nodes)

        if self.use_pe:
            keypoint_pe = self.cal_positional_encoding(keypoints)
            bone_pe = self.cal_positional_encoding(bone_vec)
            keypoints = torch.cat([keypoints, keypoint_pe], dim=-1)
            bone_vec = torch.cat([bone_vec, bone_pe], dim=-1)

        # joint stream
        keypoints = self.joint_input_layer(keypoints).view(batch_size * self.num_nodes, -1)
        keypoint_feat = self.joint_gcn((keypoints, edge_index))[0]

        # bone stream
        bone_vec = self.bone_input_layer(bone_vec).view(batch_size * self.num_nodes, -1)
        bone_feat = self.bone_gcn((bone_vec, edge_index))[0]

        output_features = torch.cat([keypoint_feat, bone_feat], dim=-1)
        output_features = self.header_forward(output_features, batch_size)

        return output_features