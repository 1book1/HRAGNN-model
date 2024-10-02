import torch.nn as nn
import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import torch.optim as optim
from adabound import AdaBound


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class RelationAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout, i):
        super(RelationAttention, self).__init__()
        self.W = nn.ParameterDict({
            f'relation_{i}': nn.Parameter(torch.Tensor(in_features, out_features))
        })
        self.a = nn.ParameterDict({
            f'relation_{i}': nn.Parameter(torch.Tensor(2 * out_features, 1))
        })
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)

        self.init_weights(i)

    def init_weights(self, i):
        nn.init.xavier_uniform_(self.W[f'relation_{i}'])
        nn.init.xavier_uniform_(self.a[f'relation_{i}'])

    def _prepare_attention_input(self, Wh, relation_type):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * Wh.size(1))

    def forward(self, h, adj, rel):
        Wh = torch.matmul(h, self.W[f'relation_{rel}'])
        e = self._prepare_attention_input(Wh, rel)
        attention = self.leakyrelu(torch.matmul(e, self.a[f'relation_{rel}']).squeeze(2))
        attention = torch.where(adj.to_dense() > 0, attention, torch.tensor(float('-inf')).to(attention.device))
        attention = F.softmax(attention, dim=1)

        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)

        return h_prime, attention


class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=True, aggregation='mean'):  # Reduce dropout rate
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self.aggregation = aggregation
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        support = self.dropout(support)

        if self.aggregation == 'mean':
            degree = torch.sparse.sum(adj, dim=1).to_dense()
            degree_inv = torch.pow(degree, -1)
            degree_inv[degree_inv == float('inf')] = 0
            degree_inv = degree_inv.view(-1, 1)
            norm_adj = torch.sparse_coo_tensor(adj._indices(), adj._values() * degree_inv.squeeze(), adj.size())
            output = torch.sparse.mm(norm_adj, support)
        elif self.aggregation == 'sum':
            output = torch.sparse.mm(adj, support)
        elif self.aggregation == 'max':
            output = torch.sparse.mm(adj.to_dense(), support)
            output, _ = torch.max(output, dim=1)
        else:
            raise ValueError("Unsupported aggregation method.")

        if self.bias is not None:
            output += self.bias.unsqueeze(0)

        output = F.leaky_relu(output, 0.01)
        output = self.dropout(output)

        return output


class HRANGNN(nn.Module):
    def __init__(self, in_features, hidden_features, num_layers=2, dropout=0.3):
        super(HRANGNN, self).__init__()
        self.attention_head1 = RelationAttention(in_features, hidden_features[0], dropout, i=0)
        self.attention_head2 = RelationAttention(in_features, hidden_features[0], dropout, i=1)
        self.attention_head3 = RelationAttention(in_features, hidden_features[0], dropout, i=2)

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(hidden_features[0], hidden_features[1], dropout))
        self.gnn_layers.append(GNNLayer(hidden_features[1], hidden_features[2], dropout))

        self.residual_connection = nn.Linear(hidden_features[1], hidden_features[2])

        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, relation):
        relation_outputs = []

        weights = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]))

        h_prime1, attention_weights = self.attention_head1(x, adj[0], 0)
        relation_outputs.append(h_prime1)

        h_prime2, attention_weights = self.attention_head2(x, adj[1], 1)
        relation_outputs.append(h_prime2)

        h_prime3, attention_weights = self.attention_head3(x, adj[2], 2)
        relation_outputs.append(h_prime3)

        # Aggregate the results of different relationship types
        normalized_weights = F.softmax(weights, dim=0)

        h_prime = torch.sum(
            torch.stack([normalized_weights[i] * relation_outputs[i] for i in range(len(relation_outputs))]), dim=0)
        h_prime = self.activation(h_prime)

        h_prime = self.gnn_layers[0](h_prime, adj[relation])
        residual = h_prime
        h_prime = self.gnn_layers[1](h_prime, adj[relation])
        #h_prime = self.gnn_layers[2](h_prime, adj[relation])
        if residual.size(1) != h_prime.size(1):
            residual = self.residual_connection(residual)
        h_prime = h_prime + residual

        # h_prime = self.activation(h_prime)
        h_prime = self.dropout(h_prime)

        return h_prime


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)

        return x


class MVFN(nn.Module):
    def __init__(self, num_view, num_cls, hmvfn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hmvfn_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hmvfn_dim, num_cls)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)

        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
                          (-1, pow(self.num_cls, 2), 1))
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_cls, i + 1), 1))
        mvfn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(mvfn_feat)

        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc):  # 3, 4
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = HRANGNN(dim_list[i], dim_he_list)
        model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = MVFN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e, lr_c):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = AdaBound(
            list(model_dict["E{:}".format(i + 1)].parameters()) + list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e, gamma=1e-5, weight_decay=1e-4, amsbound=True)
    if num_view >= 2:
        optim_dict["C"] = AdaBound(model_dict["C"].parameters(), lr=lr_c, gamma=1e-5, amsbound=True)
    return optim_dict


