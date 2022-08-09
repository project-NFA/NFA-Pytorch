import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import dataloader
import torch.nn.functional as F

class FeatureEncoder:
    def __init__(self, x_index, channels):
        self.embedding_table = []
        for key in x_index:
            embedding = nn.Embedding(key, channels).to(world.device)
            embedding.weight.requires_grad = False
            nn.init.xavier_normal_(embedding.weight, gain=1)
            self.embedding_table.append(embedding)
        self.input_channels = len(x_index) * channels
        self.channels = channels
        self.m = len(x_index)

    def encode(self, x):
        n = len(x)

        fea = x[0]
        emb = torch.tensor([]).to(world.device)
        for j in range(self.m):
            emb = torch.cat([emb, self.embedding_table[j].weight[fea[j]]], 0)
        emb = emb.reshape(1, self.input_channels)

        for i in range(1, n):
            fea = x[i]
            e = torch.tensor([]).to(world.device)
            for j in range(self.m):
                e = torch.cat([e, self.embedding_table[j].weight[fea[j]]], 0)
            emb = torch.cat([emb, e.reshape(1, self.input_channels)], 0)

        return emb

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class NFALightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(NFALightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.user_index, self.item_index = self.dataset.get_index()
        self.user_fea, self.item_fea = self.dataset.get_fea()
        self.user_encoder = FeatureEncoder(self.user_index, self.latent_dim)
        self.item_encoder = FeatureEncoder(self.item_index, self.latent_dim)
        self.embedding_user = self.user_encoder.encode(self.user_fea)
        self.embedding_item = self.item_encoder.encode(self.item_fea)
        self.user_encode_layer = nn.Linear(self.latent_dim * len(self.user_index), self.latent_dim)
        self.item_encode_layer = nn.Linear(self.latent_dim * len(self.item_index), self.latent_dim)
        # nn.init.xavier_normal_(self.user_encode_layer.weight, gain=1)
        # nn.init.xavier_normal_(self.item_encode_layer.weight, gain=1)
        '''
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        '''
        self.u2i_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.i2u_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
        nn.init.normal_(self.u2i_trans_layer.weight, std=0.1)
        nn.init.normal_(self.u2i_trans_layer.bias, std=0.1)
        nn.init.normal_(self.i2u_trans_layer.weight, std=0.1)
        nn.init.normal_(self.i2u_trans_layer.bias, std=0.1)
        #self.u2i_trans_layer.weight = nn.Parameter(0.01 * torch.randn([self.latent_dim, self.latent_dim]) + torch.eye(self.latent_dim, self.latent_dim))
        #self.u2i_trans_layer.weight.requires_grad = False
        self.f = nn.Sigmoid()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.Graph = self.dataset.getSparseGraph()

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def set_u2i_trans(self, flag):
        self.u2i_trans_layer.weight.requires_grad = flag

    def set_i2u_trans(self, flag):
        self.i2u_trans_layer.weight.requires_grad = flag

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def i2u_trans(self, x):
        #return x
        return self.i2u_trans_layer(x)

    def u2i_trans(self, x):
        #return x
        return self.u2i_trans_layer(x)

    def computer(self):
        """
        propagate methods for lightGCN
        """

        users_emb = self.f(self.user_encode_layer(self.embedding_user))
        items_emb = self.f(self.item_encode_layer(self.embedding_item))
        #users_emb = self.embedding_user.weight
        #items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            u2i_trans_emb = self.u2i_trans(users_emb)
            i2u_trans_emb = self.i2u_trans(items_emb)
            users_all_emb = torch.cat([users_emb, i2u_trans_emb])
            items_all_emb = torch.cat([u2i_trans_emb, items_emb])
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    users_emb = torch.sparse.mm(g_droped[f], users_all_emb)
                    items_emb = torch.sparse.mm(g_droped[f], items_all_emb)
                    users_emb = users_emb[:self.num_users]
                    items_emb = items_emb[self.num_users:]
                    temp_emb.append(torch.cat([users_emb, items_emb]))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                users_emb = torch.sparse.mm(g_droped, users_all_emb)
                items_emb = torch.sparse.mm(g_droped, items_all_emb)
                users_emb = users_emb[:self.num_users]
                items_emb = items_emb[self.num_users:]
                all_emb = torch.cat([users_emb, items_emb])
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        #light_out = all_emb
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        all_users = self.u2i_trans(all_users)
        #all_items = self.i2u_trans(all_items)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user[users]
        pos_emb_ego = self.embedding_item[pos_items]
        neg_emb_ego = self.embedding_item[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        users_res_emb = self.i2u_trans(self.u2i_trans(users_emb)) - users_emb
        pos_res_emb = self.u2i_trans(self.i2u_trans(pos_emb)) - pos_emb
        neg_res_emb = self.u2i_trans(self.i2u_trans(neg_emb)) - neg_emb
        reb_loss = (1 / 2) * (users_res_emb.norm(2).pow(2) +
                              pos_res_emb.norm(2).pow(2) +
                              neg_res_emb.norm(2).pow(2)) / float(len(users))
        users_emb = self.u2i_trans(users_emb)
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reb_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = self.u2i_trans(all_users[users])
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class NFAGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(NFAGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.user_index, self.item_index = self.dataset.get_index()
        self.user_fea, self.item_fea = self.dataset.get_fea()
        self.user_encoder = FeatureEncoder(self.user_index, self.latent_dim)
        self.item_encoder = FeatureEncoder(self.item_index, self.latent_dim)
        self.embedding_user = self.user_encoder.encode(self.user_fea)
        self.embedding_item = self.item_encoder.encode(self.item_fea)
        self.user_encode_layer = nn.Linear(self.latent_dim * len(self.user_index), self.latent_dim)
        self.item_encode_layer = nn.Linear(self.latent_dim * len(self.item_index), self.latent_dim)
        '''
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        '''
        self.convs = nn.ModuleList()
        for _ in range(self.n_layers):
            conv = nn.Linear(self.latent_dim, self.latent_dim)
            nn.init.xavier_normal_(conv.weight, gain=1)
            nn.init.normal_(conv.bias, std=0.1)
            self.convs.append(conv)
        self.u2i_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.i2u_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
        #self.u2i_trans_layer.weight = nn.Parameter(0.01 * torch.randn([self.latent_dim, self.latent_dim]) + torch.eye(self.latent_dim, self.latent_dim))
        #self.u2i_trans_layer.weight.requires_grad = False
        self.f = nn.Sigmoid()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.Graph = self.dataset.getSparseGraph()
        nn.init.normal_(self.u2i_trans_layer.weight, std=0.1)
        nn.init.normal_(self.i2u_trans_layer.weight, std=0.1)
        nn.init.normal_(self.u2i_trans_layer.bias, std=0.1)
        nn.init.normal_(self.i2u_trans_layer.bias, std=0.1)

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def i2u_trans(self, x):
        #return x
        return self.i2u_trans_layer(x)

    def u2i_trans(self, x):
        #return x
        return self.u2i_trans_layer(x)

    def computer(self):
        """
        propagate methods for lightGCN
        """

        users_emb = self.f(self.user_encode_layer(self.embedding_user))
        items_emb = self.f(self.item_encode_layer(self.embedding_item))
        #users_emb = self.embedding_user.weight
        #items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            u2i_trans_emb = self.u2i_trans(users_emb)
            i2u_trans_emb = self.i2u_trans(items_emb)
            users_all_emb = torch.cat([users_emb, i2u_trans_emb])
            items_all_emb = torch.cat([u2i_trans_emb, items_emb])
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    users_emb = torch.sparse.mm(g_droped[f], users_all_emb)
                    items_emb = torch.sparse.mm(g_droped[f], items_all_emb)
                    users_emb = users_emb[:self.num_users]
                    items_emb = items_emb[self.num_users:]
                    temp_emb.append(self.relu(self.convs[layer](torch.cat([users_emb, items_emb]))))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                users_emb = torch.sparse.mm(g_droped, users_all_emb)
                items_emb = torch.sparse.mm(g_droped, items_all_emb)
                users_emb = users_emb[:self.num_users]
                items_emb = items_emb[self.num_users:]
                all_emb = self.relu(self.convs[layer](torch.cat([users_emb, items_emb])))
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        #light_out = torch.cat(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        #all_users = torch.cat([self.u2i_trans(l1) for l1 in torch.chunk(all_users, self.n_layers + 1, dim=1)], dim=1)
        all_users = self.u2i_trans(all_users)
        #all_items = self.i2u_trans(all_items)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user[users]
        pos_emb_ego = self.embedding_item[pos_items]
        neg_emb_ego = self.embedding_item[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())


        users_res_emb = self.i2u_trans(self.u2i_trans(users_emb)) - users_emb
        pos_res_emb = self.u2i_trans(self.i2u_trans(pos_emb)) - pos_emb
        neg_res_emb = self.u2i_trans(self.i2u_trans(neg_emb)) - neg_emb
        '''
        users_res_emb = torch.cat([self.i2u_trans(self.u2i_trans(l1)) for l1 in torch.chunk(users_emb, self.n_layers + 1, dim=1)], dim=1) - users_emb
        pos_res_emb = torch.cat([self.u2i_trans(self.i2u_trans(l1)) for l1 in torch.chunk(pos_emb, self.n_layers + 1, dim=1)], dim=1) - pos_emb
        neg_res_emb = torch.cat(
            [self.u2i_trans(self.i2u_trans(l1)) for l1 in torch.chunk(neg_emb, self.n_layers + 1, dim=1)], dim=1) - neg_emb
        '''
        reb_loss = (1 / 2) * (users_res_emb.norm(2).pow(2) +
                              pos_res_emb.norm(2).pow(2) +
                              neg_res_emb.norm(2).pow(2)) / float(len(users))
        #users_emb = torch.cat([self.u2i_trans(l1) for l1 in torch.chunk(users_emb, self.n_layers + 1, dim=1)], dim=1)
        users_emb = self.u2i_trans(users_emb)
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reb_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = self.u2i_trans(all_users[users])
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma



class NFANGCF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(NFANGCF, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.user_index, self.item_index = self.dataset.get_index()
        self.user_fea, self.item_fea = self.dataset.get_fea()
        self.user_encoder = FeatureEncoder(self.user_index, self.latent_dim)
        self.item_encoder = FeatureEncoder(self.item_index, self.latent_dim)
        self.embedding_user = self.user_encoder.encode(self.user_fea)
        self.embedding_item = self.item_encoder.encode(self.item_fea)
        self.user_encode_layer = nn.Linear(self.latent_dim * len(self.user_index), self.latent_dim)
        self.item_encode_layer = nn.Linear(self.latent_dim * len(self.item_index), self.latent_dim)
        '''
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        '''
        self.convs_bi = nn.ModuleList()
        self.convs_gc = nn.ModuleList()
        for _ in range(self.n_layers):
            conv_bi = nn.Linear(self.latent_dim, self.latent_dim)
            self.convs_bi.append(conv_bi)
            conv_gc = nn.Linear(self.latent_dim, self.latent_dim)
            self.convs_gc.append(conv_gc)
        self.u2i_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.i2u_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
        '''
        self.u2i_trans_layer.weight = nn.Parameter(
            0.01 * torch.randn([self.latent_dim, self.latent_dim]) + torch.eye(self.latent_dim, self.latent_dim))
        '''
        #self.u2i_trans_layer.weight.requires_grad = False
        self.f = nn.Sigmoid()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.Graph = self.dataset.getSparseGraph()
        nn.init.normal_(self.u2i_trans_layer.weight, std=0.1)
        nn.init.normal_(self.i2u_trans_layer.weight, std=0.1)
        nn.init.normal_(self.u2i_trans_layer.bias, std=0.1)
        nn.init.normal_(self.i2u_trans_layer.bias, std=0.1)

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def i2u_trans(self, x):
        #return x
        return self.i2u_trans_layer(x)

    def u2i_trans(self, x):
        #return x
        return self.u2i_trans_layer(x)
        # return self.u2i_trans_layer(x)

    def computer(self):
        """
        propagate methods for lightGCN
        """

        users_emb = self.f(self.user_encode_layer(self.embedding_user))
        items_emb = self.f(self.item_encode_layer(self.embedding_item))
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            u2i_trans_emb = self.u2i_trans(users_emb)
            i2u_trans_emb = self.i2u_trans(items_emb)
            users_all_emb = torch.cat([users_emb, i2u_trans_emb])
            items_all_emb = torch.cat([u2i_trans_emb, items_emb])
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    users_side_emb = torch.sparse.mm(g_droped[f], users_all_emb)
                    items_side_emb = torch.sparse.mm(g_droped[f], items_all_emb)
                    users_sum_emb = self.convs_gc[layer](users_side_emb)
                    items_sum_emb = self.convs_gc[layer](items_side_emb)
                    users_bi_emb = self.convs_bi[layer](torch.mul(users_all_emb, users_side_emb))
                    items_bi_emb = self.convs_bi[layer](torch.mul(items_all_emb, items_side_emb))
                    users_emb = self.relu(users_sum_emb + users_bi_emb)
                    items_emb = self.relu(items_sum_emb + items_bi_emb)
                    users_emb = users_emb[:self.num_users]
                    items_emb = items_emb[self.num_users:]
                    temp_emb.append(torch.cat([users_emb, items_emb]))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = F.normalize(side_emb, p=2, dim=1)
            else:
                users_side_emb = torch.sparse.mm(g_droped, users_all_emb)
                items_side_emb = torch.sparse.mm(g_droped, items_all_emb)
                users_sum_emb = self.convs_gc[layer](users_side_emb)
                items_sum_emb = self.convs_gc[layer](items_side_emb)
                users_bi_emb = self.convs_bi[layer](torch.mul(users_all_emb, users_side_emb))
                items_bi_emb = self.convs_bi[layer](torch.mul(items_all_emb, items_side_emb))
                users_emb = self.relu(users_sum_emb + users_bi_emb)
                items_emb = self.relu(items_sum_emb + items_bi_emb)
                users_emb = users_emb[:self.num_users]
                items_emb = items_emb[self.num_users:]
                all_emb = F.normalize(torch.cat([users_emb, items_emb]), p=2, dim=1)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        #light_out = torch.cat(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        all_users = self.u2i_trans(all_users)
        #all_items = self.i2u_trans(all_items)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user[users]
        pos_emb_ego = self.embedding_item[pos_items]
        neg_emb_ego = self.embedding_item[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        users_res_emb = self.i2u_trans(self.u2i_trans(users_emb)) - users_emb
        pos_res_emb = self.u2i_trans(self.i2u_trans(pos_emb)) - pos_emb
        neg_res_emb = self.u2i_trans(self.i2u_trans(neg_emb)) - neg_emb
        reb_loss = (1 / 2) * (users_res_emb.norm(2).pow(2) +
                              pos_res_emb.norm(2).pow(2) +
                              neg_res_emb.norm(2).pow(2)) / float(len(users))
        users_emb = self.u2i_trans(users_emb)
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reb_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = self.u2i_trans(all_users[users])
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class NFAPinSage(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(NFAPinSage, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.user_index, self.item_index = self.dataset.get_index()
        self.user_fea, self.item_fea = self.dataset.get_fea()
        self.user_encoder = FeatureEncoder(self.user_index, self.latent_dim)
        self.item_encoder = FeatureEncoder(self.item_index, self.latent_dim)
        self.embedding_user = self.user_encoder.encode(self.user_fea)
        self.embedding_item = self.item_encoder.encode(self.item_fea)
        self.user_encode_layer = nn.Linear(self.latent_dim * len(self.user_index), self.latent_dim)
        self.item_encode_layer = nn.Linear(self.latent_dim * len(self.item_index), self.latent_dim)
        '''
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        '''
        self.convs_self = nn.ModuleList()
        self.convs_nei = nn.ModuleList()
        for _ in range(self.n_layers):
            conv_self = nn.Linear(2 * self.latent_dim, self.latent_dim)
            self.convs_self.append(conv_self)
            conv_nei = nn.Linear(self.latent_dim, self.latent_dim)
            self.convs_nei.append(conv_nei)
        self.u2i_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.i2u_trans_layer = nn.Linear(self.latent_dim, self.latent_dim)
        '''
        self.u2i_trans_layer.weight = nn.Parameter(
            .01 * torch.randn([self.latent_dim, self.latent_dim]) + torch.eye(self.latent_dim, self.latent_dim))
        self.u2i_trans_layer.weight.requires_grad = False
        '''
        self.f = nn.Sigmoid()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.Graph = self.dataset.getOriginGraph()
        nn.init.normal_(self.u2i_trans_layer.weight, std=0.1)
        nn.init.normal_(self.i2u_trans_layer.weight, std=0.1)
        nn.init.normal_(self.u2i_trans_layer.bias, std=0.1)
        nn.init.normal_(self.i2u_trans_layer.bias, std=0.1)

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def i2u_trans(self, x):
        #return x
        return self.i2u_trans_layer(x)

    def u2i_trans(self, x):
        #return x
        #return self.relu(self.u2i_trans_layer(x))
        return self.u2i_trans_layer(x)

    def computer(self):
        """
        propagate methods for lightGCN
        """

        users_emb = self.relu(self.user_encode_layer(self.embedding_user))
        items_emb = self.relu(self.item_encode_layer(self.embedding_item))
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            u2i_trans_emb = self.u2i_trans(users_emb)
            i2u_trans_emb = self.i2u_trans(items_emb)
            users_all_emb = torch.cat([users_emb, i2u_trans_emb])
            items_all_emb = torch.cat([u2i_trans_emb, items_emb])
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    users_side_emb = torch.sparse.mm(g_droped[f], users_all_emb)
                    items_side_emb = torch.sparse.mm(g_droped[f], items_all_emb)
                    users_nei_emb = self.relu(self.convs_nei[layer](users_side_emb))
                    items_nei_emb = self.relu(self.convs_nei[layer](items_side_emb))
                    users_emb = self.relu(self.convs_self[layer](torch.cat([users_all_emb, users_nei_emb], dim=1)))
                    items_emb = self.relu(self.convs_self[layer](torch.cat([items_all_emb, items_nei_emb], dim=1)))
                    users_emb = users_emb[:self.num_users]
                    items_emb = items_emb[self.num_users:]
                    temp_emb.append(torch.cat([users_emb, items_emb]))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = F.normalize(side_emb, p=2, dim=1)
            else:
                users_side_emb = torch.sparse.mm(g_droped, users_all_emb)
                items_side_emb = torch.sparse.mm(g_droped, items_all_emb)
                users_nei_emb = self.relu(self.convs_nei[layer](users_side_emb))
                items_nei_emb = self.relu(self.convs_nei[layer](items_side_emb))
                users_emb = self.relu(self.convs_self[layer](torch.cat([users_all_emb, users_nei_emb], dim=1)))
                items_emb = self.relu(self.convs_self[layer](torch.cat([items_all_emb, items_nei_emb], dim=1)))
                users_emb = users_emb[:self.num_users]
                items_emb = items_emb[self.num_users:]
                all_emb = F.normalize(torch.cat([users_emb, items_emb]), p=2, dim=1)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        all_users = self.u2i_trans(all_users)
        #all_items = self.i2u_trans(all_items)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user[users]
        pos_emb_ego = self.embedding_item[pos_items]
        neg_emb_ego = self.embedding_item[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        users_res_emb = self.i2u_trans(self.u2i_trans(users_emb)) - users_emb
        pos_res_emb = self.u2i_trans(self.i2u_trans(pos_emb)) - pos_emb
        neg_res_emb = self.u2i_trans(self.i2u_trans(neg_emb)) - neg_emb
        reb_loss = (1 / 2) * (users_res_emb.norm(2).pow(2) +
                              pos_res_emb.norm(2).pow(2) +
                              neg_res_emb.norm(2).pow(2)) / float(len(users))
        users_emb = self.u2i_trans(users_emb)
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reb_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = self.u2i_trans(all_users[users])
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
