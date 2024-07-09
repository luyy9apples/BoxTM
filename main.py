from random import sample
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import itertools
import utils
from data_reader import TextReader

from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

import box_embeddings
from box_embeddings.parameterizations.box_tensor import *
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection

from scipy.sparse import csr_matrix, load_npz

import time
import os
import sys
from tqdm import tqdm

import yaml
import csv

np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
cluster_type = 'ap'

class net(nn.Module):
    def __init__(self, vocab_num, level_num=3, topic_num=50, hidden_num=50, emb_size=50, freeze_word=False, init_word_scale=(1e-4,1e-2,0.9,0.999), init_topic_scale=(1e-4,1e-2,0.9,0.999), intersect_temp=0.0001, volume_temp=0.1, epsilon=1e-6, use_clamp=False):
        super(net, self).__init__()

        self.vocab_num = vocab_num
        self.topic_num = topic_num
        self.hidden_num = hidden_num
        self.emb_size = emb_size
        self.epsilon = epsilon

        self.use_adapt = False
        self.level_num = level_num

        self.volume_temp = volume_temp
        self.inter_temp = intersect_temp
        self.box_vol = Volume(volume_temperature=volume_temp, intersection_temperature=intersect_temp)
        self.box_int = Intersection(intersection_temperature=intersect_temp)

        self.use_clamp = use_clamp
        self.cluster_num = self.level_num - 1
        self.topic_num_list = np.array([topic_num] + [0] * self.cluster_num, dtype=int)
        self.adj_list = [0] * self.cluster_num
        self.p_word_topic_list = [0] * (self.cluster_num + 1)
        self.beta_list = [0] * (self.cluster_num + 1)

        # word box embeddings
        (min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale) = init_word_scale
        x_min = np.random.uniform(min_lower_scale, min_higher_scale, size=(vocab_num, emb_size))
        x_delta = np.random.uniform(delta_lower_scale, delta_higher_scale, size=(vocab_num, emb_size))
        x_max = x_min + x_delta
        self.word_min = nn.Embedding.from_pretrained(torch.tensor(x_min), freeze=freeze_word)
        self.word_max = nn.Embedding.from_pretrained(torch.tensor(x_max), freeze=freeze_word)
        print('hidden size={}'.format(hidden_num))

        (topic_min_lower_scale, topic_min_higher_scale, topic_delta_lower_scale, topic_delta_higher_scale) = init_topic_scale
        y_min = np.random.uniform(topic_min_lower_scale, topic_min_higher_scale, size=(topic_num, emb_size))
        y_delta = np.random.uniform(topic_delta_lower_scale, topic_delta_higher_scale, size=(topic_num, emb_size))
        y_max = y_min + y_delta
        self.topic_min = nn.Embedding.from_pretrained(torch.tensor(y_min), freeze=False)
        self.topic_max = nn.Embedding.from_pretrained(torch.tensor(y_max), freeze=False)

        self.encoder = nn.Sequential(
            nn.Linear(vocab_num, hidden_num),
            nn.Tanh()
        )

        self.l1_mean = nn.Linear(hidden_num, topic_num)
        self.l2_logvar = nn.Linear(hidden_num, topic_num)

        self.decoder = nn.Sequential(
            nn.Linear(topic_num, topic_num),
            nn.Softmax(dim=1)
        )

    def update_topic_parent(self, t_min, t_max):
        self.topic_min = nn.Embedding.from_pretrained(t_min, freeze=False)
        self.topic_max = nn.Embedding.from_pretrained(t_max, freeze=False)
        self.topic_cumsum = np.cumsum(self.topic_num_list)

    def get_word_box(self):
        if self.use_clamp:
            word_min_tensor = torch.clamp(self.word_min.weight.clone(), min=0.0, max=1.0)
            word_max_tensor = torch.clamp(self.word_max.weight.clone(), min=0.0, max=1.0)
        else:
            word_min_tensor = self.word_min.weight.clone()
            word_max_tensor = self.word_max.weight.clone()
        word_box = BoxTensor(torch.stack([word_min_tensor, word_max_tensor], axis=1))
        return word_box

    def get_word_box_ind(self, ind):
        ind_min = self.word_min(ind)
        ind_max = self.word_max(ind)
        if self.use_clamp:
            ind_min = torch.clamp(ind_min, min=0.0, max=1.0)
            ind_max = torch.clamp(ind_max, min=0.0, max=1.0)
        ind_box = BoxTensor(torch.stack([ind_min, ind_max], axis=-2))
        return ind_box

    def get_level_topic_idx(self, level=0):
        if level == 0:
            l_idx = np.arange(0, self.topic_num_list[0])
        else:
            l_idx = np.arange(self.topic_cumsum[level-1], self.topic_cumsum[level])
        l_idx_t = torch.LongTensor(l_idx).to(device)
        return l_idx_t

    def get_topic_min(self, level=0):
        if level == -1:
            topic_min_tensor = self.topic_min.weight.clone()
        else:
            l_idx_t = self.get_level_topic_idx(level=level)
            topic_min_tensor = self.topic_min(l_idx_t)

        return topic_min_tensor

    def get_topic_max(self, level=0):
        if level == -1:
            topic_max_tensor = self.topic_max.weight.clone()
        else:
            l_idx_t = self.get_level_topic_idx(level=level)
            topic_max_tensor = self.topic_max(l_idx_t)

        return topic_max_tensor

    def get_topic_box_ind(self, ind, level=0):
        l_idx_t = self.get_level_topic_idx(level=level)
        l_ind = l_idx_t[ind]

        topic_min_tensor = self.topic_min(l_ind)
        topic_max_tensor = self.topic_max(l_ind)

        if self.use_clamp:
            topic_min_tensor = torch.clamp(topic_min_tensor, min=0.0, max=1.0)
            topic_max_tensor = torch.clamp(topic_max_tensor, min=0.0, max=1.0)
            
        topic_box = BoxTensor(torch.stack([topic_min_tensor, topic_max_tensor], axis=1))
        return topic_box

    def get_topic_box(self, level=0):
        topic_min_tensor = self.get_topic_min(level=level)
        topic_max_tensor = self.get_topic_max(level=level)

        if self.use_clamp:
            topic_min_tensor = torch.clamp(topic_min_tensor, min=0.0, max=1.0)
            topic_max_tensor = torch.clamp(topic_max_tensor, min=0.0, max=1.0)

        topic_box = BoxTensor(torch.stack([topic_min_tensor, topic_max_tensor], axis=1))
        return topic_box

    def get_p_word_all(self):
        word_box = self.get_word_box()
        p_word = self.box_vol(word_box)
        return p_word

    def get_p_topic(self, level=0):
        topic_box = self.get_topic_box(level=level)
        p_topic = self.box_vol(topic_box)
        return p_topic

    def get_word_delta(self):
        word_min_tensor = self.word_min.weight.clone()
        word_max_tensor = self.word_max.weight.clone()
        if self.use_clamp:
            word_min_tensor = torch.clamp(word_min_tensor, min=0.0, max=1.0)
            word_max_tensor = torch.clamp(word_max_tensor, min=0.0, max=1.0)
        # word_delta = word_max_tensor - word_min_tensor
        euler_gamma = box_embeddings.common.constant.EULER_GAMMA
        word_delta = F.softplus(word_max_tensor - word_min_tensor - 2*euler_gamma*self.inter_temp, beta=1/self.volume_temp)
        return word_delta

    def get_topic_delta(self, level=0):
        topic_min_tensor = self.get_topic_min(level=level)
        topic_max_tensor = self.get_topic_max(level=level)
        # TODO
        # topic_delta = topic_max_tensor - topic_min_tensor
        euler_gamma = box_embeddings.common.constant.EULER_GAMMA
        topic_delta = F.softplus(topic_max_tensor - topic_min_tensor - 2*euler_gamma*self.inter_temp, beta=1/self.volume_temp)
        return topic_delta

    def get_word_global(self):
        word_global_min = torch.min(self.word_min.weight.clone(), dim=0)
        word_global_max = torch.max(self.word_max.weight.clone(), dim=0)
        return word_global_min, word_global_max

    def get_topic_global(self):
        topic_global_min = torch.min(self.topic_min.weight.clone(), dim=0)
        topic_global_max = torch.max(self.topic_max.weight.clone(), dim=0)
        return topic_global_min, topic_global_max

    def get_p_cond_word(self, pos, sample):
        sample_box = self.get_word_box_ind(sample)

        pos_min = self.word_min(pos).squeeze(0)
        pos_max = self.word_max(pos).squeeze(0)
        if self.use_clamp:
            pos_min = torch.clamp(pos_min, min=0.0, max=1.0)
            pos_max = torch.clamp(pos_max, min=0.0, max=1.0)
        pos_box = BoxTensor([pos_min, pos_max])

        inter = self.box_int(pos_box, sample_box)
        ln_inter_v = self.box_vol(inter)
        ln_p_word_pos = ln_inter_v
        return ln_p_word_pos

    def get_p_word_inter(self, c_ind_list, p_ind_list):
        c_box = self.get_word_box_ind(c_ind_list)
        p_box = self.get_word_box_ind(p_ind_list)

        inter = self.box_int(c_box, p_box)
        ln_inter_v = self.box_vol(inter)
        return ln_inter_v

    def get_p_word(self, ind_list):
        i_box = self.get_word_box_ind(ind_list)
        i_v = self.box_vol(i_box)
        return i_v

    def get_p_parent_child_inter(self, level=0):
        p_level = level + 1
        p_t_box = self.get_topic_box(level=p_level)
        t_box = self.get_topic_box(level=level)

        p_t_box.broadcast(target_shape=(self.topic_num_list[p_level], 1, self.emb_size))
        t_box.broadcast(target_shape=(1, self.topic_num_list[level], self.emb_size))

        inter = self.box_int(p_t_box, t_box)
        ln_inter_v = self.box_vol(inter)
        return ln_inter_v


    def get_p_child_cond_parent(self, level=0):
        ln_p_parent_child_inter = self.get_p_parent_child_inter(level=level)
        p_level = level + 1
        ln_p_parent = torch.transpose(self.get_p_topic(level=p_level).unsqueeze(0), 0, 1)
        ln_p_cond = ln_p_parent_child_inter - ln_p_parent
        return ln_p_cond

    def with_topk_w(self, topic_min, topic_max, word_box, ln_word_v):
        topic_box = BoxTensor(torch.stack([topic_min, topic_max], axis=1))
        ln_topic_v = self.box_vol(topic_box)
        topic_num = topic_min.shape[0]
        topic_box.broadcast(target_shape=(topic_num, 1, self.emb_size))

        inter = self.box_int(topic_box, word_box)
        ln_inter_v = self.box_vol(inter)
        
        p_word_topic = ln_inter_v - ln_word_v - ln_topic_v.unsqueeze(1)

        topk = 5 
        _, indices = torch.topk(p_word_topic, topk, dim=1)
    
        topk_w_box = self.get_word_box_ind(indices)

        # ignore
        ln_v = self.box_vol(topk_w_box)
        v_thresh = 0.1 * torch.exp(torch.max(ln_word_v))
        (ignore_row, ignore_col) = (torch.exp(ln_v) >= v_thresh).nonzero(as_tuple=True)

        topk_w_min = topk_w_box.z
        topk_w_max = topk_w_box.Z

        topk_w_min[ignore_row, ignore_col, :] = 1.0
        topk_w_max[ignore_row, ignore_col, :] = 0.0
        ###

        t_w_min = torch.cat([topic_min.unsqueeze(1), topk_w_min], dim=1)
        t_w_max = torch.cat([topic_max.unsqueeze(1), topk_w_max], dim=1)

        t_w_min = torch.min(t_w_min, dim=1).values
        t_w_max = torch.max(t_w_max, dim=1).values

        # t_w_min = torch.mean(t_w_min, dim=1)
        # t_w_max = torch.mean(t_w_max, dim=1)

        return t_w_min, t_w_max

    def build_adj_cluster(self):

        topic_min_l0 = self.get_topic_min(level=0)
        topic_max_l0 = self.get_topic_max(level=0)

        topic_min = topic_min_l0
        topic_max = topic_max_l0

        ###
        word_box = self.get_word_box()
        ln_word_v = self.box_vol(word_box)
        word_box.broadcast(target_shape=(1, self.vocab_num, self.emb_size))

        t_w_min, t_w_max = self.with_topk_w(topic_min, topic_max, word_box, ln_word_v)
        ### 
        
        dim_num = topic_min.shape[0]
        child_num = self.topic_num_list[0]
        self.topic_num_list = [child_num]

        topic_min_new = [topic_min_l0]
        topic_max_new = [topic_max_l0]
        
        while True:
            l = len(self.topic_num_list) - 1

            t_w_min_max = torch.stack([t_w_min, t_w_max], axis=1)
            t_w_box = BoxTensor(t_w_min_max)

            re_t_w_min_max = t_w_min_max.repeat(1, dim_num, 1).reshape(dim_num, dim_num, 2, self.emb_size)
            re_t_w_box = BoxTensor(re_t_w_min_max)

            inter = self.box_int(re_t_w_box, t_w_box)
            ln_inter_v = self.box_vol(inter)
            
            ln_v = self.box_vol(t_w_box)

            # ignore general words
            v_thresh = 0.1
            ignore_indices = (torch.exp(ln_v).squeeze(0) >= v_thresh).nonzero(as_tuple=True)[0]

            mtx = ln_inter_v - ln_v

            # coefficient of variation
            std, m = torch.std_mean(mtx, dim=1, unbiased=False)
            cv = torch.abs(std / (m + self.epsilon))
            mtx = mtx * cv.unsqueeze(1)

            # AP
            dist_mtx = torch.exp(mtx) # affinity
            # ignore general words
            dist_mtx[ignore_indices,:] = 0.0
            if cluster_type == 'hier': # dist / DBSCAN
                dist_mtx = 1.0 - dist_mtx

            diag = torch.diag(dist_mtx)
            diag_mtx = torch.diag_embed(diag)
            dist_mtx = dist_mtx - diag_mtx
            
            t_dist_mtx = dist_mtx.detach().cpu().numpy()

            if cluster_type == 'ap':
                clustering = AffinityPropagation(random_state=5, affinity='precomputed').fit(t_dist_mtx)
            elif cluster_type == 'hier':
                clustering = AgglomerativeClustering(n_clusters=20, affinity='precomputed', linkage='complete').fit(t_dist_mtx)
            clustering_label = clustering.labels_
            # self.center_topics = clustering.cluster_centers_indices_
            parent_num = np.unique(clustering_label[:child_num]).shape[0] + max(np.count_nonzero(clustering_label[:child_num] == -1)-1, 0)
            adj_np = np.zeros((child_num, parent_num), dtype=np.float64)

            p_c_indices = []

            pid_it_dict = {}
            pid_it = 0
            for cid, pid in enumerate(clustering_label):
                if cid >= child_num: break
                if pid > -1:
                    if pid not in pid_it_dict:
                        pid_it_dict[pid] = pid_it
                        pid_it += 1
                        # with topk words
                        p_c_ind = np.nonzero(clustering_label == pid)[0]
                        p_c_indices.append(p_c_ind)
                else:
                    p_c_indices.append([cid])
                    pid_it += 1

            for pid in range(parent_num):
                for cid in p_c_indices[pid]:
                    if cid < child_num: adj_np[cid, pid] = 1.0

            topic_min_p_l = []
            topic_max_p_l = []
            for indices in p_c_indices:
                indices_t = torch.LongTensor(indices).to(device)
                t_min_c = t_w_min[indices_t]
                t_max_c = t_w_max[indices_t]
                
                # t_min_p = torch.min(t_min_c, dim=0).values
                # t_max_p = torch.max(t_max_c, dim=0).values

                t_min_p = torch.mean(t_min_c, dim=0)
                t_max_p = torch.mean(t_max_c, dim=0)

                topic_min_p_l.append(t_min_p)
                topic_max_p_l.append(t_max_p)

            topic_min_p = torch.vstack(topic_min_p_l)
            topic_max_p = torch.vstack(topic_max_p_l)

            self.topic_num_list.append(parent_num)

            adj_l = torch.tensor(adj_np, dtype=torch.float32, requires_grad=False).to(device)
            self.adj_list[l] = adj_l

            topic_min_new.append(topic_min_p)
            topic_max_new.append(topic_max_p)

            child_num = parent_num
            dim_num = parent_num

            t_w_min, t_w_max = self.with_topk_w(topic_min_p, topic_max_p, word_box, ln_word_v)

            if not self.use_adapt and len(self.topic_num_list) == self.level_num:
                break

            if self.use_adapt and parent_num <= 10:
                break

        topic_min_new_t = torch.vstack(topic_min_new)
        topic_max_new_t = torch.vstack(topic_max_new)

        self.level_num = len(self.topic_num_list)
        self.cluster_num = self.level_num - 1

        return topic_min_new_t, topic_max_new_t


    def build_adj(self, IV):
        self.cv_select_indices = {}
        for level in range(self.cluster_num):
            ln_p_parent_child_inter = IV['level_ln_pci_v'][level]
            p_level = level + 1
            ln_p_parent = torch.transpose(IV['level_ln_topic_v'][p_level].unsqueeze(0), 0, 1)
            ln_p_cond = ln_p_parent_child_inter - ln_p_parent
            _, inds = torch.topk(ln_p_cond, k=1, dim=0)
            
            if level == 0:
                std, mean = torch.std_mean(ln_p_cond, dim=0)
                cv = torch.abs(std / (mean + self.epsilon))
                cv_thresh = torch.max(cv) * 0.25
                self.cv_select_indices[level] = (cv > cv_thresh).nonzero().squeeze()
            elif level > 0:
                root_p_num = torch.sum(self.adj_list[level-1], dim=0)
                select_root_indices = root_p_num.nonzero().squeeze(1)
                self.cv_select_indices[level] = select_root_indices

            inds = inds.squeeze(0)
            cols = torch.transpose(inds[self.cv_select_indices[level]].unsqueeze(0), 0, 1)
            rows = torch.transpose(self.cv_select_indices[level].unsqueeze(0), 0, 1)

            new_adj = torch.zeros(self.topic_num_list[level], self.topic_num_list[p_level], requires_grad=False).float()
            new_adj[rows, cols] = 1.0
            self.adj_list[level] = new_adj
        level += 1
        root_p_num = torch.sum(self.adj_list[level-1], dim=0)
        select_root_indices = root_p_num.nonzero().squeeze(1)
        self.cv_select_indices[level] = select_root_indices


    def get_uq_topic_word(self, IV, topk=15):
        topic_dist_list = []
        p_word_topic_list = []
        topic_dist_c_list = []
        l_tw_indices_set = []
        l_ptw_thresh = []
        l_ptw_std = []

        p_word = IV['ln_word_v'].detach()
        for l in range(self.level_num):
            topic_dist, p_word_topic = self.get_beta(IV, level=l)
            if len(topic_dist.shape) < 2: topic_dist = topic_dist.unsqueeze(0)
            
            select_root_indices = self.cv_select_indices[l]
            topic_dist_c = topic_dist[select_root_indices,:].detach()

            if len(topic_dist_c.shape) < 2: topic_dist_c = topic_dist_c.unsqueeze(0)
            topic_dist_list.append(topic_dist)
            p_word_topic_list.append(p_word_topic)
            topic_dist_c_list.append(topic_dist_c)

            _, indices = torch.topk(topic_dist_c, topk*2, dim=1)
            indices = torch.unique(indices.flatten())
            indices_set = torch.zeros(p_word.shape[0]).to(device)
            indices_set[indices] = 1.0
            l_tw_indices_set.append(indices_set)
            l_ptw_thresh.append(torch.mean(p_word[indices]).item())
            l_ptw_std.append(torch.std(p_word[indices]).item())

        in_w_indices = {l:{} for l in range(self.level_num)}
        not_w_i_list = {l:{} for l in range(self.level_num)}
        for i in range(self.cluster_num):
            p_l = self.cluster_num - i
            union_w_indices = torch.zeros(p_word.shape[0]).to(device)
            for c_l in range(p_l):
                u_ind_set = l_tw_indices_set[p_l] + l_tw_indices_set[c_l]
                c_u_w_indices = torch.nonzero(u_ind_set == 2.0, as_tuple=True)[0]

                c_u_w_ind_set = torch.zeros(p_word.shape[0]).to(device)
                c_u_w_ind_set[c_u_w_indices] = 1.0

                union_w_indices = union_w_indices + c_u_w_ind_set
            union_w_indices_arr = torch.nonzero(union_w_indices > 0, as_tuple=True)[0]

            if len(union_w_indices_arr) == 0:
                in_w_indices[p_l] = l_tw_indices_set[p_l]
            else:
                p_union_w = p_word[union_w_indices_arr]

                not_w_i = torch.nonzero(p_union_w < l_ptw_thresh[p_l], as_tuple=True)[0]
                not_w_indices = union_w_indices_arr[not_w_i]
                not_w_i_list[p_l] = not_w_indices.detach().cpu().numpy().tolist()

                not_w_set = torch.zeros(p_word.shape[0]).to(device)
                not_w_set[not_w_indices] = 1.0

                in_w_indices[p_l] = l_tw_indices_set[p_l] - not_w_set

            for c_l in range(p_l):
                l_tw_indices_set[c_l] = l_tw_indices_set[c_l] - in_w_indices[p_l]
                l_tw_indices_set[c_l] = (l_tw_indices_set[c_l] > 0).float()

        in_w_indices[0] = l_tw_indices_set[0]

        # edit topic dist
        uq_topic_words = {l:[] for l in range(self.level_num)}
        topic_words = {l:[] for l in range(self.level_num)}
        for l in range(self.level_num):
            _, indices = torch.topk(topic_dist_list[l], topk*2, dim=1)
            indices = indices.detach().cpu().numpy().tolist()
            topic_words[l] = indices
            min_thresh = l_ptw_thresh[l] - 2 * l_ptw_std[l]
            max_thresh = l_ptw_thresh[l] + 2 * l_ptw_std[l]
            uq_topic_words[l] = [[wid for wid in ind if in_w_indices[l][wid] > 0 and p_word[wid] > min_thresh and p_word[wid] < max_thresh] for ind in indices]
            uq_topic_words[l] = [uq_tw[:topk] for uq_tw in uq_topic_words[l]]

        return uq_topic_words, topic_words, (l_ptw_thresh, l_ptw_std, not_w_i_list)


    def get_beta(self, IV, level=0):
        ln_word_v = IV['ln_word_v']
        ln_topic_v = IV['level_ln_topic_v'][level]
        ln_inter_v = IV['level_ln_twi_v'][level]

        p_word_topic = ln_inter_v - ln_word_v - ln_topic_v.unsqueeze(1)
        beta = torch.softmax(p_word_topic, dim=1).float()

        std, m = torch.std_mean(beta, dim=0, unbiased=False)
        cv = torch.abs(std / (m + self.epsilon))
        cv = cv.detach()
        beta = beta * cv
        return beta, p_word_topic

    def infer(self, x):
        mean, logvar = self.encode(x)
        h = self.reparametrization(mean, logvar)
        return self.decoder(h)

    def encode(self, x):
        pi = self.encoder(x)
        return self.l1_mean(pi), self.l2_logvar(pi)

    def reparametrization(self, mu, logvar):
        std = 0.5 * torch.exp(logvar)
        eps = torch.randn(std.size()).to(device)

        # N(mu, std^2) = N(0, 1) * std + mu
        z = eps * std + mu
        return z

    def recon(self, IV, theta_list, level):
        beta, _ = self.get_beta(IV, level=level)
        d_e = theta_list[level] @ beta
        # with norm
        n = torch.transpose(torch.norm(d_e, dim=1).unsqueeze(0), 0, 1).detach()
        d_norm = d_e / (n + self.epsilon)
        d_l = torch.log( d_norm + self.epsilon )
        return d_l

    def decode(self, z, IV):
        theta = self.decoder(z)
        theta_list = [theta]
        m = nn.Softmax(dim=1)
        for l in range(self.cluster_num):
            ln_p_parent_child_inter = IV['level_ln_pci_v'][l]
            p_level = l + 1
            ln_p_parent = torch.transpose(IV['level_ln_topic_v'][p_level].unsqueeze(0), 0, 1)
            ln_p_cond = ln_p_parent_child_inter - ln_p_parent
            ln_p_cond = ln_p_cond.float()
            theta_l = m(theta_list[-1] @ (2.0 * torch.transpose(ln_p_cond, 0, 1)))
            theta_list.append(theta_l)

        d_list = list(itertools.chain(map(lambda l: self.recon(IV, theta_list, l), range(self.level_num))))
        d_tensor = torch.stack(d_list, dim=0)
        d = torch.sum(d_tensor, dim=0)
        return d, theta_list

    def cal_med_var(self):
        # Volume of word boxes
        ln_word_v = self.get_p_word_all()
        # Volume of topic boxes
        level_ln_topic_v = list(itertools.chain(map(lambda l: self.get_p_topic(level=l), range(self.level_num))))
        # Volume of parent-child inter boxes
        level_ln_pci_v = list(itertools.chain(map(lambda l: self.get_p_parent_child_inter(level=l), range(self.cluster_num))))
        
        # Volume of topic-word inter boxes
        word_box = self.get_word_box()
        word_box.broadcast(target_shape=(1, self.vocab_num, self.emb_size))

        level_topic_box = list(itertools.chain(map(lambda l: self.get_topic_box(level=l), range(self.level_num))))
        for l in range(self.level_num):
            level_topic_box[l].broadcast(target_shape=(self.topic_num_list[l], 1, self.emb_size))
        
        level_ln_twi_v = list(itertools.chain(map(lambda topic_box:self.box_vol(self.box_int(topic_box, word_box)), level_topic_box)))

        IV = {}
        IV['ln_word_v'] = ln_word_v
        IV['level_ln_topic_v'] = level_ln_topic_v
        IV['level_ln_pci_v'] = level_ln_pci_v
        IV['level_ln_twi_v'] = level_ln_twi_v
        return IV

    def forward(self, x, IV):
        mean, logvar = self.encode(x)
        z = self.reparametrization(mean, logvar)
        x_hat, _ = self.decode(z, IV)
        return x_hat, mean, logvar


class BoxTM(object):
    def __init__(self, reader=None, level_num=3, topic_num=50, hidden_num=50, emb_size=50, model_path='./', save_name='model.pkl', freeze_word=False, init_word_scale=(1e-4,1e-2,0.9,0.999), init_topic_scale=(1e-4,1e-2,0.9,0.999), intersect_temp=0.0001, volume_temp=0.1, learning_rate=5e-3, r1=0.5, r2=0.5, w1=5e-3, sample_size=128, max_w2=0.05, max_r3=0.05, use_clamp=False, coor_path=None):
        if reader == None:
            raise Exception(" [!] Expected data reader")

        self.vocab_num = reader.vocab_size
        self.hidden_num = hidden_num
        self.reader = reader
        self.model_path = model_path
        self.save_name = save_name

        print("BoxTM init model.")
        self.Net = net(self.vocab_num, level_num, topic_num, hidden_num, emb_size, freeze_word, init_word_scale, init_topic_scale, intersect_temp, volume_temp, use_clamp=use_clamp).to(device)

        print(self.Net)

        if coor_path:
            self.use_coor = True
            self._build_coor_samples(coor_path)
        else:
            self.use_coor = False

        self.learning_rate = learning_rate
        self.topic_num = topic_num
        self.sample_size = sample_size

        self.r1 = r1
        self.r2 = r2
        self.max_r3 = max_r3

        self.w1 = w1
        self.max_w2 = max_w2

        self.epsilon = 1e-6

        # optimizer uses ADAM
        self.optimizer = optim.Adam(self.Net.parameters(), lr=self.learning_rate)

    def _build_coor_samples(self, coor_path):
        coor_mtx = load_npz(coor_path)
        coor_mtx += coor_mtx.transpose()
        self.coor_wor = coor_mtx.sum(axis=1).A.flatten()
        self.coor_tars, self.coor_ctxs = coor_mtx.nonzero()
        self.coor_data = coor_mtx.data

    def _compute_word_box_loss(self, IV):
        word_unary, topic_unary = self._compute_unary_loss()

        tw_loss = self._compute_topic_word_loss(IV, top_k=5)

        word_box_loss = self.r1 * word_unary + self.r2 * topic_unary + self.r3 * tw_loss

        return word_box_loss


    def compute_box_loss(self, IV):
        if self.use_coor:
            coor_loss = self._compute_coor_loss(IV, sample_size=self.sample_size)
        else:
            coor_loss = 0.

        parent_inter_loss, child_v_loss = self._compute_parent_loss(IV)
        ht_loss = parent_inter_loss + child_v_loss

        word_box_loss = self._compute_word_box_loss(IV)

        box_loss = self.w1 * coor_loss + self.w2 * ht_loss + word_box_loss

        return box_loss


    def compute_loss(self, x, y, mean, logvar, IV):
        # reconstruct loss
        likelihood = -torch.sum(y * x, dim=1)

        # kl divergence
        kld = -0.5 * torch.sum(1 - torch.square(mean) + logvar - torch.exp(logvar), dim=1)

        elbo_loss = likelihood + kld

        # box loss
        box_loss = self.compute_box_loss(IV)

        return likelihood, kld, elbo_loss, box_loss

    def _compute_coor_loss(self, IV, sample_size=1024):
        total_sample_size = len(self.coor_data)
        ind = np.random.choice(total_sample_size, sample_size)

        batch_targets = torch.LongTensor(self.coor_tars[ind]).to(device)
        batch_contexts = torch.LongTensor(self.coor_ctxs[ind]).to(device)

        batch_data = torch.tensor(self.coor_data[ind], requires_grad=False).to(device)
        batch_wor = torch.tensor(self.coor_wor[self.coor_tars[ind]], requires_grad=False).to(device)
        batch_coor = torch.log(1.0 + batch_data) - torch.log(1.0 + batch_wor)

        sample_tc = self.Net.get_p_cond_word(batch_targets, batch_contexts)
        sample_t = IV['ln_word_v'][self.coor_tars[ind]]
        pred_coor = sample_tc - sample_t

        p = F.log_softmax(batch_coor, dim=0)
        q = F.log_softmax(pred_coor, dim=0)
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        coor_loss = kl_loss(q.unsqueeze(0), p.unsqueeze(0))
        return coor_loss


    def _compute_parent_loss(self, IV):
        zero = torch.tensor(0.0, requires_grad=False).to(device)
        inter_loss_list = [zero]
        v_loss_list = [zero]

        ln_p_child = IV['level_ln_topic_v'][0]
        for l in range(self.Net.cluster_num):
            inds = self.Net.adj_list[l].nonzero()
            if len(inds.shape) != 2 or inds.shape[0] <= 0 or inds.shape[1] <= 0: continue
            rows = torch.transpose(inds[:,1].unsqueeze(0), 0, 1)
            cols = torch.transpose(inds[:,0].unsqueeze(0), 0, 1)

            ln_p_parent_child_inter = IV['level_ln_pci_v'][l]
            ln_p_parent_child_inter = ln_p_parent_child_inter[rows, cols]

            inter_loss_l = -torch.mean(ln_p_parent_child_inter)
            inter_loss_list.append(inter_loss_l)

            p_l = l + 1
            ln_p_parent = IV['level_ln_topic_v'][p_l]
            v_loss_l = torch.mean(torch.max(zero, 10.0 - ln_p_parent[rows] + ln_p_child[cols]))
            v_loss_list.append(v_loss_l)

            ln_p_child = ln_p_parent
        
        inter_loss = torch.max(torch.sum(torch.stack(inter_loss_list, dim=0)), zero)
        v_loss = torch.sum(torch.stack(v_loss_list, dim=0))

        return ( inter_loss, v_loss )


    def _compute_topic_word_loss(self, IV, top_k=5):
        p_word_topic_list = []
        for l in range(self.Net.level_num):
            _, l_p_word_topic = self.Net.get_beta(IV, level=l)
            p_word_topic_list.append(l_p_word_topic)
        p_word_topic = torch.cat(p_word_topic_list, dim=0)
        pos_val, _ = torch.topk(p_word_topic, top_k, dim=1)
        neg_val, _ = torch.topk(p_word_topic, top_k, dim=1, largest=False)

        zero = torch.tensor(0.0, requires_grad=False).to(device)
        pos_neg_loss = torch.mean(torch.max(zero, 10.0 - pos_val + neg_val))
        return pos_neg_loss


    def _compute_unary_loss(self, unary_method='delta'):
        if unary_method == 'universe':
            word_global_min, word_global_max = self.Net.get_word_global()
            topic_global_min, topic_global_max = self.Net.get_topic_global()
            word_unary = torch.mean(torch.softplus(word_global_max - word_global_min))
            topic_unary = torch.mean(torch.softplus(topic_global_max - topic_global_min))
        elif unary_method == 'delta':
            word_delta = self.Net.get_word_delta()
            topic_delta = self.Net.get_topic_delta()
            word_unary = torch.mean(torch.square(word_delta))
            topic_unary = torch.mean(torch.square(topic_delta))
        return word_unary, topic_unary

    def save_params(self, savename):
        params = {'topic_num_list': self.Net.topic_num_list}
        savename = '.'.join(savename.split('.')[:-1])
        savepath = savename+'.params.npy'
        np.save(savepath, params)

    def load_params(self, savename):
        savename = '.'.join(savename.split('.')[:-1])
        savepath = savename+'.params.npy'
        params = np.load(savepath, allow_pickle=True).item()
        self.Net.topic_num_list = params['topic_num_list']

        self.Net.level_num = len(params['topic_num_list'])
        self.Net.cluster_num = self.Net.level_num - 1

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self, typen):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if typen != 'child':
            torch.save(self.Net.state_dict(), f'{self.model_path}/{typen}_{self.save_name}')
            self.save_params(f'{self.model_path}/{typen}_{self.save_name}')
            print(f'Models save to  {self.model_path}/{typen}_{self.save_name}')
        else:
            torch.save(self.Net.state_dict(), f'{self.model_path}/{self.save_name}')
            self.save_params(f'{self.model_path}/{self.save_name}')
            print(f'Models save to  {self.model_path}/{self.save_name}')

    def load_model(self, typen):
        if typen != 'child':
            model_path = os.path.join(self.model_path, typen+'_'+self.save_name)
        else: model_path = os.path.join(self.model_path, self.save_name)
        m_state_dict = torch.load(model_path, map_location=device)
        
        params_path = model_path
        self.load_params(params_path)
        self.Net.update_topic_parent(m_state_dict['topic_min.weight'].to(device), m_state_dict['topic_max.weight'].to(device))

        self.Net.load_state_dict(m_state_dict)

        print('BoxTM model loaded from {}.'.format(model_path))

    def get_word_topic(self, data, vocab=None):
        word_topic = self.Net.infer(torch.from_numpy(data).to(device))
        word_topic = self.to_np(word_topic)
        return word_topic

    def get_topic_dist(self):
        topic_dist, _ = self.Net.get_beta(level=0)
        return topic_dist
   
    def get_topic_word(self, IV, top_k=15, vocab=None, level=0):
        topic_dist, _ = self.Net.get_beta(IV, level=level)
        vals, indices = torch.topk(topic_dist, top_k, dim=1)
        indices = self.to_np(indices).tolist()
        topic_words = [[vocab[idx] for idx in indices[i]] for i in range(topic_dist.shape[0])]
        return topic_words, indices


    def evaluate(self, IV, uq=False, top_k=15):
        start = 0
        uq_topic_word, ori_topic_word, _ = self.Net.get_uq_topic_word(IV)
        for l in range(start, self.Net.level_num):
            if uq: topic_word = [[self.reader.vocab[wid] for wid in tw[:top_k]] for tw in uq_topic_word[l]]
            else: topic_word = [[self.reader.vocab[wid] for wid in tw[:top_k]] for tw in ori_topic_word[l]]
            
            select_root_indices = self.to_np(self.Net.cv_select_indices[l])
            
            print(f'[Level-{l} Topic]')
            if l == 0:
                for k, top_word_k in enumerate(topic_word):
                    if k not in select_root_indices: continue
                    print(f'Topic {k}:{top_word_k}')
            else:
                adj = self.Net.adj_list[l-1].detach().cpu().numpy()
                for k, top_word_k in enumerate(topic_word):
                    c_indices = np.nonzero(adj[:,k])[0].tolist()
                    if len(c_indices) > 0:
                        print(f'Topic {k} {c_indices}:{top_word_k}')
        return 0, 0


    # get topic_word and print Top word
    def sampling(self, IV, uq=False, datatype='test'):
        start = 0
        topk = 10
        test_data, test_label , _ = self.reader.get_matrix(datatype, mode='count')
        topic_dist_list = []
        topic_dist_c_list = []
        coherence_list = []
        coh_log = []

        uq_topic_word, ori_topic_word, _ = self.Net.get_uq_topic_word(IV)
        for l in range(self.Net.level_num):
            if uq: topic_dist = uq_topic_word[l]
            else: topic_dist = ori_topic_word[l]
            
            select_root_indices = self.to_np(self.Net.cv_select_indices[l])
            t_num = select_root_indices.shape[0]
            topic_dist_c = [topic_dist[idx] for idx in select_root_indices if len(topic_dist[idx]) > 1]

            topic_dist_list.append(topic_dist)
            topic_dist_c_list.append(topic_dist_c)

            coherence = utils.compute_coherence(test_data, topic_dist_c, topk)
            coherence_list.append(coherence)

            coh_log.append(f'[Level-{l}({t_num})] {coherence:.4f}')

        all_topic_dist = []
        for tdc in topic_dist_c_list: all_topic_dist.extend(tdc)
        coherence_all = utils.compute_coherence(test_data, all_topic_dist, topk)
        coh_log_str = " ".join(coh_log)
        print(f'Topic coherence: {coh_log_str} [all] {coherence_all:.4f}')
        
        if coherence_all > self.best_coherence_all:
            self.best_coherence_all = coherence_all
            print("New best coherence found!!")
            self.save_model('all')

        if coherence_list[-1] > self.best_coherence_root:
            self.best_coherence_root = coherence_list[-1]
            print("New best coherence found!!")
            self.save_model('root')
        
        if coherence_list[start] > self.best_coherence_child:
            self.best_coherence_child = coherence_list[start]
            print("New best coherence found!!")
            self.save_model('child')

        # metrics for hierachical topic structure
        clnpmi = []
        clnpmi_log = []
        
        for l in range(self.Net.cluster_num):
            clnpmi_tmp = []

            select_root_indices = self.to_np(self.Net.cv_select_indices[l])
            t_num = select_root_indices.shape[0]
            c_indices = select_root_indices
            
            for c_idx in c_indices:
                p_idx = self.to_np(self.Net.adj_list[l][c_idx,:].nonzero().squeeze())
                c_topic = topic_dist_list[l][c_idx]
                p_topic = topic_dist_list[l+1][p_idx]
                if len(c_topic) == 0 or len(p_topic) == 0: continue
                clnpmi_tmp.append(utils.compute_clnpmi(c_topic, p_topic, test_data))

            clnpmi_mean = np.mean(clnpmi_tmp)
            clnpmi.append(np.array(clnpmi_tmp))
            clnpmi_log.append(f'[Level-{l}({t_num})] {clnpmi_mean:.4f}')
        
        clnpmi_log_str = " ".join(clnpmi_log)
        print(f'clnpmi: {clnpmi_log_str} [all] {np.mean([d for l in clnpmi[start:] for d in l]):.4f}')

        return coherence_all

    def average_sampling(self, IV, uq=False):
        start = 0
        test_data, _ , _ = self.reader.get_matrix('test', mode='count')
        topic_dist_list = []
        topic_dist_c_list = []
        coh_log = []
        TU_log = []
        topic_word_list = {N:[] for N in [5, 10, 15]}

        uq_topic_word, ori_topic_word, _ = self.Net.get_uq_topic_word(IV)
        root_p_num = torch.sum(self.Net.adj_list[self.Net.cluster_num-1], dim=0)
        select_root_indices = self.to_np(root_p_num.nonzero().squeeze())
        for l in range(start, self.Net.cluster_num+1):
            if uq: topic_dist = uq_topic_word[l]
            else: topic_dist = ori_topic_word[l]

            select_root_indices = self.to_np(self.Net.cv_select_indices[l])
            t_num = select_root_indices.shape[0]

            topic_dist = [topic_dist[idx] for idx in select_root_indices if len(topic_dist[idx]) > 0]
            topic_dist_c = [td for td in topic_dist if len(td) > 1]

            topic_dist_list.append(topic_dist)
            topic_dist_c_list.append(topic_dist_c)

            coherence = 0.
            TU = 0.
            for N in [5, 10, 15]:
                coherence += utils.compute_coherence(test_data, topic_dist_c, N)
                topic_word = [tw[:N] for tw in topic_dist]

                topic_word_list[N].extend(topic_word)
                TU += utils.evaluate_topic_diversity(topic_word)
            
            coherence /= 3.0
            TU /= 3.0

            coh_log.append(f'[Level-{l}({t_num})] {coherence:.4f}')
            TU_log.append(f'[Level-{l}({t_num})] {TU:.4f}')

        all_topic_dist = []
        for tdc in topic_dist_c_list: all_topic_dist.extend(tdc)
        coherence = 0.
        TU = 0.
        for N in [5, 10, 15]:
            coherence += utils.compute_coherence(test_data, all_topic_dist, N)
            topic_word = topic_word_list[N]
            TU += utils.evaluate_topic_diversity(topic_word)
        
        coherence /= 3.0
        TU /= 3.0

        coh_log_str = ' '.join(coh_log)
        print(f"Ave Topic coherence: {coh_log_str} [all] {coherence:.4f}")
        TU_log_str = ' '.join(TU_log)
        print(f"Ave TU: {TU_log_str} [all] {TU:.4f}")
        
        return coherence



    def get_batches(self, batch_size=512, rand=True):
        self.train_data = self.train_data
        self.train_label = self.train_label

        count = 0
        while True:
            if not rand:
                beg = (count * batch_size) % self.train_data.shape[0]
                end = ((count + 1) * batch_size) % self.train_data.shape[0]
                if beg > end:
                    beg -= self.train_data.shape[0]

                idx = np.arange(beg, end)
            else:
                idx = np.random.randint(0, self.train_data.shape[0], batch_size)

            data = self.train_data[idx].toarray()
            data = torch.from_numpy(data).to(device)
            yield data


    def train(self, epochs=1000, batch_size=512, data_type='train', input_type='tfidf'):
        self.t_begin = time.time()
        self.batch_size = batch_size
        self.train_data, self.train_label, self.train_text = self.reader.get_sparse_matrix(data_type, mode=input_type) # mode='tfidf'/'count'

        self.train_generator = self.get_batches(batch_size)
        data_size = self.train_data.shape[0]
        n_batchs = data_size // batch_size
        
        self.best_coherence_all = -1
        self.best_coherence_child = -1
        self.best_coherence_root = -1
        ones = torch.ones(batch_size).to(device)
        loss_all = []

        max_r3 = self.max_r3
        max_w2 = self.max_w2

        ep_n = 0
        ep_thresh = pow(1.2, ep_n)
        epoch_m = 100
        
        for epoch in tqdm(range(epochs)):
            self.r3 = max_r3 - max_r3 * np.exp(-0.02 * epoch)
            self.w2 = max_w2 - max_w2 * np.exp(-0.02 * epoch)

            self.Net.train()
            epoch_loss_all = []
            epoch_likelihood_all = []
            epoch_kld_all = []

            if ep_thresh < epoch_m and (epoch+1) >= ep_thresh:
                ep_n += 1
                ep_thresh = pow(1.2, ep_n)

                with torch.no_grad():
                    topic_min_parent, topic_max_parent = self.Net.build_adj_cluster()

                self.Net.update_topic_parent(topic_min_parent, topic_max_parent)
                self.optimizer = optim.Adam(self.Net.parameters(), lr=self.learning_rate)

            if (epoch + 1) == epoch_m:
                self.load_model('root')
                self.optimizer = optim.Adam(self.Net.parameters(), lr=self.learning_rate)

            for _ in tqdm(range(n_batchs),leave=False):
                self.optimizer.zero_grad()

                IV = self.Net.cal_med_var()

                with torch.no_grad(): self.Net.build_adj(IV)

                ori_docs = next(self.train_generator)
                gen_docs, mean, logvar = self.Net(ori_docs, IV)

                likelihood, kld, elbo_loss, box_loss = self.compute_loss(ori_docs, gen_docs, mean, logvar, IV)
                batch_loss =  elbo_loss + box_loss

                batch_loss.backward(ones, retain_graph=False)
                self.optimizer.step()

                epoch_loss_all.append(self.to_np(batch_loss.detach()))
                epoch_likelihood_all.append(self.to_np(likelihood))
                epoch_kld_all.append(self.to_np(kld))
                

            epoch_loss = np.mean(epoch_loss_all)
            epoch_likelihood = np.mean(epoch_likelihood_all)
            epoch_kld = np.mean(epoch_kld_all)
            loss_all.append(epoch_loss)

            print(f'\nEpoch: {epoch}/{epochs}, loss: {epoch_loss:.6f}, likelihood: {epoch_likelihood:.2f}, kld: {epoch_kld:.2f}')
            
            self.Net.eval()
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    IV = self.Net.cal_med_var()
                    self.sampling(IV, uq=False, datatype='valid')
                    if (epoch + 1) % 30 == 0: self.evaluate(IV, uq=False)

        self.t_end = time.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

    def test(self):
        self.load_model('all')

        self.Net.eval()
        self.best_coherence_all = 999
        self.best_coherence_child = 999
        self.best_coherence_root = 999

        IV = self.Net.cal_med_var()
        self.Net.build_adj(IV)

        self.evaluate(IV, uq=False, top_k=30)
        self.sampling(IV, uq=False)
        self.average_sampling(IV, uq=False)


def get_config(config_file, config_name):
    print(' [*] load {}'.format(config_file))
    with open(config_file, 'r') as fin:
        f_data = fin.read()
    config = yaml.safe_load(f_data)
    return config[config_name]


def main(mode='Train', config_file='./BoxTM_config.yaml', config_name='nyt'):
    # get parameter setting
    config = get_config(config_file, config_name)
    dataset = config['dataset']
    input_type = config['input_type']
    save_name = config['save_name']
    batch_size = int(config['batch_size'])
    epochs = int(config['epochs'])
    hidden_num = int(config['hidden_num'])
    emb_size = int(config['emb_size'])
    learning_rate = float(config['learning_rate'])
    level_num = int(config['level_num'])
    topic_nums = int(config['topic_nums'])
    freeze_word = config['freeze_word']
    init_word_scale = tuple([float(v) for v in config['init_word_scale']])
    init_topic_scale = tuple([float(v) for v in config['init_topic_scale']])
    
    sample_size = int(config['sample_size'])
    intersect_temp = float(config['intersect_temp'])
    volume_temp = float(config['volume_temp'])

    r1 = float(config['r1'])
    r2 = float(config['r2'])
    w1 = float(config['w1'])
    
    if 'w2' in config: max_w2 = float(config['w2'])
    else: max_w2= 0.05

    if 'r3' in config: max_r3 = float(config['r3'])
    else: max_r3 = 0.05
    
    use_clamp = True

    base_path = '.'
    data_path = f"{base_path}/data/{dataset}"
    coor_path = f'{data_path}/cooccurr/cooccurr.npz'

    reader = TextReader(data_path, emb_size=emb_size)

    model_path = f'{base_path}/model/{dataset}_{topic_nums}_{reader.vocab_size}'
    model = BoxTM(reader, level_num, topic_nums, hidden_num, emb_size, model_path, save_name=save_name, freeze_word=freeze_word, learning_rate=learning_rate, init_word_scale=init_word_scale, init_topic_scale=init_topic_scale, sample_size=sample_size, intersect_temp=intersect_temp, volume_temp=volume_temp, r1=r1, r2=r2, w1=w1, max_w2=max_w2, max_r3=max_r3, use_clamp=use_clamp, coor_path=coor_path)
    
    if mode == 'Train':
        model.train(epochs=epochs, batch_size=batch_size, data_type='train', input_type=input_type)
    elif mode == 'Test':
        model.test()
    else:
        print(f'Unknowned mode {mode}!')

if __name__ == '__main__':
    _, mode, config_file, config_name = sys.argv

    main(mode=mode, config_file=config_file, config_name=config_name)