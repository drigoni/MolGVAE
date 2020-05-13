#!/usr/bin/env/python
"""
Usage:
    MolGVAE.py [options]

Options:
    -h --help                   Show this screen
    --dataset NAME              Dataset name: ZINC or QM9
    --config-file FILE          Hyperparameter configuration file path (in JSON format)
    --config CONFIG             Hyperparameter configuration dictionary (in JSON format)
    --data_dir NAME             Data dir name
    --restore FILE              File to restore weights from.
    --freeze-graph-model        Freeze weights of graph model components
    --restrict_data NAME        [0,1] Load only a subset of the entire dataset
"""

from docopt import docopt
import traceback
import sys
from model.GGNN_core import ChemModel
from model.data_augmentation import *
import time
import tensorflow as tf
from rdkit import Chem
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0, 1, 2, 3

"""
Comments provide the expected tensor shapes where helpful.

Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edege types (3)
es:     maximum number of BFS transitions in this batch
v:      number of vertices per graph in this batch
h:      GNN hidden size
"""

class MolGVAE(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
                        'suffix': None,              
                        'log_dir': './results',
                        'task_sample_ratios': {},
                        'use_edge_bias': True,                                          # whether use edge bias in gnn
                        'clamp_gradient_norm': 1.0,
                        'out_layer_dropout_keep_prob': 1.0,
                        'tie_fwd_bkwd': True,
                        'task_ids': [0],                                                # id of property prediction
                        'random_seed': 0,                                               # fixed for reproducibility
                        'batch_size': 13 if dataset == 'zinc' else 150,         # qm9 128->8431Mb  150->14403
                                                                                # zinc 8->8431Mb  13->14401
                        "qed_trade_off_lambda": 10,                             # originale 10
                        'prior_learning_rate': 0.05,
                        'stop_criterion': 0.01,
                        'num_epochs': 1000 if dataset == 'zinc' else 1000,
                        'num_teacher_forcing': 1000 if dataset == 'zinc' else 1000,
                        'number_of_generation': 20000,
                        'optimization_step': 0,      
                        'maximum_distance': 50,
                        "use_argmax_nodes": False,                      # use random sampling or argmax during node sampling
                        "use_argmax_bonds": False,                      # use random sampling or argmax during bonds generations
                        'use_mask': False,                              # true to use node mask
                        'residual_connection_on': True,                 # whether residual connection is on
                        'residual_connections': {
                                2: [0],
                                4: [0, 2],
                                6: [0, 2, 4],
                                8: [0, 2, 4, 6],
                                10: [0, 2, 4, 6, 8],
                                12: [0, 2, 4, 6, 8, 10],
                                14: [0, 2, 4, 6, 8, 10, 12],
                                16: [0, 2, 4, 6, 8, 10, 12, 14],
                                18: [0, 2, 4, 6, 8, 10, 12, 14, 16],
                                20: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                            },
                        'num_timesteps': 5,                                    # gnn propagation step
                        'hidden_size_decoder': 200,                             # decoder hidden size dimension
                        'hidden_size_encoder': 100,                             # encoder hidden size dimension
                        "kl_trade_off_lambda": 0.05,                             # kl tradeoff originale 0.3
                        'learning_rate': 0.001,
                        'graph_state_dropout_keep_prob': 1,    
                        "compensate_num": 1,                                    # how many atoms to be added during generation

                        'train_file': 'data/molecules_train_%s.json' % dataset,
                        'valid_file': 'data/molecules_valid_%s.json' % dataset,
                        'test_file': 'data/molecules_test_%s.json' % dataset,

                        'try_different_starting': True,
                        "num_different_starting": 6,
                        'generation': 0,                    # 0 = do training, 1 = do only gen, 2 = do only rec
                        'reconstruction_en': 20,            # number of encoding in reconstruction
                        'reconstruction_dn': 1,             # number of decoding in reconstruction

                        'use_graph': False,                 # use gnn
                        'use_gin': True,                    # use gin as gnn
                        'gin_epsilon': 0,                   # gin epsilon
                        "label_one_hot": False,             # one hot label or not
                        "multi_bfs_path": False,            # whether sample several BFS paths for each molecule
                        "bfs_path_count": 30,
                        "path_random_order": False,         # False: canonical order, True: random order
                        "sample_transition": False,         # whether use transition sampling
                        'edge_weight_dropout_keep_prob': 1,
                        'check_overlap_edge': False,
                        "truncate_distance": 10,
                        "use_gpu": True,
                        "use_rec_multi_threads": True,
                        "use_set_losses": False,            # whether to use crossentropy or a loss over sets of nodes
                        })

        return params

    def prepare_specific_graph_model(self) -> None:
        # params
        h_dim_en = self.params['hidden_size_encoder']
        h_dim_de = self.params['hidden_size_decoder']
        expanded_h_dim = h_dim_de + h_dim_en + 1  # 1 for focus bit
        hist_dim = self.histograms['hist_dim']

        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        # mask out invalid node
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None], name='node_mask')  # [b v]
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, (), name="num_vertices")
        # adj for encoder
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32, [None, self.num_edge_types, None, None], name="adjacency_matrix")  # [b, e, v, v]
        # labels for node symbol prediction
        self.placeholders['node_symbols'] = tf.placeholder(tf.float32, [None, None, self.params['num_symbols']])  # [b, v, edge_type]
        # mask out cross entropies in decoder
        self.placeholders['iteration_mask']=tf.placeholder(tf.float32, [None, None]) # [b, es]
        # adj matrices used in decoder
        self.placeholders['incre_adj_mat']=tf.placeholder(tf.float32, [None, None, self.num_edge_types, None, None], name='incre_adj_mat')  # [b, es, e, v, v]
        # distance 
        self.placeholders['distance_to_others']=tf.placeholder(tf.int32, [None, None, None], name='distance_to_others')  # [b, es,v]
        # maximum iteration number of this batch
        self.placeholders['max_iteration_num']=tf.placeholder(tf.int32, [], name='max_iteration_num')  # number
        # node number in focus at each iteration step
        self.placeholders['node_sequence']=tf.placeholder(tf.float32, [None, None, None], name='node_sequence')  # [b, es, v]
        # mask out invalid edge types at each iteration step 
        self.placeholders['edge_type_masks']=tf.placeholder(tf.float32, [None, None, self.num_edge_types, None], name='edge_type_masks')  # [b, es, e, v]
        # ground truth edge type labels at each iteration step 
        self.placeholders['edge_type_labels']=tf.placeholder(tf.float32, [None, None, self.num_edge_types, None], name='edge_type_labels')  # [b, es, e, v]
        # mask out invalid edge at each iteration step 
        self.placeholders['edge_masks']=tf.placeholder(tf.float32, [None, None, None], name='edge_masks')  # [b, es, v]
        # ground truth edge labels at each iteration step 
        self.placeholders['edge_labels']=tf.placeholder(tf.float32, [None, None, None], name='edge_labels')  # [b, es, v]
        # ground truth labels for whether it stops at each iteration step
        self.placeholders['local_stop']=tf.placeholder(tf.float32, [None, None], name='local_stop')  # [b, es]
        # z_prior sampled from standard normal distribution
        self.placeholders['z_prior']=tf.placeholder(tf.float32, [None, None, h_dim_en], name='z_prior')  # the prior of z sampled from normal distribution
        # put in front of kl latent loss
        self.placeholders['kl_trade_off_lambda']=tf.placeholder(tf.float32, [], name='kl_trade_off_lambda')  # number
        # overlapped edge features
        self.placeholders['overlapped_edge_features']=tf.placeholder(tf.int32, [None, None, None], name='overlapped_edge_features') # [b, es, v]

        # weights for encoder and decoder GNN.
        if self.params['use_graph']:
            if self.params["residual_connection_on"]:
                # weights for encoder and decoder GNN. Different weights for each iteration
                for scope in ['_encoder', '_decoder']:
                    if scope == '_encoder':
                        new_h_dim = h_dim_en
                    else:
                        new_h_dim = expanded_h_dim
                        # For each GNN iteration
                    for iter_idx in range(self.params['num_timesteps']):
                        with tf.variable_scope("gru_scope"+scope+str(iter_idx), reuse=False):
                            self.weights['edge_weights'+scope+str(iter_idx)] = tf.Variable(glorot_init([self.num_edge_types, new_h_dim, new_h_dim]))
                            if self.params['use_edge_bias']:
                                self.weights['edge_biases'+scope+str(iter_idx)] = tf.Variable(np.zeros([self.num_edge_types, 1, new_h_dim]).astype(np.float32))

                            cell = tf.contrib.rnn.GRUCell(new_h_dim)
                            cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
                            self.weights['node_gru'+scope+str(iter_idx)] = cell
            else:
                for scope in ['_encoder', '_decoder']:
                    if scope == '_encoder':
                        new_h_dim= h_dim_en
                    else:
                        new_h_dim=expanded_h_dim
                    self.weights['edge_weights'+scope] = tf.Variable(glorot_init([self.num_edge_types, new_h_dim, new_h_dim]))
                    if self.params['use_edge_bias']:
                        self.weights['edge_biases'+scope] = tf.Variable(np.zeros([self.num_edge_types, 1, new_h_dim]).astype(np.float32))
                    with tf.variable_scope("gru_scope"+scope):
                        cell = tf.contrib.rnn.GRUCell(new_h_dim)
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                             state_keep_prob=self.placeholders['graph_state_keep_prob'])
                        self.weights['node_gru'+scope] = cell
        elif self.params['use_gin']:
            self.weights['gin_epsilon'] = tf.constant(self.params['gin_epsilon'], tf.float32)
            for scope in ['_encoder', '_decoder']:
                if scope == '_encoder':
                    new_h_dim = h_dim_en
                else:
                    new_h_dim = expanded_h_dim
                    # For each GNN iteration
                for iter_idx in range(self.params['num_timesteps']):
                    with tf.variable_scope("gin_scope" + scope + str(iter_idx), reuse=False):
                        self.weights['edge_weights' + scope + str(iter_idx)] = tf.Variable(
                            glorot_init([self.num_edge_types, new_h_dim, new_h_dim]))
                        if self.params['use_edge_bias']:
                            self.weights['edge_biases' + scope + str(iter_idx)] = tf.Variable(
                                np.zeros([self.num_edge_types, 1, new_h_dim]).astype(np.float32))
                        self.weights['mlp' + scope + str(iter_idx)] = MLP(new_h_dim,
                                                                      new_h_dim,
                                                                      [new_h_dim, new_h_dim],
                                                                      self.placeholders['out_layer_dropout_keep_prob'])


        # Weights final part encoder. They map all nodes in one point in the latent space
        self.weights['mean_weights'] = tf.Variable(glorot_init([h_dim_en * (self.params['num_timesteps'] + 1), h_dim_en]), name="mean_weights")
        self.weights['mean_biases'] = tf.Variable(np.zeros([1, h_dim_en]).astype(np.float32), name="mean_biases")
        self.weights['variance_weights'] = tf.Variable(glorot_init([h_dim_en * (self.params['num_timesteps'] + 1), h_dim_en]), name="variance_weights")
        self.weights['variance_biases'] = tf.Variable(np.zeros([1, h_dim_en]).astype(np.float32), name="variance_biases")

        # histograms for the first part of the decoder
        self.placeholders['histograms'] = tf.placeholder(tf.int32, (None, hist_dim), name="histograms")
        self.placeholders['n_histograms'] = tf.placeholder(tf.int32, (None), name="n_histograms")
        self.placeholders['hist'] = tf.placeholder(tf.int32, (None, hist_dim), name="hist")
        #self.weights['mlp_hist'] = MLP(h_dim_en + 2*hist_dim, h_dim_en, [h_dim_en + 2*hist_dim, h_dim_en], 1)
        self.weights['latent_space_dec0'] = tf.Variable(glorot_init([h_dim_en + 2*hist_dim, h_dim_en]))
        self.weights['latent_space_bias_dec0'] = tf.Variable(np.zeros([1, h_dim_en]).astype(np.float32))

        # The weights for generating nodel symbol logits    
        self.weights['node_symbol_weights'] = tf.Variable(glorot_init([h_dim_de , self.params['num_symbols']]))
        self.weights['node_symbol_biases'] = tf.Variable(np.zeros([1, self.params['num_symbols']]).astype(np.float32))


        # gen edges
        # self.weights['mlp_edges'] = MLP(h_dim_en+h_dim_de, 20, [h_dim_de, h_dim_de], self.placeholders['out_layer_dropout_keep_prob'])
        self.weights['edge_gen'] = tf.Variable(glorot_init([h_dim_en + h_dim_de, 1]))
        self.weights['edge_gen_bias'] = tf.Variable(np.zeros([1, 1]).astype(np.float32))
        self.weights['edge_type_gen'] = tf.Variable(glorot_init([h_dim_en + h_dim_de, self.num_edge_types]))
        self.weights['edge_type_gen_bias'] = tf.Variable(np.zeros([1, self.num_edge_types]).astype(np.float32))

        feature_dimension = 6 * expanded_h_dim
        # record the total number of features
        self.params["feature_dimension"] = 6
        # weights for generating edge type logits
        for i in range(self.num_edge_types):
            self.weights['edge_type_%d' % i] = tf.Variable(glorot_init([feature_dimension, feature_dimension]))
            self.weights['edge_type_biases_%d' % i] = tf.Variable(np.zeros([1, feature_dimension]).astype(np.float32))
            self.weights['edge_type_output_%d' % i] = tf.Variable(glorot_init([feature_dimension, 1]))
        # weights for generating edge logits
        self.weights['edge_iteration'] = tf.Variable(glorot_init([feature_dimension, feature_dimension]))
        self.weights['edge_iteration_biases'] = tf.Variable(np.zeros([1, feature_dimension]).astype(np.float32))
        self.weights['edge_iteration_output'] = tf.Variable(glorot_init([feature_dimension, 1]))
        # Weights for the stop node
        self.weights["stop_node"] = tf.Variable(glorot_init([1, expanded_h_dim]))
        # Weight for distance embedding
        self.weights['distance_embedding'] = tf.Variable(glorot_init([self.params['maximum_distance'], expanded_h_dim]))
        # Weight for overlapped edge feature
        self.weights["overlapped_edge_weight"] = tf.Variable(glorot_init([2, expanded_h_dim]))
        # weights for linear projection on qed prediction input
        self.weights['qed_weights'] = tf.Variable(glorot_init([h_dim_en, h_dim_en]))
        self.weights['qed_biases'] = tf.Variable(np.zeros([1, h_dim_en]).astype(np.float32))
        # use node embeddings
        self.weights["node_embedding"] = tf.Variable(glorot_init([self.params["num_symbols"], h_dim_en]))
        
        # graph state mask
        self.ops['graph_state_mask'] = tf.expand_dims(self.placeholders['node_mask'], 2)

    # transform one hot vector to dense embedding vectors
    def get_node_embedding_state(self, one_hot_state):
        node_nums=tf.argmax(one_hot_state, axis=2)
        return tf.nn.embedding_lookup(self.weights["node_embedding"], node_nums) * self.ops['graph_state_mask']

    def compute_final_node_representations_with_residual(self, h, adj, scope_name):  # scope_name: _encoder or _decoder
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        # _decoder uses a larger latent space because concat of symbol and latent representation
        if scope_name=="_decoder":
            h_dim = self.params['hidden_size_encoder'] + self.params['hidden_size_decoder'] + 1
        else:
            h_dim = self.params['hidden_size_encoder']
        h = tf.reshape(h, [-1, h_dim]) # [b*v, h]
        # record all hidden states at each iteration
        all_hidden_states=[h]
        for iter_idx in range(self.params['num_timesteps']):
            with tf.variable_scope("gru_scope"+scope_name+str(iter_idx), reuse=None) as g_scope:
                for edge_type in range(self.num_edge_types):
                    # the message passed from this vertice to other vertices
                    m = tf.matmul(h, self.weights['edge_weights'+scope_name+str(iter_idx)][edge_type])  # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases'+scope_name+str(iter_idx)][edge_type]            # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])                                                   # [b, v, h]
                    # collect the messages from other vertices to each vertice
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                # all messages collected for each node
                acts = tf.reshape(acts, [-1, h_dim])                                                    # [b*v, h]
                # add residual connection here
                layer_residual_connections = self.params['residual_connections'].get(iter_idx)
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [all_hidden_states[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]
                # concat current hidden states with residual states
                acts= tf.concat([acts] + layer_residual_states, axis=1)                                 # [b, (1+num residual connection)* h]

                # feed msg inputs and hidden states to GRU
                h = self.weights['node_gru'+scope_name+str(iter_idx)](acts, h)[1]                       # [b*v, h]
                # record the new hidden states
                all_hidden_states.append(h)
        last_h = tf.reshape(all_hidden_states[-1], [-1, v, h_dim])
        return last_h

    def compute_final_node_representations_without_residual(self, h, adj, edge_weights, edge_biases, node_gru, gru_scope_name):
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        if gru_scope_name=="gru_scope_decoder":
            h_dim = self.params['hidden_size_encoder'] + self.params['hidden_size_decoder']
        else:
            h_dim = self.params['hidden_size_encoder']
        h = tf.reshape(h, [-1, h_dim])

        with tf.variable_scope(gru_scope_name) as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    m = tf.matmul(h, tf.nn.dropout(edge_weights[edge_type],
                               keep_prob=self.placeholders['edge_weight_dropout_keep_prob']))           # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += edge_biases[edge_type]                                                     # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])                                                   # [b, v, h]
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)  #adj[edge_type]->[b,v,v]   m->[b,v,h]  res->b[v,h]
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])                                                    # [b*v, h]
                h = node_gru(acts, h)[1]                                                                # [b*v, h]
            last_h = tf.reshape(h, [-1, v, h_dim])
        return last_h


    def compute_final_node_with_GIN(self, h, adj, scope_name):  # scope_name: _encoder or _decoder
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        # _decoder uses a larger latent space because concat of symbol and latent representation
        if scope_name=="_decoder":
            h_dim = self.params['hidden_size_encoder'] + self.params['hidden_size_decoder'] + 1
        else:
            h_dim = self.params['hidden_size_encoder']
        h = tf.reshape(h, [-1, h_dim])  # [b*v, h]
        weigths_concat = h
        for iter_idx in range(self.params['num_timesteps']):
            with tf.variable_scope("gin_scope"+scope_name+str(iter_idx), reuse=None) as g_scope:
                for edge_type in range(self.num_edge_types):
                    # the message passed from this vertice to other vertices
                    m = tf.matmul(h, self.weights['edge_weights'+scope_name+str(iter_idx)][edge_type])  # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases'+scope_name+str(iter_idx)][edge_type]            # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])                                                   # [b, v, h]
                    # collect the messages from other vertices to each vertice
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                # all messages collected for each node
                acts = tf.reshape(acts, [-1, h_dim])                                                    # [b*v, h]
                input = tf.multiply((1 + self.weights['gin_epsilon']), h) + acts
                h = self.weights['mlp' + scope_name + str(iter_idx)](input)
                weigths_concat = tf.concat([weigths_concat, h], axis=1)
        last_h = tf.reshape(h, [-1, v, h_dim])
        lats_weigths_concat = tf.reshape(weigths_concat, [-1, v, h_dim * (self.params['num_timesteps'] + 1)])
        return last_h, lats_weigths_concat

    def compute_mean_and_logvariance(self):
        h_dim = self.params['hidden_size_encoder']
        reshped_last_h = tf.reshape(self.ops['final_node_representations'][1], [-1, h_dim * (self.params['num_timesteps'] + 1)])
        mean = tf.matmul(reshped_last_h, self.weights['mean_weights']) + self.weights['mean_biases']
        logvariance = tf.matmul(reshped_last_h, self.weights['variance_weights']) + self.weights['variance_biases']
        self.ops['mean'] = mean
        self.ops['logvariance'] = logvariance

    def sample_with_mean_and_logvariance(self):
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size_encoder']
        # Sample from normal distribution
        z_prior = tf.reshape(self.placeholders['z_prior'], [-1, h_dim])
        # Train: sample from u, Sigma. Generation: sample from 0,1
        z_sampled = tf.cond(self.placeholders['is_generative'], lambda: z_prior, # standard normal
                    lambda: tf.add(self.ops['mean'], tf.multiply(tf.sqrt(tf.exp(self.ops['logvariance'])), z_prior)))
        # filter
        z_sampled = tf.reshape(z_sampled, [-1, v, h_dim]) * self.ops['graph_state_mask']
        self.ops['z_sampled'] = z_sampled


    """
    Construct the nodes representations
    """
    def construct_nodes(self):
        h_dim_de = self.params['hidden_size_decoder']
        num_symbols = self.params['num_symbols']
        batch_size = tf.shape(self.ops['z_sampled'])[0]

        # save the new atom [b, v, h]
        atoms = tf.TensorArray(dtype=tf.int32, size=batch_size, element_shape=[None, 1])
        init_atoms = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=[None, h_dim_de])
        fx_prob = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=[None, num_symbols])
        # iteration on all the molecules in the batch size
        idx, atoms, init_atoms, fx_prob = tf.while_loop(
            lambda idx, atoms, init_atoms, fx_prob: tf.less(idx, batch_size),
            # numbers of example sampled
            self.for_each_molecula,
            (tf.constant(0), atoms, init_atoms, fx_prob),
            parallel_iterations=self.params['batch_size']
        )

        atoms = atoms.stack() * tf.cast(self.ops['graph_state_mask'], tf.int32)
        init_h_states = init_atoms.stack() * self.ops['graph_state_mask']
        nodes_type_probs = fx_prob.stack() * self.ops['graph_state_mask']

        # save the embedding representations of the atoms
        self.ops['initial_nodes_decoder'] = init_h_states
        self.ops['node_symbol_prob'] = nodes_type_probs
        self.ops['sampled_atoms'] = atoms

        self.ops['latent_node_symbols'] = tf.one_hot(tf.squeeze(self.ops['sampled_atoms'], axis=-1), self.params['num_symbols'],
                                                     name='latent_node_symbols') * self.ops['graph_state_mask']

        #self.ops['latent_node_symbols'] = tf.Print(self.ops['latent_node_symbols'],
        #                                           [tf.shape(self.ops['latent_node_symbols']), self.ops['latent_node_symbols']],
        #                                           message="latent_node_symbols ", summarize=1000)  # TODO: pr

    """
    Cycles each atom in order to generate the nodes
    """
    def for_each_molecula(self, idx_sample, atoms,  init_vertices_all, fx_prob_all):
        h_dim_de = self.params['hidden_size_decoder']
        num_symbols = self.params['num_symbols']
        v = self.placeholders['num_vertices']  # bucket size dimension, not all time the real one.
        current_hist = self.placeholders['hist'][idx_sample]
        hist_dim = self.histograms['hist_dim']
        zero_hist = tf.zeros([hist_dim], tf.int32)

        sampled_atoms = tf.TensorArray(dtype=tf.int32, size=v, element_shape=[1])
        vertices = tf.TensorArray(dtype=tf.float32, size=v, element_shape=[h_dim_de])
        fx_prob = tf.TensorArray(dtype=tf.float32, size=v, element_shape=[num_symbols])
        # iteration on all the atoms in a molecule
        idx_atoms, a, v, fx, _, _, _ = tf.while_loop(
            lambda idx_atoms, s_atoms, vertices, fx_prob, s_idx, zero_hist, current_hist: tf.less(idx_atoms, v),
            self.generate_nodes,
            (tf.constant(0), sampled_atoms, vertices, fx_prob, idx_sample, zero_hist, current_hist)
        )

        atoms = atoms.write(idx_sample, a.stack())
        init_vertices_all = init_vertices_all.write(idx_sample, v.stack())
        fx_prob_all = fx_prob_all.write(idx_sample, fx.stack())

        return idx_sample + 1, atoms, init_vertices_all, fx_prob_all

    """
    For each node (atoms) calculates histograms and new hidden representations
    """
    def generate_nodes(self, idx_atom, atoms, init_vertices, fx_prob, idx_sample, updated_hist, sampled_hist):
        is_generative = self.placeholders['is_generative']

        s_atom, init_v, fx, new_updated_hist, new_sampled_hist = tf.cond(is_generative,
                        lambda: self.generate_mode_sampling(idx_atom, idx_sample, updated_hist, sampled_hist),
                        lambda: self.training_sampling(idx_atom, idx_sample, updated_hist, sampled_hist))

        atoms = atoms.write(idx_atom, s_atom)
        init_vertices = init_vertices.write(idx_atom, init_v)
        fx_prob = fx_prob.write(idx_atom, fx)
        return idx_atom + 1, atoms, init_vertices, fx_prob, idx_sample, new_updated_hist, new_sampled_hist


    def training_sampling(self, idx_atom, idx_sample, updated_hist, sampled_hist):
        # applying teach forcing
        current_sample_hist = self.placeholders['hist'][idx_sample]
        current_sample_hist_casted = tf.cast(current_sample_hist, dtype=tf.float32)
        current_sample_z = self.ops['z_sampled'][idx_sample][idx_atom]
        # node = current_sample_z

        # concatenation of the latent space point with the difference between the sampled histogram and the current histogram
        current_hist_casted = tf.cast(updated_hist, dtype=tf.float32)
        hist_diff = tf.subtract(current_sample_hist_casted, current_hist_casted)
        hist_diff_pos = tf.where(hist_diff > 0, hist_diff, tf.zeros_like(hist_diff))
        conc = tf.concat([current_sample_z, current_hist_casted, hist_diff_pos], axis=0)
        exp = tf.expand_dims(conc, 0)   # [1, z + Hdiff + Hcurrent]

        # build a node with NN (K)
        hist_emb = tf.nn.tanh(tf.matmul(exp, self.weights['latent_space_dec0']) + self.weights['latent_space_bias_dec0'])
        #hist_emb = self.weights['mlp_hist'](exp)
        node_prob = tf.concat([tf.expand_dims(current_sample_z, 0), hist_emb], -1)
        node = node_prob

        # node = tf.Print(node, [tf.shape(node), node], message="node_latent_space ", summarize=1000)  # TODO: pr
        fx_logit = tf.squeeze(tf.matmul(node_prob, self.weights['node_symbol_weights']) + self.weights['node_symbol_biases'])
        if self.params['use_mask']:
            fx_logit, mask = self.mask_mols(fx_logit, hist_diff_pos)
        fx_prob = tf.nn.softmax(fx_logit)
        # fx_prob = tf.Print(fx_prob, [fx_prob], message="training sampling: probs after masking ", summarize=1000)  # TODO: pr

        # test the mask
        #test = tf.cond(tf.reduce_any(mask),
        #                       lambda: mask,
        #                       lambda: tf.ones_like(mask))
        #test = tf.cast(test, tf.float32)
        #val = tf.argmax(self.placeholders['node_symbols'][idx_sample][idx_atom])
        #self.ops['assert'] = tf.Assert(tf.equal(test[val], 1), [test, val])


        # update the histogram
        probs_value = tf.cond(self.placeholders['use_teacher_forcing_nodes'],
                              lambda: self.placeholders['node_symbols'][idx_sample][idx_atom],
                              lambda: fx_prob)

        #probs_value = tf.Print(probs_value, [probs_value], message="training: probs_value", summarize=1000)  # TODO: pr
        s_atom = self.sample_atom(probs_value, True)
        # s_atom = tf.Print(s_atom, [s_atom], message="training: s_atom ", summarize=1000)  # TODO: pr
        current_new_hist = self.update_hist(updated_hist, s_atom)
        #current_new_hist = tf.Print(current_new_hist, [current_new_hist], message="training: current_new_hist", summarize=1000)  # TODO: pr

        return tf.expand_dims(s_atom, 0), tf.squeeze(node), fx_prob, current_new_hist, sampled_hist


    def generate_mode_sampling(self, idx_atom, idx_sample, updated_hist,  sampled_hist):
        hist_dim = self.histograms['hist_dim']
        current_sample_z = self.ops['z_sampled'][idx_sample][idx_atom]
        # node = current_sample_z

        current_sample_hist_casted = tf.cast(sampled_hist, dtype=tf.float32)
        current_hist_casted = tf.cast(updated_hist, dtype=tf.float32)
        hist_diff = tf.subtract(current_sample_hist_casted, current_hist_casted)
        hist_diff_pos = tf.where(hist_diff > 0, hist_diff, tf.zeros_like(hist_diff))
        conc = tf.concat([current_sample_z, current_hist_casted, hist_diff_pos], axis=0)
        exp = tf.expand_dims(conc, 0)

        # build a node with NN (K)
        hist_emb = tf.nn.tanh(tf.matmul(exp, self.weights['latent_space_dec0']) + self.weights['latent_space_bias_dec0'])
        # hist_emb = self.weights['mlp_hist'](exp)
        node_prob = tf.concat([tf.expand_dims(current_sample_z, 0), hist_emb], -1)
        node = node_prob

        fx_logit = tf.squeeze(tf.matmul(node_prob, self.weights['node_symbol_weights']) + self.weights['node_symbol_biases'])
        if self.params['use_mask']:
            fx_logit, mask = self.mask_mols(fx_logit, hist_diff_pos)
        fx_prob = tf.nn.softmax(fx_logit)
        s_atom = self.sample_atom(fx_prob, False)
        new_updated_hist = self.update_hist(updated_hist, s_atom)

        # sampling one compatible histogram with the current new histogram
        reshape = tf.reshape(new_updated_hist, (-1, hist_dim))  # reshape the dimension from [n_valences] to [1, n_valences]
        m1 = self.placeholders['histograms'] >= reshape  # vector of 0 and 1


        m2 = tf.reduce_sum(tf.cast(m1, dtype=tf.int32), axis=1)  # [b]
        m3 = tf.equal(m2, tf.constant(hist_dim))    # [b]
        m4 = tf.cast(m3, dtype=tf.int32)  # [b]
        m5 = tf.multiply(self.placeholders['n_histograms'], m4)  # [b]
        mSomma = tf.reduce_sum(m5)
        new_sampled_hist = tf.cond(tf.equal(tf.constant(0), mSomma),
                                  lambda: self.case_random_sampling(),
                                  lambda: self.case_sampling(m5, mSomma))

        return tf.expand_dims(s_atom, 0), tf.squeeze(node), fx_prob, new_updated_hist, new_sampled_hist


    def mask_mols(self, logits, hist):
        mol_valence_list = []
        for key in dataset_info(self.params['dataset'])['maximum_valence'].keys():
            mol_valence_list.append(dataset_info(self.params['dataset'])['maximum_valence'][key])
        mol_valence = tf.constant(mol_valence_list)
        H_b = tf.cast(hist > 0, tf.int32)  # obtaining a vector of only 1 and 0
        idx = tf.where(tf.less_equal(H_b, 0))  # obtaining al the indexes with 0-values in the hist
        valences = tf.cast(idx + 1, tf.int32)  # valences to avoid
        equals = tf.not_equal(mol_valence, valences)  # broadcasting.
        mask_bool = tf.reduce_all(equals, 0)
        mask = tf.cast(mask_bool, tf.float32)
        logits_masked = tf.cond(tf.reduce_any(mask_bool),
                               lambda: logits + (mask * LARGE_NUMBER - LARGE_NUMBER),
                               lambda: logits)
        return logits_masked, mask_bool


    """
    Histograms sampling with probs
    """
    def case_sampling(self, m5, mSomma):
        prob = m5 / mSomma
        m8 = tf.distributions.Categorical(probs=prob).sample()
        m9 = self.placeholders['histograms'][m8]
        return m9

    """
    Histograms uniform sampling
    """
    def case_random_sampling(self):
        max_n = tf.shape(self.placeholders['histograms'])[0]
        # max_n = tf.Print(max_n, [max_n], message="generate sampling random", summarize=1000)  # TODO: pr
        idx = tf.random_uniform([], maxval=max_n, dtype=tf.int32)
        return self.placeholders['histograms'][idx]


    """
    Sample the id of the atom for a value fo probabilities. 
    In training always apply argmax, while in generation it is possible to choose among distribution or argmax
    """
    def sample_atom(self, fx_prob, training):
        if training:
                idx = tf.argmax(fx_prob, output_type=tf.int32)
        else:
            if self.params['use_argmax_nodes']:
                idx = tf.argmax(fx_prob, output_type=tf.int32)
            else:
                idx = tf.distributions.Categorical(probs=fx_prob).sample()
        return idx

    """
    Update of the histogram according to the new atom.
    """
    def update_hist(self, old_hist, id_atom):
        hist_dim = self.histograms['hist_dim']
        mol_valence_list = []
        # this is ok even if the dictionary is not ordered
        for key in dataset_info(self.params['dataset'])['maximum_valence'].keys():
            mol_valence_list.append(dataset_info(self.params['dataset'])['maximum_valence'][key])
        # make the mol-valence array
        mol_valence = tf.constant(mol_valence_list)
        # take the atom valence
        atmo_val = mol_valence[id_atom]
        # build an array to be used in add operation
        array = tf.one_hot(atmo_val - 1, hist_dim, dtype=tf.int32)  # remember that valence start from 1
        # summing the two array
        new_hist = tf.add(old_hist, array)
        return new_hist

    def construct_logit_matrices(self):
        v = self.placeholders['num_vertices']
        batch_size = tf.shape(self.ops['latent_node_symbols'])[0]
        # prep valences
        mol_valence_list = []
        for key in dataset_info(self.params['dataset'])['maximum_valence'].keys():
            mol_valence_list.append(dataset_info(self.params['dataset'])['maximum_valence'][key])
        mol_valence = tf.constant(mol_valence_list)
        indexes = tf.argmax(self.ops['latent_node_symbols'], axis=-1)  # [b, v]
        valences = tf.nn.embedding_lookup(mol_valence, indexes)

        #    The tensor array used to collect the cross entropy losses at each step
        edges_pred = tf.TensorArray(dtype=tf.float32, size=v)
        edges_type_pred = tf.TensorArray(dtype=tf.float32, size=v)
        idx_final, edges_pred, edges_type_pred, _ = \
            tf.while_loop(
                lambda idx, edges_pred, edges_type_pred, valences: idx < v, self.generate_edges,
                (tf.constant(0), edges_pred, edges_type_pred, valences)
            )

        self.ops['edges_pred'] = tf.transpose(edges_pred.stack(), [1,0,2]) * self.ops['graph_state_mask']
        self.ops['edges_type_pred'] = tf.transpose(edges_type_pred.stack(), [1, 3, 0, 2])

        # mask diagonal in order to put all probabilities to 1 in the non existence of the edge
        diag_0 = tf.one_hot(tf.range(v), depth=v, on_value=0.0, off_value=1.0, dtype=tf.float32)
        self.ops['edges_pred'] = self.ops['edges_pred'] * tf.expand_dims(diag_0, 0)

        gt_edges_pred = tf.reduce_sum(self.placeholders['adjacency_matrix'], axis=1)
        gt_edges_type_pred = self.placeholders['adjacency_matrix']

        # gt = tf.Print(gt, [gt[0,:,0,0]], message="adj", summarize=1000)  # TODO: pr

        # binary cross-entropy masked and balanced
        n_yes_edges = tf.reduce_sum(gt_edges_pred)
        n_no_edges = tf.reduce_sum(1 - gt_edges_pred)

        edge_loss =- tf.reduce_sum((tf.log(self.ops['edges_pred'] + SMALL_NUMBER) * gt_edges_pred)*(n_no_edges/n_yes_edges) +
                                   tf.log((1 - self.ops['edges_pred']) + SMALL_NUMBER) * (1-gt_edges_pred),
                                   axis=[1, 2])

        # edge type cross entropy balanced and masked
        loss_batchEdge = tf.reduce_sum(tf.log(self.ops['edges_type_pred'] + SMALL_NUMBER) * gt_edges_type_pred, axis=[2, 3])
        edge_type_loss = loss_batchEdge[:, 0]
        n_type_1_edge = tf.reduce_sum(gt_edges_type_pred[:, 0, :, :])
        for i in range(0, self.num_edge_types):
            sum_tmp = tf.reduce_sum(gt_edges_type_pred[:, i, :, :])
            sum_tmp = tf.where(sum_tmp > 0, sum_tmp, n_type_1_edge)
            weights_temp = n_type_1_edge / sum_tmp
            edge_type_loss += loss_batchEdge[:, i] * weights_temp
        edge_type_loss = - edge_type_loss
        # edge_type_loss =- tf.reduce_sum(tf.log(self.ops['edges_type_pred'] + SMALL_NUMBER) * gt_edges_type_pred,
        #                                 axis=[1, 2, 3])

        self.ops['cross_entropy_losses'] = edge_loss + edge_type_loss

        corr_edge = tf.cast(self.ops['edges_pred'] >= 0.5, tf.float32)
        corr_edge = tf.cast(tf.not_equal(corr_edge, gt_edges_pred),
                             tf.float32)
        edges_type_pred_masked = self.ops['edges_type_pred'] * tf.expand_dims(gt_edges_pred, axis=1)
        corr_type_edge = tf.cast(tf.not_equal(tf.argmax(edges_type_pred_masked, axis=1),
                                              tf.argmax(gt_edges_type_pred, axis=1)),
                                 tf.float32)

        self.ops['edge_pred_error'] = tf.reduce_sum(corr_edge, axis=[1, 2])
        self.ops['edge_type_pred_error'] = tf.reduce_sum(corr_type_edge, axis=[1, 2])

    def generate_edges(self, idx, edges_pred, edges_type_pred, valences):
        v = self.placeholders['num_vertices']
        h_dim_en = self.params['hidden_size_encoder']
        h_dim_de = self.params['hidden_size_decoder']
        batch_size = tf.shape(self.ops['latent_node_symbols'])[0]
        edges_val_req = [i+1 for i in range(0, self.num_edge_types)]
        edges_val_req = tf.expand_dims(edges_val_req, 0)
        edges_val_req = tf.expand_dims(edges_val_req, 0)
        edges_val_req = tf.tile(edges_val_req, [batch_size, v, 1])

        latent_node_state = self.get_node_embedding_state(self.ops['latent_node_symbols'])
        self.ops["initial_repre_for_decoder"] = filtered_z_sampled = tf.concat([self.ops['initial_nodes_decoder'],
                                                                                latent_node_state],
                                                                               axis=2)  # [b, v, h + h]

        # node in focus feature
        node_focus = filtered_z_sampled[:, idx, :]
        node_focus = tf.expand_dims(node_focus, axis=1)
        node_focus_feature = tf.tile(node_focus, [1, v, 1]) + filtered_z_sampled

        # node in focus valences
        node_focus_valences = valences[:, idx]
        node_focus_valences = tf.expand_dims(node_focus_valences, axis=1)
        node_focus_feature_valences = tf.tile(node_focus_valences, [1, v])

        # generate mask
        mask_min = tf.stack([node_focus_feature_valences, valences], axis=-1)
        mask_min = tf.reduce_min(mask_min, -1)
        mask_min = tf.tile(tf.expand_dims(mask_min, 2), [1,1, self.num_edge_types])
        # mask_min = tf.Print(mask_min, [mask_min[0]], message="mask_min", summarize=1000)  # TODO: pr
        mask = tf.cast(edges_val_req <= mask_min, tf.float32)
        mask = tf.reshape(mask, [-1, self.num_edge_types])
        # mask = tf.Print(mask, [mask[0]], message="mask", summarize=1000)  # TODO: pr

        edge_rep = tf.reshape(node_focus_feature, [-1,h_dim_en + h_dim_de])  # [b * v, h_dec + h_enc]
        edge_pred_tmp = tf.matmul(edge_rep, self.weights['edge_gen']) + self.weights['edge_gen_bias']  # [b*v, num_edges + 1]
        edge_pred_tmp = tf.nn.sigmoid(edge_pred_tmp)
        edge_pred_tmp = tf.reshape(edge_pred_tmp, [batch_size, v, 1]) * self.ops['graph_state_mask']
        edge_pred_tmp = tf.squeeze(edge_pred_tmp, axis=-1)

        edge_type_pred_tmp = tf.matmul(edge_rep, self.weights['edge_type_gen']) + self.weights['edge_type_gen_bias']  # [b*v, num_edges + 1]
        edge_type_pred_tmp = tf.nn.softmax(edge_type_pred_tmp + (mask * LARGE_NUMBER - LARGE_NUMBER))
        edge_type_pred_tmp = tf.reshape(edge_type_pred_tmp, [batch_size, v, self.num_edge_types]) * self.ops['graph_state_mask']


        edges_pred = edges_pred.write(idx, edge_pred_tmp)
        edges_type_pred = edges_type_pred.write(idx, edge_type_pred_tmp)
        return idx+1, edges_pred, edges_type_pred, valences



    def fully_connected(self, input, hidden_weight, hidden_bias, output_weight):
        output=tf.nn.relu(tf.matmul(input, hidden_weight) + hidden_bias)       
        output=tf.matmul(output, output_weight) 
        return output


    def construct_loss(self):
        v = self.placeholders['num_vertices']
        h_dim_en = self.params['hidden_size_encoder']
        kl_trade_off_lambda =self.placeholders['kl_trade_off_lambda']
        # Edge loss
        # self.ops["edge_loss"] = tf.reduce_sum(self.ops['cross_entropy_losses'] * self.placeholders['iteration_mask'], axis=1)
        self.ops["edge_loss"] = self.ops['cross_entropy_losses']
        # KL loss
        kl_loss = 1 + self.ops['logvariance'] - tf.square(self.ops['mean']) - tf.exp(self.ops['logvariance'])
        kl_loss = tf.reshape(kl_loss, [-1, v, h_dim_en]) * self.ops['graph_state_mask']
        self.ops['kl_loss'] = -0.5 * tf.reduce_sum(kl_loss, [1,2])
        # Node symbol loss
        self.ops['node_symbol_loss'] = -tf.reduce_sum(tf.log(self.ops['node_symbol_prob'] + SMALL_NUMBER) *
                                                      self.placeholders['node_symbols'], axis=[1, 2])

        if self.params['use_set_losses']:
            # first version
            # iou_values = self.IoU(self.ops['latent_node_symbols'], self.placeholders['node_symbols'])
            # selection = tf.one_hot(tf.argmax(self.ops['node_symbol_prob'], 2), self.params['num_symbols'],
            #           name='latent_node_symbols') * self.ops['graph_state_mask']
            #iou_values = self.IoU(tf.cast(selection, tf.float32), self.placeholders['node_symbols'])
            # self.ops['node_symbol_loss'] = self.ops['node_symbol_loss'] * (1 - iou_values)
            # second version
            self.ops['node_symbol_loss'] = 1 - self.IoU(self.ops['node_symbol_prob'], self.placeholders['node_symbols'])



        latent_node_symbol = tf.cast(tf.not_equal(tf.argmax(self.ops['latent_node_symbols'], axis=-1),
                                          tf.argmax(self.placeholders['node_symbols'], axis=-1)),
                                     tf.float32)
        mols_errors = self.ops['edge_pred_error'] + self.ops['edge_type_pred_error'] + tf.reduce_sum(latent_node_symbol, axis= -1)
        # mols_errors = tf.Print(mols_errors, [mols_errors], message="mols_errors", summarize=1000)  # TODO: pr
        self.ops['reconstruction'] = tf.reduce_sum(tf.cast(tf.equal(mols_errors, 0), tf.float32))
        # after because it rewrite the operations
        self.ops['node_pred_error'] = tf.reduce_mean(latent_node_symbol)
        self.ops['edge_pred_error'] = tf.reduce_mean(self.ops['edge_pred_error'])
        self.ops['edge_type_pred_error'] = tf.reduce_mean(self.ops['edge_type_pred_error'])

        # Add in the loss for calculating QED
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(h_dim_en, 1, [],
                                                                           self.placeholders['out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(h_dim_en, 1, [],
                                                                               self.placeholders['out_layer_dropout_keep_prob'])
                normalized_z_sampled=tf.nn.l2_normalize(self.ops['z_sampled'], 2)
                self.ops['qed_computed_values']=computed_values = self.gated_regression(normalized_z_sampled,
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id], h_dim_en,
                                                        self.weights['qed_weights'], self.weights['qed_biases'],
                                                        self.placeholders['num_vertices'], self.placeholders['node_mask'])
                diff = computed_values - self.placeholders['target_values'][internal_id,:]  # [b]
                task_target_mask = self.placeholders['target_mask'][internal_id,:]
                task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
                diff = diff * task_target_mask  # Mask out unused values [b]
                self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
                task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num  # number
                # Normalise loss to account for fewer task-specific examples in batch:
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['qed_loss'].append(task_loss)
                if task_id ==0: # Assume it is the QED score
                    z_sampled_shape=tf.shape(self.ops['z_sampled'])
                    flattened_z_sampled=tf.reshape(self.ops['z_sampled'], [z_sampled_shape[0], -1])
                    self.ops['l2_loss'] = 0.01* tf.reduce_sum(flattened_z_sampled * flattened_z_sampled, axis=1) /2
                    # Calculate the derivative with respect to QED + l2 loss
                    self.ops['derivative_z_sampled'] = tf.gradients(self.ops['qed_computed_values'] -
                                        self.ops['l2_loss'], self.ops['z_sampled'])
        self.ops['total_qed_loss'] = tf.reduce_sum(self.ops['qed_loss']) # number
        self.ops['mean_edge_loss'] = tf.reduce_mean(self.ops["edge_loss"]) # record the mean edge loss
        self.ops['mean_node_symbol_loss'] = tf.reduce_mean(self.ops["node_symbol_loss"])
        self.ops['mean_kl_loss'] = tf.reduce_mean(kl_trade_off_lambda *self.ops['kl_loss'])
        self.ops['mean_total_qed_loss'] = self.params["qed_trade_off_lambda"]*self.ops['total_qed_loss']
        return tf.reduce_mean(self.ops["edge_loss"] + self.ops['node_symbol_loss'] + \
                              kl_trade_off_lambda *self.ops['kl_loss'])\
                              + self.params["qed_trade_off_lambda"]*self.ops['total_qed_loss']

    def IoU(self, y_pred, y_true):
        # van Beers, Floris, et al. "Deep Neural Networks with Intersection over Union Loss for Binary Image Segmentation.
        # " Proceedings of the 8th International Conference on Pattern Recognition Applications and Methods. 2019.
        I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
        S = tf.reduce_sum(y_pred + y_true, axis=(1, 2))
        U = S - I
        return I / U

    def gated_regression(self, last_h, regression_gate, regression_transform, hidden_size, projection_weight, projection_bias, v, mask):
        # last_h: [b x v x h]
        last_h = tf.reshape(last_h, [-1, hidden_size])   # [b*v, h]    
        # linear projection on last_h
        last_h = tf.nn.relu(tf.matmul(last_h, projection_weight)+projection_bias) # [b*v, h]  
        # same as last_h
        gate_input = last_h
        # linear projection and combine                                       
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * tf.nn.tanh(regression_transform(last_h)) # [b*v, 1]
        gated_outputs = tf.reshape(gated_outputs, [-1, v])                  # [b, v]
        masked_gated_outputs = gated_outputs * mask                           # [b x v]
        output = tf.reduce_sum(masked_gated_outputs, axis = 1)                                                # [b]
        output=tf.sigmoid(output)
        return output

    def calculate_incremental_results(self, raw_data, bucket_sizes, file_name):
        incremental_results = []
        # copy the raw_data if more than 1 BFS path is added
        new_raw_data = []
        for idx, d in enumerate(raw_data):
            # Use canonical order or random order here. canonical order starts from index 0. random order starts from random nodes
            if not self.params["path_random_order"]:
                # Use several different starting index if using multi BFS path
                if self.params["multi_bfs_path"]:
                    list_of_starting_idx = list(range(self.params["bfs_path_count"]))
                else:
                    list_of_starting_idx = [0]  # the index 0
            else:
                # get the node length for this molecule
                node_length=len(d["node_features"])
                if self.params["multi_bfs_path"]:
                    list_of_starting_idx = np.random.choice(node_length, self.params["bfs_path_count"], replace=True)  #randomly choose several
                else:
                    list_of_starting_idx = [random.choice(list(range(node_length)))]  # randomly choose one

            # default it is only one element to 0
            for list_idx, starting_idx in enumerate(list_of_starting_idx):
                # choose a bucket
                chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                                    for v in [e[0], e[2]]]))
                chosen_bucket_size = bucket_sizes[chosen_bucket_idx]

                # Calculate incremental results without master node
                nodes_no_master, edges_no_master = to_graph(d['smiles'], self.params["dataset"])                
                incremental_adj_mat,distance_to_others,node_sequence,edge_type_masks,edge_type_labels,local_stop, edge_masks, edge_labels, overlapped_edge_features=\
                construct_incremental_graph(dataset, edges_no_master, chosen_bucket_size, 
                                            len(nodes_no_master), nodes_no_master, self.params, initial_idx=starting_idx)
                if self.params["sample_transition"] and list_idx > 0:
                    incremental_results[-1] = [x+y for x, y in zip(incremental_results[-1], [incremental_adj_mat,distance_to_others,
                                       node_sequence,edge_type_masks,edge_type_labels,local_stop, edge_masks, edge_labels, overlapped_edge_features])]
                else:
                    incremental_results.append([incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks, 
                                               edge_type_labels, local_stop, edge_masks, edge_labels, overlapped_edge_features])
                    # copy the raw_data here 
                    new_raw_data.append(d)
                if idx % 50 == 0:
                    print('finish calculating %d incremental matrices' % idx, end="\r")
        return incremental_results, new_raw_data

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data, is_training_data, file_name, bucket_sizes=None):
        if bucket_sizes is None:
            bucket_sizes = dataset_info(self.params["dataset"])["bucket_sizes"]
        incremental_results, raw_data=self.calculate_incremental_results(raw_data, bucket_sizes, file_name)
        bucketed = defaultdict(list)
        x_dim = len(raw_data[0]["node_features"][0])

        for d, (incremental_adj_mat,distance_to_others,node_sequence,edge_type_masks,edge_type_labels,local_stop, edge_masks, edge_labels, overlapped_edge_features)\
                            in zip(raw_data, incremental_results):
            # choose a bucket
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                                for v in [e[0], e[2]]]))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]            
            # total number of nodes in this data point
            n_active_nodes = len(d["node_features"])
            bucketed[chosen_bucket_idx].append({
                'smiles': d['smiles'],
                'adj_mat': graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types, self.params['tie_fwd_bkwd']),
                'incre_adj_mat': incremental_adj_mat,
                'distance_to_others': distance_to_others,
                'overlapped_edge_features': overlapped_edge_features,
                'node_sequence': node_sequence,
                'edge_type_masks': edge_type_masks,
                'edge_type_labels': edge_type_labels,
                'edge_masks': edge_masks,
                'edge_labels': edge_labels,
                'local_stop': local_stop,
                'number_iteration': len(local_stop),
                'init': d["node_features"] + [[0 for _ in range(x_dim)] for __ in
                                              range(chosen_bucket_size - n_active_nodes)],
                'labels': [d["targets"][task_id][0] for task_id in self.params['task_ids']],
                'mask': [1. for _ in range(n_active_nodes) ] + [0. for _ in range(chosen_bucket_size - n_active_nodes)],
                'hist': d['hist'],
            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'])]
                          for bucket_idx, bucket_data in bucketed.items()]

        # every position indicates the bucket size
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    def make_batch(self, elements, maximum_vertice_num):
        # get maximum number of iterations in this batch. used to control while_loop
        max_iteration_num=-1
        for d in elements:
            max_iteration_num=max(d['number_iteration'], max_iteration_num)
        batch_data = {'smiles':[], 'adj_mat': [], 'init': [], 'labels': [], 'edge_type_masks':[], 'edge_type_labels':[], 'edge_masks':[],
                'edge_labels':[],'node_mask': [], 'task_masks': [], 'node_sequence':[],
                'iteration_mask': [], 'local_stop': [], 'incre_adj_mat': [], 'distance_to_others': [], 
                'max_iteration_num': max_iteration_num, 'overlapped_edge_features': [], 'hist': []}
        for d in elements: 
            # sparse to dense for saving memory           
            incre_adj_mat = incre_adj_mat_to_dense(d['incre_adj_mat'], self.num_edge_types, maximum_vertice_num)
            distance_to_others = distance_to_others_dense(d['distance_to_others'], maximum_vertice_num)
            overlapped_edge_features = overlapped_edge_features_to_dense(d['overlapped_edge_features'], maximum_vertice_num)
            node_sequence = node_sequence_to_dense(d['node_sequence'],maximum_vertice_num)
            edge_type_masks = edge_type_masks_to_dense(d['edge_type_masks'], maximum_vertice_num,self.num_edge_types)
            edge_type_labels = edge_type_labels_to_dense(d['edge_type_labels'], maximum_vertice_num,self.num_edge_types)
            edge_masks = edge_masks_to_dense(d['edge_masks'], maximum_vertice_num)
            edge_labels = edge_labels_to_dense(d['edge_labels'], maximum_vertice_num)

            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['init'].append(d['init'])
            batch_data['node_mask'].append(d['mask'])

            batch_data['incre_adj_mat'].append(incre_adj_mat +
                [np.zeros((self.num_edge_types, maximum_vertice_num,maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['distance_to_others'].append(distance_to_others + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['overlapped_edge_features'].append(overlapped_edge_features + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['node_sequence'].append(node_sequence + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['edge_type_masks'].append(edge_type_masks + 
                [np.zeros((self.num_edge_types, maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['edge_masks'].append(edge_masks + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['edge_type_labels'].append(edge_type_labels + 
                [np.zeros((self.num_edge_types, maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['edge_labels'].append(edge_labels + 
                [np.zeros((maximum_vertice_num)) 
                            for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['iteration_mask'].append([1 for _ in range(d['number_iteration'])]+
                                     [0 for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['local_stop'].append([int(s) for s in d["local_stop"]]+ 
                                     [0 for _ in range(max_iteration_num-d['number_iteration'])])
            batch_data['hist'].append(d['hist'])
            batch_data['smiles'].append(d['smiles'])

            target_task_values = []
            target_task_mask = []
            for target_val in d['labels']:
                if target_val is None:  # This is one of the examples we didn't sample...
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)
            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)
        return batch_data

    """
    Prepare the feed dict for obtaining the nodes (atoms) during generation
    """
    def get_dynamic_feed_dict(self, elements, latent_node_symbol, incre_adj_mat, num_vertices,
                              distance_to_others, overlapped_edge_dense, node_sequence, edge_type_masks, edge_masks,
                              random_normal_states, is_generative):
        if incre_adj_mat is None:
            incre_adj_mat = np.zeros((1, 1, self.num_edge_types, 1, 1))
            distance_to_others = np.zeros((1, 1, 1))
            overlapped_edge_dense = np.zeros((1, 1, 1))
            node_sequence = np.zeros((1, 1, 1))
            edge_type_masks = np.zeros((1, 1, self.num_edge_types, 1))
            edge_masks = np.zeros((1, 1, 1))
            latent_node_symbol = np.zeros((1, 1, self.params["num_symbols"]))

        prob = self.histograms['filter'][1][int(num_vertices)]
        values = self.histograms['filter'][0][int(num_vertices)]
        prob_sum = np.sum(prob)
        if prob_sum == 0:
            sampled_idx = np.random.choice(len(self.histograms['train'][0]))
            values = self.histograms['train'][1]
        else:
            sampled_idx = np.random.choice(len(self.histograms['train'][0]), p=prob)
        return {
            self.placeholders['use_teacher_forcing_nodes']: False,
            self.placeholders['is_generative']: is_generative,
            self.placeholders['z_prior']: random_normal_states, # [1, v, h]
            self.placeholders['incre_adj_mat']: incre_adj_mat,  # [1, 1, e, v, v]
            self.placeholders['num_vertices']: num_vertices,  # v
            self.placeholders['node_symbols']: [elements['init']],
            self.ops['latent_node_symbols']: latent_node_symbol,
            self.placeholders['adjacency_matrix']: [elements['adj_mat']],
            self.placeholders['node_mask']: [elements['mask']],

            self.placeholders['graph_state_keep_prob']: 1,
            self.placeholders['edge_weight_dropout_keep_prob']: 1,
            self.placeholders['iteration_mask']: [[1]],
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
            self.placeholders['distance_to_others']: distance_to_others,  # [1, 1,v]
            self.placeholders['overlapped_edge_features']: overlapped_edge_dense,
            self.placeholders['max_iteration_num']: 1,
            self.placeholders['node_sequence']: node_sequence,  # [1, 1, v]
            self.placeholders['edge_type_masks']: edge_type_masks,  # [1, 1, e, v]
            self.placeholders['edge_masks']: edge_masks,  # [1, 1, v]
            self.placeholders['histograms']: self.histograms['train'][0],
            self.placeholders['n_histograms']: values,
            self.placeholders['hist']: [self.histograms['train'][0][sampled_idx]],
        }

    """
    Prepare the feed dict for accessing the nodes (atoms)
    """
    def get_dynamic_nodes_feed_dict(self, elements, num_vertices, z_sampled, is_generative):
        return {
                self.placeholders['use_teacher_forcing_nodes']: False,
                self.placeholders['is_generative']: is_generative,
                self.ops['z_sampled']: z_sampled,  # [hl]
                self.placeholders['num_vertices']: num_vertices,     # v
                self.placeholders['node_symbols']: [elements['init']],
                self.placeholders['node_mask']: [elements['mask']],
                self.placeholders['graph_state_keep_prob']: 1,
                self.placeholders['edge_weight_dropout_keep_prob']: 1,
                self.placeholders['iteration_mask']: [[1]],
                self.placeholders['out_layer_dropout_keep_prob'] : 1.0,
                self.placeholders['histograms']: self.histograms['train'][0],
                self.placeholders['n_histograms']: self.histograms['train'][1],
                self.placeholders['hist']: [elements['hist']],
            }


    """
    Prepare the feed dict for searching the edges amongs atoms
    """
    def get_dynamic_edge_feed_dict(self, elements, latent_nodes, latent_node_symbol, num_vertices):
        return {
                self.placeholders['adjacency_matrix']: elements['adj_mat'],  # [1, 1, e, v, v]
                self.placeholders['num_vertices']: num_vertices,  # v
                self.ops['initial_nodes_decoder']: latent_nodes,
                self.ops['latent_node_symbols']: latent_node_symbol,
                self.placeholders['adjacency_matrix']: [elements['adj_mat']],
                self.placeholders['node_mask']: [elements['mask']],
                self.placeholders['graph_state_keep_prob']: 1,
                self.placeholders['edge_weight_dropout_keep_prob']: 1,
                self.placeholders['iteration_mask']: [[1]],
                self.placeholders['out_layer_dropout_keep_prob']: 1.0,
                self.placeholders['max_iteration_num']: 1,
            }

    """
    Prepare the feed dict for accessing the sampling point in the latent space
    """
    def get_dynamic_mean_feed_dict(self, elements, num_vertices, latent_points, is_generative):
            return {
                self.placeholders['z_prior']: latent_points,  # [hl]
                self.placeholders['num_vertices']: num_vertices,  # v
                self.placeholders['node_mask']: [elements['mask']],
                self.placeholders['node_symbols']: [elements['init']],
                self.placeholders['adjacency_matrix']: [elements['adj_mat']],
                self.placeholders['graph_state_keep_prob']: 1,
                self.placeholders['edge_weight_dropout_keep_prob']: 1,
                self.placeholders['iteration_mask']: [[1]],
                self.placeholders['out_layer_dropout_keep_prob']: 1.0,
                self.placeholders['max_iteration_num']: 1,
                self.placeholders['is_generative']: is_generative,
            }

    def get_node_symbol(self, batch_feed_dict):
        fetch_list = [self.ops['initial_nodes_decoder'], self.ops['node_symbol_prob'], self.ops['sampled_atoms']]
        result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
        return result

    def node_symbol_one_hot(self, sampled_node_symbol, real_n_vertices, max_n_vertices):
        one_hot_representations=[]
        for idx in range(max_n_vertices):
            representation = [0] * self.params["num_symbols"]
            if idx < real_n_vertices:
                atom_type=sampled_node_symbol[idx]
                representation[atom_type]=1
            one_hot_representations.append(representation)
        return one_hot_representations

    def search_and_generate_molecule(self, valences,
                             sampled_node_symbol, real_n_vertices,
                             elements, max_n_vertices, latent_nodes):
        # New molecule
        new_mol = Chem.MolFromSmiles('')
        new_mol = Chem.rdchem.RWMol(new_mol)
        # Add atoms
        add_atoms(new_mol, sampled_node_symbol, self.params["dataset"])
        # Add edges

        sampled_node_symbol_one_hot = self.node_symbol_one_hot(sampled_node_symbol, real_n_vertices, max_n_vertices)

        # get feed_dict
        feed_dict=self.get_dynamic_edge_feed_dict(elements, latent_nodes, [sampled_node_symbol_one_hot], max_n_vertices)
        # fetch nn predictions
        fetch_list = [self.ops['edge_predictions'], self.ops['edge_type_predictions']]
        edge_probs, edge_type_probs = self.sess.run(fetch_list, feed_dict=feed_dict)
        edge_probs = edge_probs[0]
        edge_probs_bin = edge_probs > 0.5
        edge_type_probs = edge_type_probs[0]
        for row in range(len(edge_probs[0])):
            for col in range(row+1, len(edge_probs[0])):  # only half matrix
                if edge_probs_bin[row, col] == True:
                    # choose an edge type
                    if not self.params["use_argmax_bonds"]:
                        bond=np.random.choice(np.arange(self.num_edge_types),p=edge_type_probs[:, row, col])
                    else:
                        bond=np.argmax(edge_type_probs[:, row, col])
                    # add the bond
                    new_mol.AddBond(int(row), int(col), number_to_bond[bond])


        # Remove unconnected node
        remove_extra_nodes(new_mol)
        new_mol=Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
        return new_mol

    def gradient_ascent(self, random_normal_states, derivative_z_sampled):        
        return random_normal_states + self.params['prior_learning_rate'] * derivative_z_sampled

    """
    Optimization in latent space. Generate one molecule for each optimization step.
    """
    def optimization_over_prior(self, random_normal_states, num_vertices, generated_all_similes, elements, count, is_generating):
        # record how many optimization steps are taken
        step=0
        # generate a new molecule
        self.generate_graph_with_state(random_normal_states, num_vertices, generated_all_similes, elements, step, count, is_generating)
        fetch_list = [self.ops['derivative_z_sampled'], self.ops['qed_computed_values'], self.ops['l2_loss']]
        for _ in range(self.params['optimization_step']):   
            # get current qed and derivative
            batch_feed_dict = self.get_dynamic_feed_dict(elements, None, None, num_vertices, None,
                                       None, None, None, None,
                                       random_normal_states, is_generating)
            derivative_z_sampled, qed_computed_values, l2_loss= self.sess.run(fetch_list, feed_dict=batch_feed_dict)
            # update the states
            random_normal_states=self.gradient_ascent(random_normal_states, 
                                                      derivative_z_sampled[0])
            # generate a new molecule
            step+=1
            self.generate_graph_with_state(random_normal_states, num_vertices,
                                           generated_all_similes, elements, step, count, is_generating)
        return random_normal_states

    def generate_graph_with_state(self, random_normal_states, num_vertices,
                                  generated_all_similes, elements, step, count, is_generating):
        # Get back node symbol predictions
        # Prepare dict
        node_symbol_batch_feed_dict = self.get_dynamic_feed_dict(elements, None, None,
                                     num_vertices, None, None, None, None, None, random_normal_states, is_generating)
        # Get predicted node probabilities
        [latent_nodes, predicted_node_symbol_prob, real_values] = self.get_node_symbol(node_symbol_batch_feed_dict)
        # Node numbers for each graph
        real_length = get_graph_length([elements['mask']])[0]  # [valid_node_number]
        sampled_node_symbol = np.squeeze(real_values)[:real_length]
        # Maximum valences for each node
        valences = get_initial_valence(sampled_node_symbol, self.params["dataset"]) # [v]


        # generate a new molecule
        new_mol = self.search_and_generate_molecule(np.copy(valences), sampled_node_symbol, real_length,
                                            elements, num_vertices, latent_nodes)

        generated_all_similes.append(Chem.MolToSmiles(new_mol))
        # print(Chem.MolToSmiles(best_mol))  # TODO: pr
        # exit(0)  # TODO: exit

        n_gen_max = self.params['number_of_generation']
        n_gen_cur = len(generated_all_similes)
        print("Molecules generated: ", n_gen_cur, end='\r')
        # give and indication about the number of generated molecules
        if (n_gen_cur % (n_gen_max / 100.0)) == 0:
            suff = "_" + self.params['suffix'] if self.params['suffix'] is not None else ""
            mask = "_masked" if self.params['use_mask'] else "_noMask"
            log_dir = self.params['log_dir']
            priors_file = log_dir + "/" + str(dataset) + "_decoded_generation_" + str(self.params["kl_trade_off_lambda"])\
                          + mask + suff + ".txt"
            f = open(priors_file, "a")
            f.writelines("Number of generated molecules: " + str(n_gen_cur) + "\n")
            f.close()

        if n_gen_cur >= n_gen_max:
            suff = "_" + self.params['suffix'] if self.params['suffix'] is not None else ""
            mask = "_masked" if self.params['use_mask'] else "_noMask"
            log_dir = self.params['log_dir']
            priors_file = log_dir + "/" + str(dataset) + "_decoded_generation_" + str(self.params["kl_trade_off_lambda"])\
                          + mask + suff + ".txt"
            generated = np.reshape(generated_all_similes, (1000, -1))
            f = open(priors_file, "w")
            for line in generated:
                for res in line:
                    f.write(res)
                    f.write(";,;")
                f.write("\n")
            f.close()
            print("Generation done")
            exit(0)

    def compensate_node_length(self, elements, bucket_size):
        maximum_length=bucket_size+self.params["compensate_num"]
        real_length=get_graph_length([elements['mask']])[0]+self.params["compensate_num"]
        elements['mask']=[1]*real_length + [0]*(maximum_length-real_length)
        elements['init']=np.zeros((maximum_length, self.params["num_symbols"]))
        elements['adj_mat']=np.zeros((self.num_edge_types, maximum_length, maximum_length))
        return maximum_length

    def generate_new_graphs(self, data):
        # bucketed: data organized by bucket
        (bucketed, bucket_sizes, bucket_at_step) = data
        bucket_counters = defaultdict(int)
        # all generated similes
        generated_all_similes=[]
        # counter
        count = 0
        # shuffle the lengths
        np.random.shuffle(bucket_at_step)
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step] # bucket number
            # data index
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            # batch data
            elements_batch = bucketed[bucket][start_idx:end_idx]
            for elements in elements_batch:
                # compensate for the length during generation
                # (this is a result that BFS may not make use of all candidate nodes during generation)
                maximum_length = self.compensate_node_length(elements, bucket_sizes[bucket])
                # initial state
                random_normal_states=generate_std_normal(1, maximum_length, self.params['hidden_size_encoder']) # [1, h]
                random_normal_states = self.optimization_over_prior(random_normal_states, maximum_length, generated_all_similes,
                                                                    elements, count, True)
                count+=1
            bucket_counters[bucket] += 1

    def reconstruction_new_graphs(self, num_vertices, generated_all_similes, elements):
        all_decoded = []
        # add the original molecule as first one in the list
        all_decoded.append(elements['smiles'])
        # print("True SMILES", elements['smiles'], "Hist", elements['hist'])  # TODO: pr
        for n_en in range(self.params['reconstruction_en']):
            # take latent from the input encoding or from prior
            random_normal_states = generate_std_normal(1, num_vertices, self.params['hidden_size_encoder'])  # [1, h]
            # is generative is always false here due to the sampling in the latent space
            feed_dict = self.get_dynamic_mean_feed_dict(elements, num_vertices, random_normal_states, False)  # always false
            # get the latent point according to the encoder distribution
            fetch_list = [self.ops['z_sampled']]
            [latent_point] = self.sess.run(fetch_list, feed_dict=feed_dict)
            # print("random_normal_states: ", random_normal_states)  # TODO: pr
            # print("data type", type(random_normal_states[0][0][0]))  # TODO: pr
            # print("z_sampled: ", latent_point)  # TODO: pr
            # print("data type", type(latent_point[0][0][0]))  # TODO: pr
            for n_dn in range(self.params['reconstruction_dn']):
                # Get back node symbol predictions
                # Prepare dict
                node_symbol_batch_feed_dict = self.get_dynamic_nodes_feed_dict(elements, num_vertices, latent_point, False)  # always false here
                # Get predicted node probabilities
                [latent_nodes, predicted_node_symbol_prob, real_values] = self.get_node_symbol(node_symbol_batch_feed_dict)
                # Node numbers for each graph
                real_length = get_graph_length([elements['mask']])[0]
                sampled_node_symbol = np.squeeze(real_values)[:real_length]
                # Maximum valences for each node
                valences = get_initial_valence(sampled_node_symbol, self.params["dataset"])  # [v]
                # randomly pick the starting point or use zero
                if self.params["path_random_order"]:
                    starting_point = random.choice(list(range(real_length)))  # randomly choose one
                else:
                    starting_point = 0
                # print("latent_nodes: ", latent_nodes)  # TODO: pr
                # print("latent_nodes[0][0][0]: ", str(latent_nodes[0][0][0]))  # TODO: pr
                # print("data type", type(latent_nodes[0][0][0]))  # TODO: pr
                # print("real_values: ", real_values)  # TODO: pr
                # print("real_length: ", real_length)  # TODO: pr
                # print("sampled_node_symbol: ", sampled_node_symbol)  # TODO: pr
                # print("valences: ", valences)  # TODO: pr
                # print("starting_point: ", starting_point)  # TODO: pr
                new_mol = self.search_and_generate_molecule(np.copy(valences),
                                                                            sampled_node_symbol, real_length,
                                                                            elements, num_vertices,
                                                                            latent_nodes)
                if new_mol is None:
                    new_mol = "None"
                all_decoded.append(Chem.MolToSmiles(new_mol))
                # print(Chem.MolToSmiles(new_mol))  # TODO: pr

        generated_all_similes.append(all_decoded)
        # print("Decoded SMILES", all_decoded)  # TODO: pr
        # exit(0)  #TODO: exit

    def reconstruction(self, data):
        (bucketed, bucket_sizes, bucket_at_step) = data
        bucket_counters = defaultdict(int)
        # all generated similes
        generated_all_similes = []
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]  # bucket number
            # data index
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            # batch data
            elements_batch = bucketed[bucket][start_idx:end_idx]

            if self.params["use_rec_multi_threads"]:
                thr = []
                for elements in elements_batch:
                    maximum_length = bucket_sizes[bucket]
                    # initial state
                    thr.append(ThreadWithReturnValue(target=self.reconstruction_new_graphs, args=(maximum_length, generated_all_similes, elements)))
                [t.join() for t in thr]
            else:
                for elements in elements_batch:
                    maximum_length = bucket_sizes[bucket]
                    # initial state
                    self.reconstruction_new_graphs(maximum_length, generated_all_similes, elements)

            print("Molecules reconstructed: ", len(generated_all_similes), end='\r')
            # exit(0)
            bucket_counters[bucket] += 1

        suff = "_" + self.params['suffix'] if self.params['suffix'] is not None else ""
        mask = "_masked" if self.params['use_mask'] else "_noMask"
        parent = "(" + str(self.params["reconstruction_en"]) + ":" + str(self.params["reconstruction_dn"]) + ")"
        log_dir = self.params['log_dir']
        recon_file = log_dir + "/" + str(dataset) + "_decoded_reconstruction_" + parent + "_" + str(self.params["kl_trade_off_lambda"]) + mask + suff + ".txt"
        f = open(recon_file, "w")
        for line in generated_all_similes:
            for res in line:
                f.write(res)
                f.write(";,;")
            f.write("\n")
        f.close()
        print('Reconstruction done')
        exit(0)

    def make_minibatch_iterator(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)
        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            batch_data = self.make_batch(elements, bucket_sizes[bucket])

            num_graphs = len(batch_data['init'])
            initial_representations = batch_data['init']
            batch_feed_dict = {
                self.placeholders['node_symbols']: batch_data['init'],
                self.placeholders['target_values']: np.transpose(batch_data['labels'], axes=[1,0]),
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs,
                self.placeholders['num_vertices']: bucket_sizes[bucket],
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
                self.placeholders['node_mask']: batch_data['node_mask'],
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_dropout_keep_prob,
                self.placeholders['iteration_mask']: batch_data['iteration_mask'],
                self.placeholders['incre_adj_mat']: batch_data['incre_adj_mat'],
                self.placeholders['distance_to_others']: batch_data['distance_to_others'],
                self.placeholders['node_sequence']: batch_data['node_sequence'],
                self.placeholders['edge_type_masks']: batch_data['edge_type_masks'],
                self.placeholders['edge_type_labels']: batch_data['edge_type_labels'],
                self.placeholders['edge_masks']: batch_data['edge_masks'],
                self.placeholders['edge_labels']: batch_data['edge_labels'],
                self.placeholders['local_stop']: batch_data['local_stop'],
                self.placeholders['max_iteration_num']: batch_data['max_iteration_num'],
                self.placeholders['kl_trade_off_lambda']: self.params['kl_trade_off_lambda'],
                self.placeholders['overlapped_edge_features']: batch_data['overlapped_edge_features'],
                self.placeholders['histograms']: self.histograms['train'][0],  # TODO: rember to change if it is needed
                self.placeholders['n_histograms']: self.histograms['train'][1],  # TODO: rember to change if it is needed
                self.placeholders['hist']: batch_data['hist']
            }
            bucket_counters[bucket] += 1
            # print(batch_data['smiles'])  # TODO: pr
            # print(batch_data['init'])  # TODO: pr
            yield batch_feed_dict

if __name__ == "__main__":
    args = docopt(__doc__)
    start = time.time()
    dataset = args.get('--dataset')
    model = MolGVAE(args)
    try:
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
    finally:
        end = time.time()
        print("Time for the overall execution: " + model.get_time_diff(end, start))

