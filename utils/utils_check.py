#!/usr/bin/env/python
import numpy as np
import pickle
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import Chem
from collections import defaultdict
import os
import planarity
from utils import sascorer
from utils import to_graph
from utils import bond_dict
from utils import dataset_info
from utils import Graph
from rdkit.Chem import Crippen
from rdkit.Chem import QED


def check_edge_prob(dataset):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, \
        qed_prediction, qed_labels,mean, logvariance = pickle.load(f)
    for ep, epl in zip(edge_prob, edge_prob_label):
        print("prediction")
        print(ep)
        print("label")
        print(epl)


# check whether a graph is planar or not
def is_planar(location, adj_list, is_dense=False):
    if is_dense:
        new_adj_list=defaultdict(list)
        for x in range(len(adj_list)):
            for y in range(len(adj_list)):
                if adj_list[x][y]==1:
                    new_adj_list[x].append((y,1))
        adj_list=new_adj_list
    edges= []
    seen= set()
    for src, l in adj_list.items():
        for dst, e in l:
            if (dst, src) not in seen:
                edges.append((src,dst))
                seen.add((src,dst))
    edges += [location, (location[1], location[0])]
    return planarity.is_planar(edges)


def check_edge_type_prob(dataset):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, \
        qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    for ep, epl in zip(edge_type_prob, edge_type_label):
        print("prediction")
        print(ep)
        print("label")
        print(epl)


def check_mean(dataset):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, \
        qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    print(mean.tolist()[:40])


def check_variance(dataset, filter=None):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, \
        qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    print(np.exp(logvariance).tolist()[:40])


def check_node_prob(dataset, filter=None):
    print(dataset)
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, \
        qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    print(node_symbol_prob[0])
    print(node_symbol[0])
    print(node_symbol_prob.shape)


def check_qed(dataset, filter=None):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, \
        qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    print(qed_prediction)
    print(qed_labels[0])
    print(np.mean(np.abs(qed_prediction-qed_labels[0])))


def generate_empty_adj_matrix(maximum_vertice_num):
    return np.zeros((1, 3, maximum_vertice_num, maximum_vertice_num))


def check_validity(dataset):
    with open('generated_smiles_%s.txt' % dataset, 'rb') as f:
        all_smiles=set(pickle.load(f))
    count=0
    for smiles in all_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            count+=1
    return len(all_smiles), count


def dump(file_name, content):
    with open(file_name, 'wb') as out_file:
        pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)


def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('made directory %s' % path)


def visualize_mol(path, new_mol):
    AllChem.Compute2DCoords(new_mol)
    print(path)
    Draw.MolToFile(new_mol, path)


def novelty_metric(dataset):
    with open('all_smiles_%s.pkl' % dataset, 'rb') as f:
        all_smiles = set(pickle.load(f))
    with open('generated_smiles_%s.txt' % dataset, 'rb') as f:
        generated_all_smiles = set(pickle.load(f))
    total_new_molecules = 0
    for generated_smiles in generated_all_smiles:
        if generated_smiles not in all_smiles:
            total_new_molecules += 1

    return float(total_new_molecules) / len(generated_all_smiles)


def count_edge_type(dataset, generated=True):
    if generated:
        filename = 'generated_smiles_%s.txt' % dataset
    else:
        filename = 'all_smiles_%s.pkl' % dataset
    with open(filename, 'rb') as f:
        all_smiles = set(pickle.load(f))

    counter = defaultdict(int)
    edge_type_per_molecule = []
    for smiles in all_smiles:
        nodes, edges = to_graph(smiles, dataset)
        edge_type_this_molecule = [0] * len(bond_dict)
        for edge in edges:
            edge_type = edge[1]
            edge_type_this_molecule[edge_type] += 1
            counter[edge_type] += 1
        edge_type_per_molecule.append(edge_type_this_molecule)
    total_sum = 0
    return len(all_smiles), counter, edge_type_per_molecule


def check_planar(dataset):
    with open("generated_smiles_%s" % dataset, 'rb') as f:
        all_smiles=set(pickle.load(f))
    total_non_planar=0
    for smiles in all_smiles:
        try:
            nodes, edges=to_graph(smiles, dataset)
        except:
            continue
        edges=[(src, dst) for src, e, dst in edges]
        if edges==[]:
            continue

        if not planarity.is_planar(edges):
            total_non_planar+=1
    return len(all_smiles), total_non_planar


def count_atoms(dataset):
    with open("generated_smiles_%s" % dataset, 'rb') as f:
        all_smiles=set(pickle.load(f))
    counter=defaultdict(int)
    atom_count_per_molecule=[] # record the counts for each molecule
    for smiles in all_smiles:
        try:
            nodes, edges=to_graph(smiles, dataset)
        except:
            continue
        atom_count_this_molecule=[0]*len(dataset_info(dataset)['atom_types'])
        for node in nodes:
            atom_type=np.argmax(node)
            atom_count_this_molecule[atom_type]+=1
            counter[atom_type]+=1
        atom_count_per_molecule.append(atom_count_this_molecule)
    total_sum=0

    return len(all_smiles), counter, atom_count_per_molecule


def check_uniqueness(dataset):
    with open('generated_smiles_%s.txt' % dataset, 'rb') as f:
        all_smiles=pickle.load(f)
    original_num = len(all_smiles)
    all_smiles=set(all_smiles)
    new_num = len(all_smiles)
    return new_num/original_num


# whether whether the graphs has no cycle or not
def check_cyclic(dataset, generated=True):
    if generated:
        with open("generated_smiles_%s" % dataset, 'rb') as f:
            all_smiles = set(pickle.load(f))
    else:
        with open("all_smiles_%s.pkl" % dataset, 'rb') as f:
            all_smiles = set(pickle.load(f))

    tree_count = 0
    for smiles in all_smiles:
        nodes, edges = to_graph(smiles, dataset)
        edges = [(src, dst) for src, e, dst in edges]
        if edges == []:
            continue
        new_adj_list = defaultdict(list)

        for src, dst in edges:
            new_adj_list[src].append(dst)
            new_adj_list[dst].append(src)
        graph = Graph(len(nodes), new_adj_list)
        if graph.isTree():
            tree_count += 1
    return len(all_smiles), tree_count


def check_sascorer(dataset):
    with open('generated_smiles_%s.txt' % dataset, 'rb') as f:
        all_smiles = set(pickle.load(f))
    sa_sum = 0
    total = 0
    sa_score_per_molecule = []
    for smiles in all_smiles:
        new_mol = Chem.MolFromSmiles(smiles)
        try:
            val = sascorer.calculateScore(new_mol)
        except:
            continue
        sa_sum += val
        sa_score_per_molecule.append(val)
        total += 1
    return sa_sum / total, sa_score_per_molecule


def check_logp(dataset):
    with open('generated_smiles_%s.txt' % dataset, 'rb') as f:
        all_smiles = set(pickle.load(f))
    logp_sum = 0
    total = 0
    logp_score_per_molecule = []
    for smiles in all_smiles:
        new_mol = Chem.MolFromSmiles(smiles)
        try:
            val = Crippen.MolLogP(new_mol)
        except:
            continue
        logp_sum += val
        logp_score_per_molecule.append(val)
        total += 1
    return logp_sum / total, logp_score_per_molecule


def check_qed(dataset):
    with open('generated_smiles_%s.txt' % dataset, 'rb') as f:
        all_smiles = set(pickle.load(f))
    qed_sum = 0
    total = 0
    qed_score_per_molecule = []
    for smiles in all_smiles:
        new_mol = Chem.MolFromSmiles(smiles)
        try:
            val = QED.qed(new_mol)
        except:
            continue
        qed_sum += val
        qed_score_per_molecule.append(val)
        total += 1
    return qed_sum / total, qed_score_per_molecule


def sssr_metric(dataset):
    with open('generated_smiles_%s.txt' % dataset, 'rb') as f:
        all_smiles = set(pickle.load(f))
    overlapped_molecule = 0
    for smiles in all_smiles:
        new_mol = Chem.MolFromSmiles(smiles)
        ssr = Chem.GetSymmSSSR(new_mol)
        overlap_flag = False
        for idx1 in range(len(ssr)):
            for idx2 in range(idx1 + 1, len(ssr)):
                if len(set(ssr[idx1]) & set(ssr[idx2])) > 2:
                    overlap_flag = True
        if overlap_flag:
            overlapped_molecule += 1
    return overlapped_molecule / len(all_smiles)
