#!/usr/bin/env/python
"""
Usage:
    make_dataset.py [options]

Options:
    -h --help        Show this screen.
    --file NAME      File
    --dataset NAME   Dataset
    --filter INT     Numero
"""

import sys
import os
from docopt import docopt
from rdkit import Chem
from rdkit.Chem import QED
import json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import utils
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# get current directory in order to work with full path and not dynamic
current_dir = os.path.dirname(os.path.realpath(__file__))



# add one edge to adj matrix
def add_edge_mat(amat, src, dest, e):
    amat[e, dest, src] = 1
    amat[e, src, dest] = 1

def graph_to_adj_mat(graph, max_n_vertices, num_edge_types):
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    for src, e, dest in graph:
        add_edge_mat(amat, src, dest, e)
    return amat

def load_filter(data, number):
    res = []
    for n, s in enumerate(data):
        m =  Chem.MolFromSmiles(s['smiles'])
        atoms = m.GetAtoms()
        if len(atoms) == number:
            res.append(s)
    return res

def load_data(file_name, number):
    print("Loading data from %s" % file_name)
    with open(file_name, 'r') as f:
        data = json.load(f)

    if number is not None:
        data = load_filter(data, int(number))

    return data


def count_molecules(data, dataset):
    if str.lower(dataset) == 'qm9':
        limit = 12
    else:
        limit = 40
    # groups molecules based on the number of atoms
    g_mol = [[] for i in range(limit)]
    for n, s in enumerate(data):
        m =  Chem.MolFromSmiles(s['smiles'])
        atoms = m.GetAtoms()
        g_mol[len(atoms)].append(n)
    print('Grouped molecules: ', [(i, len(mols)) for i, mols in enumerate(g_mol)])
    n_g_mol = [len(mols) for mols in g_mol]
    sns.scatterplot(range(len(n_g_mol)), n_g_mol)
    plt.title("Number of molecules")
    plt.xlabel("Molecules by number of atoms")
    sns.despine(offset=True)
    plt.show()

def count_number_atoms(data, dataset):
    g_mol = {s:0 for s in utils.dataset_info(dataset)['atom_types']}
    for n, s in enumerate(data):
        m = Chem.MolFromSmiles(s['smiles'])
        atoms = m.GetAtoms()
        for atom in atoms:
            if dataset == 'qm9':
                atom_str = atom.GetSymbol()
            else:
                # zinc dataset # transform using "<atom_symbol><valence>(<charge>)"  notation
                symbol = atom.GetSymbol()
                valence = atom.GetTotalValence()
                charge = atom.GetFormalCharge()
                atom_str = "%s%i(%i)" % (symbol, valence, charge)

                if atom_str not in g_mol.keys():
                    print('Unrecognized atom type %s' % atom_str)
                    return None
            g_mol[atom_str] += 1
    atom_sum = sum([i for i in g_mol.values()])
    print('Types of atoms: ', g_mol)
    if atom_sum > 0:
        print('Types of atoms %: ', {i:v/atom_sum for i, v in g_mol.items()})
    else:
        print('Types of atoms %: ', 'division by 0')


def count_edges(data):
    num_fwd_edge_types = len(utils.bond_dict) - 1
    num_edge_types = num_fwd_edge_types
    edge_type = np.zeros(num_fwd_edge_types + 1)

    for s in data:
        n_atoms = len(s['node_features'])
        smiles = s['smiles']
        adj_mat = graph_to_adj_mat(s['graph'], n_atoms, num_edge_types)
        no_edge = 1 - np.sum(adj_mat, axis=0, keepdims=True)
        adj_mat = np.concatenate([no_edge, adj_mat], axis=0)
        for edge in range(num_fwd_edge_types + 1):
            tmp_sum = np.sum(adj_mat[edge, :, :])
            edge_type[edge] += tmp_sum

    print('Types of edges: ', edge_type)
    edge_sum = sum(edge_type)
    if edge_sum > 0:
        print('Types of edges %: ', [v / edge_sum for v in edge_type])
    else:
        print('Types of edges %: ', 'division by 0')



if __name__ == "__main__":
    args = docopt(__doc__)
    file = args.get('--file')
    dataset = args.get('--dataset')
    filter = args.get('--filter')

    data = load_data(file, filter)

    count_molecules(data, dataset)
    count_number_atoms(data, dataset)
    count_edges(data)

