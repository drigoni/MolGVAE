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

# get current directory in order to work with full path and not dynamic
current_dir = os.path.dirname(os.path.realpath(__file__))

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


def count_molecules(data):
    # groups molecules based on the number of atoms
    g_mol = [[] for i in range(40)]
    for n, s in enumerate(data):
        m =  Chem.MolFromSmiles(s['smiles'])
        atoms = m.GetAtoms()
        g_mol[len(atoms)].append(n)
    print('Grouped molecules: ', [(i, len(mols)) for i, mols in enumerate(g_mol)])

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
    edge_type_this_molecule = [0] * len(utils.bond_dict)
    for s in data:
        for edge in s['graph']:
            edge_type = edge[1]
            edge_type_this_molecule[edge_type] += 1
    edge_sum = sum(edge_type_this_molecule)
    print('Types of edges: ', edge_type_this_molecule)
    if edge_sum > 0:
        print('Types of edges %: ', [v / edge_sum for v in edge_type_this_molecule])
    else:
        print('Types of edges %: ', 'division by 0')



if __name__ == "__main__":
    args = docopt(__doc__)
    file = args.get('--file')
    dataset = args.get('--dataset')
    filter = args.get('--filter')

    data = load_data(file, filter)

    count_molecules(data)
    count_number_atoms(data, dataset)
    count_edges(data)

