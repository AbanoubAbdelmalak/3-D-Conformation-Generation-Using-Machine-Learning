"""
Python qm9_process.py

Description: This file contains the preprocessing and adaptation of qm9
             dataset into pytorch geometric. The building of the dataset 
             is adopted from the work done by GeoMol with slight changes
             to make it compatable with other modules and avoid errors that
             were happening due to inconcistencis in the dataset version 
             that was built by GeoMol. The target here is to provide a 
             ready-to-use dataset that can be injected to any Pytorch-Geometric 
             based code and run without errors. The features used are the ones 
             presented by the GeoMol paper.

Author: Abanoub Abdelmalak

Date Created: May 1, 2023

"""

import glob
import networkx as nx
import numpy as np
import os.path as osp
import pickle
import random
import torch
import torch.nn.functional as F
import torch_geometric as tg
from sklearn.utils import shuffle
from torch_geometric.data import Dataset, Data, DataLoader, InMemoryDataset
from torch_geometric.utils import degree
from torch_scatter import scatter
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Optional
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import HybridizationType
from torch.nn.functional import one_hot

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}


def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

class qm9_data(InMemoryDataset):
    """
    This is a class that extends the InMemoryDataset class from pytorch that will help later to 
    batch the data for training so this class will take the path where the data is an 
    stores each graph as a single data object.
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.all_files = sorted(glob.glob(osp.join(root+'/qm9/', '*.pickle')))
        self.pickle_files = [f for i, f in enumerate(self.all_files)]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    def process(self):
        data_list = []
        counter = 0
        for i in tqdm(self.pickle_files):
            data = featurization(i, counter)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            if not(data == None):
                if isinstance(data, list):
                    for i in data:
                        data_list.append(i)
                        counter = counter + 1
                else:
                    data_list.append(data)
                    counter = counter + 1
            if counter == 500:
                break
            
        
        #data = self.collate(data_list)
        
        torch.save(self.collate(data_list), self.processed_paths[0])
        print("DataList length is: "+str(len(data_list)))
        
    
    
    def open_pickle(self, mol_path):
        # reads the mol pickle file and return the dictionary containing the data
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic
    
    @property
    def raw_file_names(self) -> List[str]:
        return self.all_files

    @property
    def processed_file_names(self):
        return ['data_v3.pt']
    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict
    
    def collate(self, data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
        keys = ['x', 'z', 'pos', 'edge_index', 'edge_attr', 'chiral_tag', 'name', 'boltzmann_weight', 'degeneracy', 'mol', 'pos_mask']
        data_dict = {key: [] for key in keys}
        slices = {key: [] for key in keys}

        for item in data_list:
            for key in keys:
                data = item[key]
                #print(key)
                #print(data)
                if isinstance(data, Tensor):
                    if data.dim() == 0:  # Handle tensors with no dimensions
                        data = data.unsqueeze(0)
                    data_dict[key].append(data)
                elif isinstance(data, str):
                    data_dict[key].append(data)
                elif key == 'mol':
                    # Serialize RDKit molecule as bytes
                    #mol_bytes = pickle.dumps(data)
                    # Encode bytes as base64 string
                    #mol_str = base64.b64encode(mol_bytes).decode('utf-8')
                    data_dict[key].append(data)
                elif isinstance(data, float) or isinstance(data, int):
                    data_dict[key].append(torch.tensor(data).unsqueeze(0))
                    slices[key].append(1)
                else:
                    data_dict[key].append([data])

                if key == 'edge_index':
                    slices[key].append(data.size(1))
                else:
                    slices[key].append(data.size(0) if isinstance(data, Tensor) else 1)

        data = Data(x=torch.cat(data_dict['x'], dim=0),
                    z=torch.cat(data_dict['z'], dim=0),
                    pos=torch.cat(data_dict['pos'], dim=0),
                    edge_index=torch.cat(data_dict['edge_index'], dim=1),
                    edge_attr=torch.cat(data_dict['edge_attr'], dim=0),
                    chiral_tag=torch.cat(data_dict['chiral_tag'], dim=0),
                    name=data_dict['name'],
                    boltzmann_weight=torch.cat(data_dict['boltzmann_weight'], dim=0),
                    degeneracy=torch.cat(data_dict['degeneracy'], dim=0),
                    mol=data_dict['mol'],
                    pos_mask=torch.cat(data_dict['pos_mask'], dim=0), dim=0)

        slices = {key: torch.tensor(slices[key]).cumsum(0) for key in slices.keys()}

        return data, slices
    
def featurization(mol_path: str):
    chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

    dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    file = open(mol_path, 'rb')
    mol_dic = pickle.load(file)

    file.close()
    #save conformers of the moelcule
    confs = mol_dic['conformers']
    name = mol_dic["smiles"]
    confs = mol_dic['conformers']
    random.shuffle(confs)  # shuffle confs
    name = mol_dic["smiles"]
    max_confs=10
    # filter mols rdkit can't intrinsically handle
    mol_ = Chem.MolFromSmiles(name)
    if mol_:
        canonical_smi = Chem.MolToSmiles(mol_)
    else:
        return None

    # skip conformers with fragments
    if '.' in name:
        return None

    # skip conformers without dihedrals
    N = confs[0]['rd_mol'].GetNumAtoms()
    if N < 4:
        return None
    if confs[0]['rd_mol'].GetNumBonds() < 4:
        return None
    if not confs[0]['rd_mol'].HasSubstructMatch(dihedral_pattern):
        return None

    pos = torch.zeros([max_confs, N, 3])
    pos_mask = torch.zeros(max_confs, dtype=torch.int64)
    k = 0
    for conf in confs:
        mol = conf['rd_mol']

        # skip mols with atoms with more than 4 neighbors for now
        n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if np.max(n_neighbors) > 4:
            continue

        # filter for conformers that may have reacted
        try:
            conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception as e:
            continue

        if conf_canonical_smi != canonical_smi:
            continue

        pos[k] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
        pos_mask[k] = 1
        k += 1
        correct_mol = mol
        if k == 10:
            break

    # return None if no non-reactive conformers were found
    if k == 0:
        return None

    type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    neighbor_dict = {}
    ring = correct_mol.GetRingInfo()
    for i, atom in enumerate(correct_mol.GetAtoms()):
        type_idx.append(types[atom.GetSymbol()])
        n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(n_ids) > 1:
            neighbor_dict[i] = torch.tensor(n_ids)
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.extend([atom.GetAtomicNum(),
                                1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
                                Chem.rdchem.HybridizationType.SP,
                                Chem.rdchem.HybridizationType.SP2,
                                Chem.rdchem.HybridizationType.SP3,
                                Chem.rdchem.HybridizationType.SP3D,
                                Chem.rdchem.HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                int(ring.IsAtomInRingOfSize(i, 4)),
                                int(ring.IsAtomInRingOfSize(i, 5)),
                                int(ring.IsAtomInRingOfSize(i, 6)),
                                int(ring.IsAtomInRingOfSize(i, 7)),
                                int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)
    chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

    row, col, edge_type, bond_features = [], [], [], []
    for bond in correct_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bt = tuple(sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
        bond_features += 2 * [int(bond.IsInRing()),
                                int(bond.GetIsConjugated()),
                                int(bond.GetIsAromatic())]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    # bond_features = torch.tensor(bond_features, dtype=torch.float).view(len(bond_type), -1)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    # edge_attr = torch.cat([edge_attr[perm], bond_features], dim=-1)
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    #num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)
    y=conf['boltzmannweight']
    if len(pos)>1:
        data_list = []
        for i in pos:
            if not(torch.all(torch.eq(i, 0))):
                data = Data(x=x, z=z, pos=i, edge_index=edge_index, edge_attr=edge_attr,  
                    chiral_tag=chiral_tag, name=name, boltzmann_weight=conf['boltzmannweight'],
                    degeneracy=conf['degeneracy'], mol=correct_mol, pos_mask=pos_mask, y=y)
                data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)
                data_list.append(data)
        return data_list
    else:
        if not(torch.all(torch.eq(pos[0], 0))):
            data = Data(x=x, z=z, pos=pos[0], edge_index=edge_index, edge_attr=edge_attr, 
                        chiral_tag=chiral_tag, name=name, boltzmann_weight=conf['boltzmannweight'],
                        degeneracy=conf['degeneracy'], mol=correct_mol, pos_mask=pos_mask,y=y)
        
        data.edge_index_dihedral_pairs = get_dihedral_pairs(data.edge_index, data=data)
        return data

def get_cycle_values(cycle_list, start_at=None):
    start_at = 0 if start_at is None else cycle_list.index(start_at)
    while True:
        yield cycle_list[start_at]
        start_at = (start_at + 1) % len(cycle_list)


def get_cycle_indices(cycle, start_idx):
    cycle_it = get_cycle_values(cycle, start_idx)
    indices = []

    end = 9e99
    start = next(cycle_it)
    a = start
    while start != end:
        b = next(cycle_it)
        indices.append(torch.tensor([a, b]))
        a = b
        end = b

    return indices
def get_current_cycle_indices(cycles, cycle_check, idx):
    c_idx = [i for i, c in enumerate(cycle_check) if c][0]
    current_cycle = cycles.pop(c_idx)
    current_idx = current_cycle[(np.array(current_cycle) == idx.item()).nonzero()[0][0]]
    return get_cycle_indices(current_cycle, current_idx)

def get_dihedral_pairs(edge_index, data):
    """
    Given edge indices, return pairs of indices that we must calculate dihedrals for
    """
    start, end = edge_index
    degrees = degree(end)
    dihedral_pairs_true = torch.nonzero(torch.logical_and(degrees[start] > 1, degrees[end] > 1))
    dihedral_pairs = edge_index[:, dihedral_pairs_true].squeeze(-1)
    
    # # first method which removes one (pseudo) random edge from a cycle
    dihedral_idxs = torch.nonzero(dihedral_pairs.sort(dim=0).indices[0, :] == 0).squeeze().detach().cpu().numpy()

    # prioritize rings for assigning dihedrals
    dihedral_pairs = dihedral_pairs.t().cpu().detach()[dihedral_idxs]
    G = nx.to_undirected(tg.utils.to_networkx(data))
    cycles = nx.cycle_basis(G)
    keep, sorted_keep = [], []

    if len(dihedral_pairs.shape) == 1:
        dihedral_pairs = dihedral_pairs.unsqueeze(0)

    for pair in dihedral_pairs:
        x, y = pair

        if sorted(pair) in sorted_keep:
            continue

        y_cycle_check = [y in cycle for cycle in cycles]
        x_cycle_check = [x in cycle for cycle in cycles]

        if any(x_cycle_check) and any(y_cycle_check):  # both in new cycle
            cycle_indices = get_current_cycle_indices(cycles, x_cycle_check, x)
            keep.extend(cycle_indices)

            sorted_keep.extend([sorted(c.cpu()) for c in cycle_indices])
            continue

        if any(y_cycle_check):
            cycle_indices = get_current_cycle_indices(cycles, y_cycle_check, y)
            keep.append(pair)
            keep.extend(cycle_indices)

            sorted_keep.append(sorted(pair))
            sorted_keep.extend([sorted(c.cpu()) for c in cycle_indices])
            continue

        keep.append(pair)

    #keep = torch.tensor(keep).to(device) 
    keep = [torch.tensor(t).to(device) for t in keep]
    
    return torch.stack(keep).t()