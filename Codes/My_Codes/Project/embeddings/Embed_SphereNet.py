
"""
Python EmbedSphereNet.py

Description: This file contains a class EmbedSphereNet which extends SphereNet
            class from DIG library. The purpose here is to use it later as a 
            module together with GeoMol. The idea is to change its output to be
            the embeddings of SphereNet.It is still a work in progress.

Author: Abanoub Abdelmalak

Date Created: May 1, 2023

"""
from dig.threedgraph.method.spherenet import SphereNet
from dig.threedgraph.utils import xyz_to_dat
from torch_geometric.nn import radius_graph
import torch
from torch_scatter import scatter

class ExtendedSphereNet(SphereNet):
    def __init__(self, *args, **kwargs):
        super(ExtendedSphereNet, self).__init__(*args, **kwargs)

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes=z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)

        emb = self.emb(dist, angle, torsion, idx_kj)

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)

        embeddings = {
            'node': [v.clone()],
            'edge': [e.clone()],
            'graph': [u.clone()]
        }

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)
            u = update_u(u, v, batch)

            embeddings['node'].append(v.clone())
            embeddings['edge'].append(e.clone())
            embeddings['graph'].append(u.clone())

        return embeddings
