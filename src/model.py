"""
@author: Truong Quoc Chien (chientrq92@gmail.com)

"""
import torch
import torch.nn.functional as F 
from torch_geometric.utils import scatter
from torch_geometric.nn import radius_graph
from torch_geometric.nn import TransformerConv
from torch.nn import Linear, BatchNorm1d, ModuleList, Sequential, ReLU, GRU
torch.manual_seed(2023)

#%% Graph Neural Networks inspired from DNA origami structure
class Nets(torch.nn.Module):
    def __init__(self, net_params):
        super(Nets, self).__init__()
        self.pos_dim   = net_params['model_pos_dim']
        self.ang_dim   = net_params['model_ang_dim']
        self.edge_dim  = net_params['model_edge_dim']
        edg_emb        = net_params['model_emb_dim']
        pos_emb        = net_params['model_emb_dim']
        ang_emb        = net_params['model_emb_dim']
        ele_emb        = net_params['model_ele_ebm_dim']
        heads          = net_params['model_attention_heads']
        self.processes = net_params['model_processes']
        dropout        = net_params['model_dropout_rate']
        
        self.pos_enc = Sequential(Linear(self.pos_dim, pos_emb//2), 
                                  ReLU(),
                                  Linear(pos_emb//2, pos_emb), 
                                  ReLU())
        self.ang_enc = Sequential(Linear(self.ang_dim, ang_emb//2), 
                                  ReLU(),
                                  Linear(ang_emb//2, ang_emb), 
                                  ReLU())
        self.edg_enc = Sequential(Linear(self.edge_dim, edg_emb//2), 
                                  ReLU(),
                                  Linear(edg_emb//2, edg_emb), 
                                  ReLU())
        # electrostatics interaction layer                          
        self.ele_interact    = Electro_interact(self.pos_dim, hidden_dim=ele_emb)
        
        # processor layers
        self.edge_layers     = ModuleList()
        self.node_layers     = ModuleList()
        self.bn_pos_layers   = ModuleList()
        self.bn_ang_layers   = ModuleList()
        self.pos_convs       = ModuleList()
        self.ang_convs       = ModuleList()
        self.pos_transfs     = ModuleList()
        self.ang_transfs     = ModuleList()
        
        for i in range(self.processes):
            self.edge_layers.append(EdgeBlock(pos_emb, ang_emb, edg_emb, 3*edg_emb, residual=False))
            self.node_layers.append(NodeBlock(pos_emb, ang_emb, edg_emb, 2*edg_emb, residual=True))
            
            self.bn_pos_layers.append(BatchNorm1d(pos_emb))
            self.bn_ang_layers.append(BatchNorm1d(ang_emb))
            
            self.pos_convs.append(TransformerConv(pos_emb, pos_emb, heads=heads, edge_dim=edg_emb))
            self.ang_convs.append(TransformerConv(ang_emb, ang_emb, heads=heads, edge_dim=edg_emb))
            
            self.pos_transfs.append(Linear(pos_emb*heads, pos_emb))
            self.ang_transfs.append(Linear(ang_emb*heads, ang_emb))
            
        self.pos_gru = GRU(pos_emb, pos_emb)
        self.ang_gru = GRU(ang_emb, ang_emb)
        
        self.pos_dec = Sequential(Linear(pos_emb, pos_emb//2),
                                  ReLU(),
                                  Linear(pos_emb//2, self.pos_dim))
        self.ang_dec = Sequential(Linear(ang_emb, ang_emb//2),
                                  ReLU(),
                                  Linear(ang_emb//2, self.ang_dim))

    def forward(self, data, coords=None):
        if coords is not None:
            pos, ang = coords[:,:3], coords[:,3:6]
            edge_index, edge_attr = data.edge_index, data.edge_attr[:,:self.edge_dim]
        else:  
            pos, ang, edge_index, edge_attr = data.x[:,:3], data.x[:,3:6], data.edge_index, data.edge_attr[:,:self.edge_dim]
        src = edge_index[0]
        dst = edge_index[1]
        helix = data.x[:,6].view(-1) 
        u         = self.pos_enc(pos)          
        phi       = self.ang_enc(ang)
        edge_attr = self.edg_enc(edge_attr)     
        h1 = u.unsqueeze(0)
        h2 = phi.unsqueeze(0)
        
        for i in range(self.processes):
            edge_attr = self.edge_layers[i](u[src], phi[src], u[dst], phi[dst], edge_attr)
            u, phi    = self.node_layers[i](u, phi, edge_index, edge_attr)
            u         = self.bn_pos_layers[i](u)
            phi       = self.bn_ang_layers[i](phi)
            u       = F.relu(self.pos_convs[i](u, edge_index, edge_attr))
            u       = self.pos_transfs[i](u)
            u, h1   = self.pos_gru(u.unsqueeze(0), h1)
            u       = u.squeeze(0)         
            phi     = F.relu(self.ang_convs[i](phi, edge_index, edge_attr))
            phi     = self.ang_transfs[i](phi)
            phi, h2 = self.ang_gru(phi.unsqueeze(0), h2)
            phi     = phi.squeeze(0)
        u   = self.pos_dec(u)
        u   = u + self.ele_interact(pos+u, helix)
        phi = self.ang_dec(phi)
        pos = pos + u
        ang = ang + phi
        return torch.cat((pos, ang), 1)

#%% Edge blocks
class EdgeBlock(torch.nn.Module):
    def __init__(self, pos_dim, ang_dim, edge_dim, hidden_dim, residual):
        super().__init__()
        self.residual = residual
        node_dim = pos_dim + ang_dim
        self.edge_mlp = Sequential(Linear(2*node_dim + edge_dim, hidden_dim), 
                                   ReLU(),
                                   BatchNorm1d(hidden_dim),
                                   Linear(hidden_dim, hidden_dim),
                                   ReLU(),
                                   Linear(hidden_dim, edge_dim))
    def forward(self, pos_src, ang_src, pos_dest, ang_dest, edge_attr):
        out = torch.cat([pos_src, ang_src, edge_attr, pos_dest, ang_dest], 1)
        out = self.edge_mlp(out)
        if self.residual:
            out = out + edge_attr
        return out

#%% Node blocks
class NodeBlock(torch.nn.Module):
    def __init__(self, pos_dim, ang_dim, edge_dim, hidden_dim, residual):
        super().__init__()
        self.residual = residual
        self.pos_mlp_1 = Sequential(Linear(pos_dim+edge_dim, hidden_dim), 
                                    ReLU(), 
                                    BatchNorm1d(hidden_dim),
                                    Linear(hidden_dim, hidden_dim),
                                    ReLU(), 
                                    Linear(hidden_dim, edge_dim),)
        self.pos_mlp_2 = Sequential(Linear(pos_dim+edge_dim, hidden_dim), 
                                    ReLU(), 
                                    BatchNorm1d(hidden_dim),
                                    Linear(hidden_dim, hidden_dim),
                                    ReLU(),
                                    Linear(hidden_dim, pos_dim),)    
        self.ang_mlp_1 = Sequential(Linear(ang_dim+edge_dim, hidden_dim), 
                                    ReLU(), 
                                    BatchNorm1d(hidden_dim),
                                    Linear(hidden_dim, hidden_dim),
                                    ReLU(), 
                                    Linear(hidden_dim, edge_dim),)
        self.ang_mlp_2 = Sequential(Linear(ang_dim+edge_dim, hidden_dim), 
                                    ReLU(), 
                                    BatchNorm1d(hidden_dim),
                                    Linear(hidden_dim, hidden_dim),
                                    ReLU(),
                                    Linear(hidden_dim, ang_dim),)      
    def forward(self, pos, ang, edge_index, edge_attr):
        row, col = edge_index
        u = torch.cat([pos[row], edge_attr], dim=1)
        u = self.pos_mlp_1(u)
        u = scatter(u, col, dim=0, reduce='mean', dim_size=pos.size(0)) 
        u = torch.cat([pos, u], dim=1)
        u = self.pos_mlp_2(u)
        phi = torch.cat([ang[row], edge_attr], dim=1)
        phi = self.ang_mlp_1(phi)
        phi = scatter(phi, col, dim=0, reduce='mean', dim_size=pos.size(0)) 
        phi = torch.cat([ang, phi], dim=1)
        phi = self.ang_mlp_2(phi)
        if self.residual:
            u = pos + u
            phi = ang + phi
        return u, phi

#%% Electrostatics interaction block
class Electro_interact(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.node_mlp_1 = Sequential(Linear(2*node_dim, hidden_dim), 
                                     ReLU(), 
                                     BatchNorm1d(hidden_dim),
                                     Linear(hidden_dim, node_dim),)   
        self.node_mlp_2 = Sequential(Linear(2*node_dim, hidden_dim), 
                                     ReLU(), 
                                     BatchNorm1d(hidden_dim),
                                     Linear(hidden_dim, node_dim),)  
        
    def forward(self, pos, helix):
        row, col = radius_graph(pos, r=2.5)
        mask = helix[row] != helix[col]
        row, col = row[mask], col[mask]
        u = torch.cat([pos[row], pos[col]], dim=1)
        u = self.node_mlp_1(u)
        u = scatter(u, col, dim=0, reduce='mean', dim_size=pos.size(0)) 
        u = torch.cat([pos, u], dim=1)
        u = self.node_mlp_2(u)
        return u

