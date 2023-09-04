"""
@author: Truong Quoc Chien (chientrq92@gmail.com)

"""
import roma
import torch
import mat73
import scipy.io as scio
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

#%% Estimate total potential energy
def total_PE(coords, g, coeff_ele=1, substr_edges=True, interfere_area=True):
    coords = coords.to(device=g.x.device)
    helix_idx  = g.x[:, 6].view(-1)
    try:
        substr_idx = g.x[:, 7].view(-1)
    except:
        substr_idx = None
    if substr_edges==True:
        SE = mech_PE(coords, g.edge_index[:,::2], g.edge_attr[::2,:])
    else:
        SE = mech_PE(coords, g.edge_index[:,::2], g.edge_attr[::2,:], substr_idx)
    if interfere_area==True:
        EE = elec_PE(coords, helix_idx)
    else:
        EE = elec_PE(coords, helix_idx, substr_idx)
    loss = SE + coeff_ele*EE
    return loss, SE.item(), EE.item()

#%% Estimate electrostatics potential energy
def elec_PE(x, helix_idx, substr_idx=None):
    pos = x[:,0:3]
    lamda_D = torch.tensor(1.2576, device=x.device)
    coeff   = torch.tensor(1.4131, device=x.device)
    row, col = radius_graph(pos, r=2.5) 
    mask = helix_idx[row] != helix_idx[col]
    if substr_idx is not None:
        interfere_mask = (substr_idx[row] == substr_idx[col])
        mask = torch.logical_and(mask, interfere_mask)
    row, col = row[mask], col[mask]
    dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
    dist = dist + 1.e-8
    PE_ele = coeff*torch.exp(-dist/lamda_D)/dist
    return PE_ele.sum()

#%% Estimate mechanical potential energy
def mech_PE(x, edge_index, edge_attr, substr_idx=None):
    n_edges = edge_index.size(1)
    row = edge_index[0,:]
    col = edge_index[1,:]
    prop = edge_attr[:,:30]
    coord_n1 = x[edge_index[0,:],:3]
    coord_n2 = x[edge_index[1,:],:3]
    Lo = torch.sqrt(torch.sum((coord_n1 - coord_n2)**2, dim=1)).unsqueeze(1)
    triad = roma.rotvec_to_rotmat(x[:,3:6])      
    tr_n1 = triad[edge_index[0,:],:,:]
    tr_n2 = triad[edge_index[1,:],:,:]
    co_index = edge_attr[:,-1] == 2
    tr_n1[co_index,:,:], tr_n2[co_index,:,:] = CO_triad_modify( tr_n1[co_index,:,:], 
                                                                tr_n2[co_index,:,:], 
                                                                coord_n1[co_index,:], 
                                                                coord_n2[co_index,:] )
    q1 = tr_n1[:,:,1].squeeze()
    q2 = tr_n2[:,:,1].squeeze()
    q = (q1 + q2)/2 
    r1 = ((coord_n2 - coord_n1)/Lo)
    r30 = torch.cross(r1,q)
    r3 = r30/torch.norm(r30,dim=1).view(r30.size(0),1).clamp(min = 1e-8)
    r2 = torch.cross(r3,r1)
    Rr = torch.cat((r1,r2,r3), dim=1).view(n_edges,3,3) 
    Rb_1 = torch.bmm(Rr,tr_n1)
    Rb_2 = torch.bmm(Rr,tr_n2)
    phi_1 = roma.rotmat_to_rotvec(Rb_1) 
    phi_2 = roma.rotmat_to_rotvec(Rb_2) 
    SE = torch.zeros((prop.size(dim=0),21), device=x.device)
    SE = element_se(prop, Lo, phi_1, phi_2)
    ss_index = edge_attr[:,-1] == 3
    SE[ss_index, 1:] = 0.0
    if substr_idx is not None:
        edge_mask = substr_idx[row] != substr_idx[col]
        SE[edge_mask,:] = 0.0 
    return SE.sum()

#%% Estimate beam energy based on co-rotational beam formula
def element_se(prop, Lo, phi_1, phi_2):
    t1x = phi_1[:,0].unsqueeze(1)
    t1y = phi_1[:,1].unsqueeze(1)
    t1z = phi_1[:,2].unsqueeze(1)
    t2x = phi_2[:,0].unsqueeze(1)
    t2y = phi_2[:,1].unsqueeze(1)
    t2z = phi_2[:,2].unsqueeze(1)
    Lg = prop[:,0].unsqueeze(1)
    ub = torch.zeros(Lg.size(), device=Lg.device)
    t1x_int = prop[:, 3].unsqueeze(1)
    t2x_int = prop[:, 4].unsqueeze(1)
    t1y_int = prop[:, 5].unsqueeze(1)
    t2y_int = prop[:, 6].unsqueeze(1)
    t1z_int = prop[:, 7].unsqueeze(1)
    t2z_int = prop[:, 8].unsqueeze(1)
    gj      = prop[:, 9].unsqueeze(1)
    ei33    = prop[:,10].unsqueeze(1) 
    ei22    = prop[:,11].unsqueeze(1)  
    ea      = prop[:,12].unsqueeze(1) 
    ga2     = prop[:,13].unsqueeze(1) 
    ga3     = prop[:,14].unsqueeze(1) 
    g_Tx_Ty = prop[:,15].unsqueeze(1) 
    g_Tx_Tz = prop[:,16].unsqueeze(1) 
    g_Ty_Tz = prop[:,17].unsqueeze(1) 
    g_Dx_Dy = prop[:,18].unsqueeze(1) 
    g_Dx_Dz = prop[:,19].unsqueeze(1) 
    g_Dy_Dz = prop[:,20].unsqueeze(1) 
    g_Tx_Dx = prop[:,21].unsqueeze(1) 
    g_Ty_Dx = prop[:,22].unsqueeze(1) 
    g_Tz_Dx = prop[:,23].unsqueeze(1) 
    g_Tx_Dy = prop[:,24].unsqueeze(1) 
    g_Ty_Dy = prop[:,25].unsqueeze(1) 
    g_Tz_Dy = prop[:,26].unsqueeze(1) 
    g_Tx_Dz = prop[:,27].unsqueeze(1) 
    g_Ty_Dz = prop[:,28].unsqueeze(1) 
    g_Tz_Dz = prop[:,29].unsqueeze(1) 
    se = torch.zeros((prop.size(dim=0),21), device=prop.device)
    se[:,0] = ((ea[:,0]*Lo[:,0]**2)/(2*Lg[:,0]) + (ea[:,0]*Lo[:,0]*ub[:,0])/Lg[:,0] - 
                ea[:,0]*Lo[:,0] + (ea[:,0]*ub[:,0]**2)/(2*Lg[:,0]) - ea[:,0]*ub[:,0] + 
               (ea[:,0]*Lg[:,0])/2)
    se[:,1] = ((ei22[:,0]*t1z[:,0]**2)/(2*Lg[:,0]) - (ei22[:,0]*t1z[:,0]*t2z[:,0])/Lg[:,0] - 
               (ei22[:,0]*t1z[:,0]*t1z_int[:,0])/Lg[:,0] + (ei22[:,0]*t1z[:,0]*t2z_int[:,0])/Lg[:,0] + 
               (ei22[:,0]*t2z[:,0]**2)/(2*Lg[:,0]) + (ei22[:,0]*t2z[:,0]*t1z_int[:,0])/Lg[:,0] - 
               (ei22[:,0]*t2z[:,0]*t2z_int[:,0])/Lg[:,0] + (ei22[:,0]*t1z_int[:,0]**2)/(2*Lg[:,0]) - 
               (ei22[:,0]*t1z_int[:,0]*t2z_int[:,0])/Lg[:,0] + (ei22[:,0]*t2z_int[:,0]**2)/(2*Lg[:,0]))       
    se[:,2] = ((ei33[:,0]*t1y[:,0]**2)/(2*Lg[:,0]) - (ei33[:,0]*t1y[:,0]*t2y[:,0])/Lg[:,0] - 
               (ei33[:,0]*t1y[:,0]*t1y_int[:,0])/Lg[:,0] + (ei33[:,0]*t1y[:,0]*t2y_int[:,0])/Lg[:,0] + 
               (ei33[:,0]*t2y[:,0]**2)/(2*Lg[:,0]) + (ei33[:,0]*t2y[:,0]*t1y_int[:,0])/Lg[:,0] - 
               (ei33[:,0]*t2y[:,0]*t2y_int[:,0])/Lg[:,0] + (ei33[:,0]*t1y_int[:,0]**2)/(2*Lg[:,0]) - 
               (ei33[:,0]*t1y_int[:,0]*t2y_int[:,0])/Lg[:,0] + (ei33[:,0]*t2y_int[:,0]**2)/(2*Lg[:,0]))    
    se[:,3] = ((gj[:,0]*t1x[:,0]**2)/(2*Lg[:,0]) - (gj[:,0]*t1x[:,0]*t2x[:,0])/Lg[:,0] - 
               (gj[:,0]*t1x[:,0]*t1x_int[:,0])/Lg[:,0] + (gj[:,0]*t1x[:,0]*t2x_int[:,0])/Lg[:,0] + 
               (gj[:,0]*t2x[:,0]**2)/(2*Lg[:,0]) + (gj[:,0]*t2x[:,0]*t1x_int[:,0])/Lg[:,0] - 
               (gj[:,0]*t2x[:,0]*t2x_int[:,0])/Lg[:,0] + (gj[:,0]*t1x_int[:,0]**2)/(2*Lg[:,0]) - 
               (gj[:,0]*t1x_int[:,0]*t2x_int[:,0])/Lg[:,0] + (gj[:,0]*t2x_int[:,0]**2)/(2*Lg[:,0]))    
    se[:,4] = ((23*ga2[:,0]*Lg[:,0]*t1z[:,0]**2)/120 + (13*ga2[:,0]*Lg[:,0]*t1z[:,0]*t2z[:,0])/60 - 
               (23*ga2[:,0]*Lg[:,0]*t1z[:,0]*t1z_int[:,0])/60 - (13*ga2[:,0]*Lg[:,0]*t1z[:,0]*t2z_int[:,0])/60 + 
               (23*ga2[:,0]*Lg[:,0]*t2z[:,0]**2)/120 - (13*ga2[:,0]*Lg[:,0]*t2z[:,0]*t1z_int[:,0])/60 - 
               (23*ga2[:,0]*Lg[:,0]*t2z[:,0]*t2z_int[:,0])/60 + (23*ga2[:,0]*Lg[:,0]*t1z_int[:,0]**2)/120 + 
               (13*ga2[:,0]*Lg[:,0]*t1z_int[:,0]*t2z_int[:,0])/60 + (23*ga2[:,0]*Lg[:,0]*t2z_int[:,0]**2)/120)   
    se[:,5] = ((23*ga3[:,0]*Lg[:,0]*t1y[:,0]**2)/120 + (13*ga3[:,0]*Lg[:,0]*t1y[:,0]*t2y[:,0])/60 - 
               (23*ga3[:,0]*Lg[:,0]*t1y[:,0]*t1y_int[:,0])/60 - (13*ga3[:,0]*Lg[:,0]*t1y[:,0]*t2y_int[:,0])/60 + 
               (23*ga3[:,0]*Lg[:,0]*t2y[:,0]**2)/120 - (13*ga3[:,0]*Lg[:,0]*t2y[:,0]*t1y_int[:,0])/60 - 
               (23*ga3[:,0]*Lg[:,0]*t2y[:,0]*t2y_int[:,0])/60 + (23*ga3[:,0]*Lg[:,0]*t1y_int[:,0]**2)/120 + 
               (13*ga3[:,0]*Lg[:,0]*t1y_int[:,0]*t2y_int[:,0])/60 + (23*ga3[:,0]*Lg[:,0]*t2y_int[:,0]**2)/120)      
    se[:, 6] = ((g_Tx_Ty[:,0]*t1x[:,0]*t1y[:,0])/Lg[:,0] - (g_Tx_Ty[:,0]*t1x[:,0]*t2y[:,0])/Lg[:,0] - 
                (g_Tx_Ty[:,0]*t2x[:,0]*t1y[:,0])/Lg[:,0] + (g_Tx_Ty[:,0]*t2x[:,0]*t2y[:,0])/Lg[:,0] - 
                (g_Tx_Ty[:,0]*t1x[:,0]*t1y_int[:,0])/Lg[:,0] - (g_Tx_Ty[:,0]*t1y[:,0]*t1x_int[:,0])/Lg[:,0] + 
                (g_Tx_Ty[:,0]*t1x[:,0]*t2y_int[:,0])/Lg[:,0] + (g_Tx_Ty[:,0]*t2x[:,0]*t1y_int[:,0])/Lg[:,0] + 
                (g_Tx_Ty[:,0]*t1y[:,0]*t2x_int[:,0])/Lg[:,0] + (g_Tx_Ty[:,0]*t2y[:,0]*t1x_int[:,0])/Lg[:,0] - 
                (g_Tx_Ty[:,0]*t2x[:,0]*t2y_int[:,0])/Lg[:,0] - (g_Tx_Ty[:,0]*t2y[:,0]*t2x_int[:,0])/Lg[:,0] + 
                (g_Tx_Ty[:,0]*t1x_int[:,0]*t1y_int[:,0])/Lg[:,0] - (g_Tx_Ty[:,0]*t1x_int[:,0]*t2y_int[:,0])/Lg[:,0] - 
                (g_Tx_Ty[:,0]*t2x_int[:,0]*t1y_int[:,0])/Lg[:,0] + (g_Tx_Ty[:,0]*t2x_int[:,0]*t2y_int[:,0])/Lg[:,0])
    se[:, 7] =  ((g_Tx_Tz[:,0]*t1x[:,0]*t1z[:,0])/Lg[:,0] - (g_Tx_Tz[:,0]*t1x[:,0]*t2z[:,0])/Lg[:,0] - 
                 (g_Tx_Tz[:,0]*t2x[:,0]*t1z[:,0])/Lg[:,0] + (g_Tx_Tz[:,0]*t2x[:,0]*t2z[:,0])/Lg[:,0] - 
                 (g_Tx_Tz[:,0]*t1x[:,0]*t1z_int[:,0])/Lg[:,0] - (g_Tx_Tz[:,0]*t1x_int[:,0]*t1z[:,0])/Lg[:,0] + 
                 (g_Tx_Tz[:,0]*t1x[:,0]*t2z_int[:,0])/Lg[:,0] + (g_Tx_Tz[:,0]*t2x[:,0]*t1z_int[:,0])/Lg[:,0] + 
                 (g_Tx_Tz[:,0]*t1x_int[:,0]*t2z[:,0])/Lg[:,0] + (g_Tx_Tz[:,0]*t2x_int[:,0]*t1z[:,0])/Lg[:,0] - 
                 (g_Tx_Tz[:,0]*t2x[:,0]*t2z_int[:,0])/Lg[:,0] - (g_Tx_Tz[:,0]*t2x_int[:,0]*t2z[:,0])/Lg[:,0] + 
                 (g_Tx_Tz[:,0]*t1x_int[:,0]*t1z_int[:,0])/Lg[:,0] - (g_Tx_Tz[:,0]*t1x_int[:,0]*t2z_int[:,0])/Lg[:,0] - 
                 (g_Tx_Tz[:,0]*t2x_int[:,0]*t1z_int[:,0])/Lg[:,0] + (g_Tx_Tz[:,0]*t2x_int[:,0]*t2z_int[:,0])/Lg[:,0])  
    se[:, 8] = ((g_Ty_Tz[:,0]*t1y[:,0]*t1z[:,0])/Lg[:,0] - (g_Ty_Tz[:,0]*t1y[:,0]*t2z[:,0])/Lg[:,0] - 
                (g_Ty_Tz[:,0]*t2y[:,0]*t1z[:,0])/Lg[:,0] + (g_Ty_Tz[:,0]*t2y[:,0]*t2z[:,0])/Lg[:,0] - 
                (g_Ty_Tz[:,0]*t1y[:,0]*t1z_int[:,0])/Lg[:,0] - (g_Ty_Tz[:,0]*t1z[:,0]*t1y_int[:,0])/Lg[:,0] + 
                (g_Ty_Tz[:,0]*t1y[:,0]*t2z_int[:,0])/Lg[:,0] + (g_Ty_Tz[:,0]*t2y[:,0]*t1z_int[:,0])/Lg[:,0] + 
                (g_Ty_Tz[:,0]*t1z[:,0]*t2y_int[:,0])/Lg[:,0] + (g_Ty_Tz[:,0]*t2z[:,0]*t1y_int[:,0])/Lg[:,0] - 
                (g_Ty_Tz[:,0]*t2y[:,0]*t2z_int[:,0])/Lg[:,0] - (g_Ty_Tz[:,0]*t2z[:,0]*t2y_int[:,0])/Lg[:,0] + 
                (g_Ty_Tz[:,0]*t1y_int[:,0]*t1z_int[:,0])/Lg[:,0] - (g_Ty_Tz[:,0]*t1y_int[:,0]*t2z_int[:,0])/Lg[:,0] - 
                (g_Ty_Tz[:,0]*t2y_int[:,0]*t1z_int[:,0])/Lg[:,0] + (g_Ty_Tz[:,0]*t2y_int[:,0]*t2z_int[:,0])/Lg[:,0])
    se[:, 9] =  ((Lg[:,0]*g_Dx_Dy[:,0]*t1y[:,0]*t2z[:,0])/144 - (Lg[:,0]*g_Dx_Dy[:,0]*t1y[:,0]*t1z[:,0])/144 + 
                 (Lg[:,0]*g_Dx_Dy[:,0]*t2y[:,0]*t1z[:,0])/144 - (Lg[:,0]*g_Dx_Dy[:,0]*t2y[:,0]*t2z[:,0])/144 + 
                 (Lg[:,0]*g_Dx_Dy[:,0]*t1y[:,0]*t1z_int[:,0])/144 + (Lg[:,0]*g_Dx_Dy[:,0]*t1z[:,0]*t1y_int[:,0])/144 - 
                 (Lg[:,0]*g_Dx_Dy[:,0]*t1y[:,0]*t2z_int[:,0])/144 - (Lg[:,0]*g_Dx_Dy[:,0]*t2y[:,0]*t1z_int[:,0])/144 - 
                 (Lg[:,0]*g_Dx_Dy[:,0]*t1z[:,0]*t2y_int[:,0])/144 - (Lg[:,0]*g_Dx_Dy[:,0]*t2z[:,0]*t1y_int[:,0])/144 + 
                 (Lg[:,0]*g_Dx_Dy[:,0]*t2y[:,0]*t2z_int[:,0])/144 + (Lg[:,0]*g_Dx_Dy[:,0]*t2z[:,0]*t2y_int[:,0])/144 - 
                 (Lg[:,0]*g_Dx_Dy[:,0]*t1y_int[:,0]*t1z_int[:,0])/144 + (Lg[:,0]*g_Dx_Dy[:,0]*t1y_int[:,0]*t2z_int[:,0])/144 + 
                 (Lg[:,0]*g_Dx_Dy[:,0]*t2y_int[:,0]*t1z_int[:,0])/144 - (Lg[:,0]*g_Dx_Dy[:,0]*t2y_int[:,0]*t2z_int[:,0])/144)
    se[:,10] =  ((Lg[:,0]*g_Dx_Dz[:,0]*t1y[:,0])/24 - (Lg[:,0]*g_Dx_Dz[:,0]*t2y[:,0])/24 - (Lg[:,0]*g_Dx_Dz[:,0]*t1y_int[:,0])/24 + 
                 (Lg[:,0]*g_Dx_Dz[:,0]*t2y_int[:,0])/24 - (Lo[:,0]*g_Dx_Dz[:,0]*t1y[:,0])/24 + (Lo[:,0]*g_Dx_Dz[:,0]*t2y[:,0])/24 + 
                 (Lo[:,0]*g_Dx_Dz[:,0]*t1y_int[:,0])/24 - (Lo[:,0]*g_Dx_Dz[:,0]*t2y_int[:,0])/24 - (g_Dx_Dz[:,0]*t1y[:,0]*ub[:,0])/24 + 
                 (g_Dx_Dz[:,0]*t2y[:,0]*ub[:,0])/24 + (g_Dx_Dz[:,0]*t1y_int[:,0]*ub[:,0])/24 - (g_Dx_Dz[:,0]*t2y_int[:,0]*ub[:,0])/24)
    se[:,11] = ((Lg[:,0]*g_Dy_Dz[:,0]*t2z[:,0])/24 - (Lg[:,0]*g_Dy_Dz[:,0]*t1z[:,0])/24 + (Lg[:,0]*g_Dy_Dz[:,0]*t1z_int[:,0])/24 - 
                (Lg[:,0]*g_Dy_Dz[:,0]*t2z_int[:,0])/24 + (Lo[:,0]*g_Dy_Dz[:,0]*t1z[:,0])/24 - (Lo[:,0]*g_Dy_Dz[:,0]*t2z[:,0])/24 - 
                (Lo[:,0]*g_Dy_Dz[:,0]*t1z_int[:,0])/24 + (Lo[:,0]*g_Dy_Dz[:,0]*t2z_int[:,0])/24 + (g_Dy_Dz[:,0]*t1z[:,0]*ub[:,0])/24 - 
                (g_Dy_Dz[:,0]*t2z[:,0]*ub[:,0])/24 - (g_Dy_Dz[:,0]*t1z_int[:,0]*ub[:,0])/24 + (g_Dy_Dz[:,0]*t2z_int[:,0]*ub[:,0])/24)
    se[:,12] = ((g_Tx_Dx[:,0]*t1x[:,0]*t1y[:,0])/12 - (g_Tx_Dx[:,0]*t1x[:,0]*t2y[:,0])/12 - (g_Tx_Dx[:,0]*t2x[:,0]*t1y[:,0])/12 + 
                (g_Tx_Dx[:,0]*t2x[:,0]*t2y[:,0])/12 - (g_Tx_Dx[:,0]*t1x[:,0]*t1y_int[:,0])/12 - (g_Tx_Dx[:,0]*t1y[:,0]*t1x_int[:,0])/12 + 
                (g_Tx_Dx[:,0]*t1x[:,0]*t2y_int[:,0])/12 + (g_Tx_Dx[:,0]*t2x[:,0]*t1y_int[:,0])/12 + (g_Tx_Dx[:,0]*t1y[:,0]*t2x_int[:,0])/12 + 
                (g_Tx_Dx[:,0]*t2y[:,0]*t1x_int[:,0])/12 - (g_Tx_Dx[:,0]*t2x[:,0]*t2y_int[:,0])/12 - (g_Tx_Dx[:,0]*t2y[:,0]*t2x_int[:,0])/12 + 
                (g_Tx_Dx[:,0]*t1x_int[:,0]*t1y_int[:,0])/12 - (g_Tx_Dx[:,0]*t1x_int[:,0]*t2y_int[:,0])/12 - (g_Tx_Dx[:,0]*t2x_int[:,0]*t1y_int[:,0])/12 + 
                (g_Tx_Dx[:,0]*t2x_int[:,0]*t2y_int[:,0])/12)
    se[:,13] =  ((g_Ty_Dx[:,0]*t1y[:,0]**2)/12 - (g_Ty_Dx[:,0]*t1y[:,0]*t2y[:,0])/6 - (g_Ty_Dx[:,0]*t1y[:,0]*t1y_int[:,0])/6 + 
                 (g_Ty_Dx[:,0]*t1y[:,0]*t2y_int[:,0])/6 + (g_Ty_Dx[:,0]*t2y[:,0]**2)/12 + (g_Ty_Dx[:,0]*t2y[:,0]*t1y_int[:,0])/6 - 
                 (g_Ty_Dx[:,0]*t2y[:,0]*t2y_int[:,0])/6 + (g_Ty_Dx[:,0]*t1y_int[:,0]**2)/12 - (g_Ty_Dx[:,0]*t1y_int[:,0]*t2y_int[:,0])/6 + 
                 (g_Ty_Dx[:,0]*t2y_int[:,0]**2)/12)  
    se[:,14] = ((g_Tz_Dx[:,0]*t1y[:,0]*t1z[:,0])/12 - (g_Tz_Dx[:,0]*t1y[:,0]*t2z[:,0])/12 - (g_Tz_Dx[:,0]*t2y[:,0]*t1z[:,0])/12 + 
                (g_Tz_Dx[:,0]*t2y[:,0]*t2z[:,0])/12 - (g_Tz_Dx[:,0]*t1y[:,0]*t1z_int[:,0])/12 - (g_Tz_Dx[:,0]*t1z[:,0]*t1y_int[:,0])/12 + 
                (g_Tz_Dx[:,0]*t1y[:,0]*t2z_int[:,0])/12 + (g_Tz_Dx[:,0]*t2y[:,0]*t1z_int[:,0])/12 + (g_Tz_Dx[:,0]*t1z[:,0]*t2y_int[:,0])/12 + 
                (g_Tz_Dx[:,0]*t2z[:,0]*t1y_int[:,0])/12 - (g_Tz_Dx[:,0]*t2y[:,0]*t2z_int[:,0])/12 - (g_Tz_Dx[:,0]*t2z[:,0]*t2y_int[:,0])/12 + 
                (g_Tz_Dx[:,0]*t1y_int[:,0]*t1z_int[:,0])/12 - (g_Tz_Dx[:,0]*t1y_int[:,0]*t2z_int[:,0])/12 - (g_Tz_Dx[:,0]*t2y_int[:,0]*t1z_int[:,0])/12 + 
                (g_Tz_Dx[:,0]*t2y_int[:,0]*t2z_int[:,0])/12)
    se[:,15] = ((g_Tx_Dy[:,0]*t1x[:,0]*t2z[:,0])/12 - (g_Tx_Dy[:,0]*t1x[:,0]*t1z[:,0])/12 + (g_Tx_Dy[:,0]*t2x[:,0]*t1z[:,0])/12 - 
                (g_Tx_Dy[:,0]*t2x[:,0]*t2z[:,0])/12 + (g_Tx_Dy[:,0]*t1x[:,0]*t1z_int[:,0])/12 + (g_Tx_Dy[:,0]*t1x_int[:,0]*t1z[:,0])/12 - 
                (g_Tx_Dy[:,0]*t1x[:,0]*t2z_int[:,0])/12 - (g_Tx_Dy[:,0]*t2x[:,0]*t1z_int[:,0])/12 - (g_Tx_Dy[:,0]*t1x_int[:,0]*t2z[:,0])/12 - 
                (g_Tx_Dy[:,0]*t2x_int[:,0]*t1z[:,0])/12 + (g_Tx_Dy[:,0]*t2x[:,0]*t2z_int[:,0])/12 + (g_Tx_Dy[:,0]*t2x_int[:,0]*t2z[:,0])/12 - 
                (g_Tx_Dy[:,0]*t1x_int[:,0]*t1z_int[:,0])/12 + (g_Tx_Dy[:,0]*t1x_int[:,0]*t2z_int[:,0])/12 + (g_Tx_Dy[:,0]*t2x_int[:,0]*t1z_int[:,0])/12 - 
                (g_Tx_Dy[:,0]*t2x_int[:,0]*t2z_int[:,0])/12) 
    se[:,16] = ((g_Ty_Dy[:,0]*t1y[:,0]*t2z[:,0])/12 - (g_Ty_Dy[:,0]*t1y[:,0]*t1z[:,0])/12 + (g_Ty_Dy[:,0]*t2y[:,0]*t1z[:,0])/12 - 
                (g_Ty_Dy[:,0]*t2y[:,0]*t2z[:,0])/12 + (g_Ty_Dy[:,0]*t1y[:,0]*t1z_int[:,0])/12 + (g_Ty_Dy[:,0]*t1z[:,0]*t1y_int[:,0])/12 - 
                (g_Ty_Dy[:,0]*t1y[:,0]*t2z_int[:,0])/12 - (g_Ty_Dy[:,0]*t2y[:,0]*t1z_int[:,0])/12 - (g_Ty_Dy[:,0]*t1z[:,0]*t2y_int[:,0])/12 - 
                (g_Ty_Dy[:,0]*t2z[:,0]*t1y_int[:,0])/12 + (g_Ty_Dy[:,0]*t2y[:,0]*t2z_int[:,0])/12 + (g_Ty_Dy[:,0]*t2z[:,0]*t2y_int[:,0])/12 - 
                (g_Ty_Dy[:,0]*t1y_int[:,0]*t1z_int[:,0])/12 + (g_Ty_Dy[:,0]*t1y_int[:,0]*t2z_int[:,0])/12 + (g_Ty_Dy[:,0]*t2y_int[:,0]*t1z_int[:,0])/12 - 
                (g_Ty_Dy[:,0]*t2y_int[:,0]*t2z_int[:,0])/12)  
    se[:,17] = ((g_Tz_Dy[:,0]*t1z[:,0]*t2z[:,0])/6 - (g_Tz_Dy[:,0]*t2z[:,0]**2)/12 - (g_Tz_Dy[:,0]*t1z_int[:,0]**2)/12 - 
                (g_Tz_Dy[:,0]*t2z_int[:,0]**2)/12 - (g_Tz_Dy[:,0]*t1z[:,0]**2)/12 + (g_Tz_Dy[:,0]*t1z[:,0]*t1z_int[:,0])/6 - 
                (g_Tz_Dy[:,0]*t1z[:,0]*t2z_int[:,0])/6 - (g_Tz_Dy[:,0]*t2z[:,0]*t1z_int[:,0])/6 + (g_Tz_Dy[:,0]*t2z[:,0]*t2z_int[:,0])/6 + 
                (g_Tz_Dy[:,0]*t1z_int[:,0]*t2z_int[:,0])/6)
    se[:,18] = ((g_Tx_Dz[:,0]*t1x[:,0])/2 - (g_Tx_Dz[:,0]*t2x[:,0])/2 - (g_Tx_Dz[:,0]*t1x_int[:,0])/2 + 
                (g_Tx_Dz[:,0]*t2x_int[:,0])/2 - (Lo[:,0]*g_Tx_Dz[:,0]*t1x[:,0])/(2*Lg[:,0]) + 
                (Lo[:,0]*g_Tx_Dz[:,0]*t2x[:,0])/(2*Lg[:,0]) + (Lo[:,0]*g_Tx_Dz[:,0]*t1x_int[:,0])/(2*Lg[:,0]) - 
                (Lo[:,0]*g_Tx_Dz[:,0]*t2x_int[:,0])/(2*Lg[:,0]) - (g_Tx_Dz[:,0]*t1x[:,0]*ub[:,0])/(2*Lg[:,0]) + 
                (g_Tx_Dz[:,0]*t2x[:,0]*ub[:,0])/(2*Lg[:,0]) + (g_Tx_Dz[:,0]*t1x_int[:,0]*ub[:,0])/(2*Lg[:,0]) - 
                (g_Tx_Dz[:,0]*t2x_int[:,0]*ub[:,0])/(2*Lg[:,0]))
    se[:,19] = ((g_Ty_Dz[:,0]*t1y[:,0])/2 - (g_Ty_Dz[:,0]*t2y[:,0])/2 - (g_Ty_Dz[:,0]*t1y_int[:,0])/2 + 
                (g_Ty_Dz[:,0]*t2y_int[:,0])/2 - (Lo[:,0]*g_Ty_Dz[:,0]*t1y[:,0])/(2*Lg[:,0]) + 
                (Lo[:,0]*g_Ty_Dz[:,0]*t2y[:,0])/(2*Lg[:,0]) + (Lo[:,0]*g_Ty_Dz[:,0]*t1y_int[:,0])/(2*Lg[:,0]) - 
                (Lo[:,0]*g_Ty_Dz[:,0]*t2y_int[:,0])/(2*Lg[:,0]) - (g_Ty_Dz[:,0]*t1y[:,0]*ub[:,0])/(2*Lg[:,0]) + 
                (g_Ty_Dz[:,0]*t2y[:,0]*ub[:,0])/(2*Lg[:,0]) + (g_Ty_Dz[:,0]*t1y_int[:,0]*ub[:,0])/(2*Lg[:,0]) - 
                (g_Ty_Dz[:,0]*t2y_int[:,0]*ub[:,0])/(2*Lg[:,0]))   
    se[:,20] =  ((g_Tz_Dz[:,0]*t1z[:,0])/2 - (g_Tz_Dz[:,0]*t2z[:,0])/2 - (g_Tz_Dz[:,0]*t1z_int[:,0])/2 + 
                (g_Tz_Dz[:,0]*t2z_int[:,0])/2 - (Lo[:,0]*g_Tz_Dz[:,0]*t1z[:,0])/(2*Lg[:,0]) + 
                (Lo[:,0]*g_Tz_Dz[:,0]*t2z[:,0])/(2*Lg[:,0]) + (Lo[:,0]*g_Tz_Dz[:,0]*t1z_int[:,0])/(2*Lg[:,0]) - 
                (Lo[:,0]*g_Tz_Dz[:,0]*t2z_int[:,0])/(2*Lg[:,0]) - (g_Tz_Dz[:,0]*t1z[:,0]*ub[:,0])/(2*Lg[:,0]) + 
                (g_Tz_Dz[:,0]*t2z[:,0]*ub[:,0])/(2*Lg[:,0]) + (g_Tz_Dz[:,0]*t1z_int[:,0]*ub[:,0])/(2*Lg[:,0]) - 
                (g_Tz_Dz[:,0]*t2z_int[:,0]*ub[:,0])/(2*Lg[:,0]))  
    return se

#%% Modify triad orientation on Crossover steps
def CO_triad_modify(tr_n1, tr_n2, coord_n1, coord_n2):
    n_co = tr_n1.size(0)
    co_vec = coord_n2 - coord_n1 
    i = torch.arange(n_co, device=tr_n1.device).reshape(n_co,1,1)
    j = torch.arange(3, device=tr_n1.device).reshape(1,3,1)
    k0 = torch.tensor([[[0]]]).to(tr_n1.device)
    k1 = torch.tensor([[[1]]]).to(tr_n1.device)
    k2 = torch.tensor([[[2]]]).to(tr_n1.device)
    y_n1 = tr_n1[i,j,k2].squeeze().view(n_co,3)         
    z_n1_trial = tr_n1[i,j,k0].squeeze().view(n_co,3)
    sign_n1 = torch.sign((co_vec*torch.cross(y_n1, z_n1_trial)).sum(dim=1)).view(n_co,1)
    z_n1 = (sign_n1.expand_as(z_n1_trial)*z_n1_trial)
    x_n1 = torch.cross(y_n1, z_n1)
    tr_n1_new = torch.cat([x_n1.unsqueeze(2), y_n1.unsqueeze(2), z_n1.unsqueeze(2)],dim=2)
    y_n2 = tr_n2[i,j,k2].squeeze().view(n_co,3)       
    z_n2_trial = tr_n2[i,j,k0].squeeze().view(n_co,3) 
    sign_n2 = torch.sign((co_vec*torch.cross(y_n2, z_n2_trial)).sum(dim=1)).view(n_co,1)
    z_n2 = (sign_n2.expand_as(z_n2_trial)*z_n2_trial)
    x_n2 = torch.cross(y_n2, z_n2)
    tr_n2_new = torch.cat([x_n2.unsqueeze(2), y_n2.unsqueeze(2), z_n2.unsqueeze(2)],dim=2)
    return tr_n1_new, tr_n2_new

#%% Estimate RMSD
def RMSD(src, target):
    src = src.to(target.device)
    Ro, tr = roma.rigid_points_registration(src[:, 0:3], target[:, 0:3])
    src_align = src[:, 0:3] @ Ro.T + tr
    rmsd = torch.sqrt((src_align - target[:, 0:3]).pow(2).sum()/src.size(0))
    return rmsd

#%% load DNA origami graph from .mat or .pt file
def get_input(file_name):
    try:
        try:
            data = scio.loadmat(file_name)
        except:
            data = scio.loadmat("user_input/"+ file_name)
    except:
        try:
            data = mat73.loadmat(file_name)
        except:
            data = mat73.loadmat(file_name)
    conn   = data["elem_conn"]
    icoord = data["init_coord"]
    prop   = data["prop"]
    # Convert to graph data
    x = torch.tensor(icoord, dtype=torch.float, device="cpu")
    edge_index = torch.tensor(conn, dtype=torch.long, device="cpu")
    edge_attr = torch.tensor(prop, dtype=torch.float, device="cpu")
    try:
        fcoord = data["finl_coord"]
        y = torch.tensor(fcoord, dtype=torch.float, device="cpu")
        dna = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, device="cpu") 
    except:
        dna = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, device="cpu") 
    return dna
