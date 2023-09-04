"""
@author: Truong Quoc Chien (chientrq92@gmail.com)

"""
import time
import torch
import tqdm, copy
from tqdm import tqdm
from src.model import Nets
from src.config import PARAMS
from src.utils import total_PE, mech_PE, RMSD

#%% Make trial prediction from ensemble model
def get_model_prediction(g, user_input=True, file_name=None, device="cuda"):
    g = g.to(device)
    # Load pre-trained models
    model_params = {k: v for k,v in PARAMS.items() if k.startswith("model_")}
    model = Nets(model_params)
    model_list_processed = ["small", "bend", "twist", "hinge", "wf2D"] 
    model_list_wf3D_processed = ["wf3D"]
    model_list_full = ["pre_train", "small", "bend", "twist", "hinge", "wf2D", "wf3D"] 
    
    # Start to predict
    y_gnn = torch.zeros(g.num_nodes, 6, device=device)
    SE_gnn = 1e10
    if user_input==True:
        model_list = model_list_full
    else:
        if g.x[:, -1].amax() > 0: 
            model_list = model_list_wf3D_processed
        else:
             model_list = model_list_processed
    start = time.time()
    for model_name in model_list:
        model.load_state_dict(torch.load("pretrained_models/" + model_name + ".pth.tar", map_location='cpu'))
        model.to(device)
        with torch.no_grad():
            y_pred = model(g)
            SE = mech_PE(y_pred, g.edge_index[:,::2], g.edge_attr[::2,:]).item()
            if (SE == SE) and (SE < SE_gnn):
                y_gnn = y_pred
                SE_gnn = SE
                best_model_name = model_name
                rmsd = RMSD(y_gnn[:,0:3], g.y[:,0:3]) if g.y is not None else None
    pred_time = time.time() - start  
    if file_name is not None:
        trial_g = copy.deepcopy(g)  
        trial_g.y = y_gnn.detach().clone()   
        torch.save(trial_g, "output/" + file_name + "_dgnn.pt") 
    return y_gnn, rmsd, best_model_name, pred_time

#%% Make refinement for trial prediction
def self_refinement(g, model_name, lr=5e-5, num_steps=200, 
                    epsilon=0.01, file_name=None, device=torch.device('cuda')):
    print("Self-refinement processing")
    
    # Load pre-trained model
    model_params = {k: v for k,v in PARAMS.items() if k.startswith("model_")}
    model = Nets(model_params)
    model.load_state_dict(torch.load("pretrained_models/" + model_name + ".pth.tar", map_location='cpu'))
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Iniital setup
    g = g.to(device)  
    with torch.no_grad():
        y_gnn = model(g)
        PE_gnn, _, _ = total_PE(y_gnn, g)
    y_refine = y_gnn
    PE_refine = PE_gnn.item()
    early_stopping_counter = 0   
    
    # Self-refinement loop
    start_refine = time.time()
    with tqdm(total=num_steps) as pbar:
        for epoch in range(num_steps):
            if (early_stopping_counter < 10):
                model.train()
                g.x.requires_grad_(True)
                optimizer.zero_grad() 
                y_pred = model(g)
                loss, _, _ = total_PE(y_pred, g, coeff_ele=1)
                loss.backward()
                optimizer.step()
                if ((PE_refine - loss.item())/PE_refine > epsilon):
                    early_stopping_counter = 0
                    y_refine = y_pred.detach().cpu()
                    PE_refine = loss.item()
                elif (epoch>10):
                    early_stopping_counter += 1   
            pbar.set_description("Potential energy: %.2e [pNnm]" %(loss.item()))
            pbar.update(1)
            
    refine_time = time.time() - start_refine
    rmsd = RMSD(y_refine[:,0:3], g.y[:,0:3]) if g.y is not None else None
    if file_name is not None:
        refine_g = copy.deepcopy(g)  
        refine_g.y = y_refine.detach().clone()   
        torch.save(refine_g, "output/" + file_name + "_refine.pt") 
    return y_refine, rmsd, refine_time
