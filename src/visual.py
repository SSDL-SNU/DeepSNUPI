"""
@author: Truong Quoc Chien (chientrq92@gmail.com)

"""
import torch
import plotly.graph_objects as go

#%% DNA structure visualization
def draw_DnaOrigami(G, coords, SE=None, EE=None):
    # Send dna graph to "cpu"
    G = G.to("cpu")
    n_nodes = G.x.size(0) 
    row, col = G.edge_index
    coords = coords.to("cpu")
    
    # init variables
    x_nodes, y_nodes, z_nodes = coords[:,0], coords[:,1], coords[:,2]     
    x_edges, y_edges, z_edges                = [], [], []
    x_edges_CO, y_edges_CO, z_edges_CO       = [], [], []
    x_edges_SS, y_edges_SS, z_edges_SS       = [], [], []
    x_edges_subs, y_edges_subs, z_edges_subs = [], [], []
    
    for i in range(G.edge_index.size(1)):
        if G.edge_attr[i,-1] == 1.0:
            x_coords = [coords[row[i],0], coords[col[i],0],None]
            x_edges += x_coords
            y_coords = [coords[row[i],1], coords[col[i],1], None]
            y_edges += y_coords
            z_coords = [coords[row[i],2], coords[col[i],2],None]
            z_edges += z_coords
            
        if G.edge_attr[i,-1] == 2.0:
            x_coords = [coords[row[i],0], coords[col[i],0],None]
            x_edges_CO += x_coords
            y_coords = [coords[row[i],1], coords[col[i],1], None]
            y_edges_CO += y_coords
            z_coords = [coords[row[i],2], coords[col[i],2],None]
            z_edges_CO += z_coords
            
        if G.edge_attr[i,-1] == 3.0:
            x_coords = [coords[row[i],0], coords[col[i],0],None]
            x_edges_SS += x_coords
            y_coords = [coords[row[i],1], coords[col[i],1], None]
            y_edges_SS += y_coords
            z_coords = [coords[row[i],2], coords[col[i],2],None]
            z_edges_SS += z_coords
            
        if G.x[row[i], -1] != G.x[col[i], -1]:
            x_coords = [coords[row[i],0], coords[col[i],0],None]
            x_edges_subs += x_coords
            y_coords = [coords[row[i],1], coords[col[i],1], None]
            y_edges_subs += y_coords
            z_coords = [coords[row[i],2], coords[col[i],2],None]
            z_edges_subs += z_coords
            
    data = []
    
    # Plot different edge types  
    trace_edges = go.Scatter3d( x=x_edges,
                                y=y_edges,
                                z=z_edges,
                                mode = 'lines',
                                line = dict(color='blue', width=4.),
                                hoverinfo = 'none')
    data.append(trace_edges)
    
    trace_CO    = go.Scatter3d( x=x_edges_CO,
                                y=y_edges_CO,
                                z=z_edges_CO,
                                mode = 'lines',
                                line = dict(color='orange', width=2),
                                hoverinfo = 'none')
    data.append(trace_CO)

    trace_SS    = go.Scatter3d( x=x_edges_SS,
                                y=y_edges_SS,
                                z=z_edges_SS,
                                mode = 'lines',
                                line = dict(color='red', width=2),
                                hoverinfo = 'none')
    data.append(trace_SS)
    
    # Create a trace for the nodes
    node_labels = torch.arange(0, n_nodes-1)
    trace_nodes = go.Scatter3d( x = x_nodes,
                                y = y_nodes,
                                z = z_nodes,
                                mode = 'markers',
                                marker = dict(  symbol='circle',
                                                size = 4.0,
                                                color = 'blue', 
                                                colorscale = ['lightgreen','magenta'], 
                                                line = dict(color='black', width=0.25)),
                                text = node_labels,
                                hoverinfo='text')
    data.append(trace_nodes)

    # We need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    
    # Also need to create the layout for our plot
    layout = go.Layout( showlegend=False,
                        scene=dict(xaxis=dict(axis),
                                yaxis=dict(axis),
                                zaxis=dict(axis),),
                        hovermode='closest')
    
    fig = go.Figure(data=data, layout=layout)
    X_range = torch.abs(coords[:, 0].max() - coords[:, 0].min()).item()/50
    Y_range = torch.abs(coords[:, 1].max() - coords[:, 1].min()).item()/50
    Z_range = torch.abs(coords[:, 2].max() - coords[:, 2].min()).item()/50
    if SE is not None and EE is not None and (SE==SE) and (EE==EE):
        fig.update_layout(title="Approximate Energy (Mech./Elec.): %.1e/%.1e[pNnm]" %(SE, EE), title_x=0.)
        
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=X_range, y=Y_range, z=Z_range))
    return fig