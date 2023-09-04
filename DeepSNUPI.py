"""
@author: Truong Quoc Chien (chientrq92@gmail.com)

"""
import torch
from src import *
from PIL import Image
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "GPU" if torch.cuda.is_available() else "CPU"
image = Image.open('cover.PNG')

# APP TITLE
st.set_page_config(layout="wide")
app_title = '<p style="font-size: 40px;">Deep SNUPI</p>'
st.markdown(app_title, unsafe_allow_html=True)
st.sidebar.subheader("PICK TOOLS")
select = st.sidebar.selectbox("DROP-DOWN",["ABOUT", "GNN predict"], key='1')
input = None
clicked_predict = None
clicked_refine = None

# OPTION 1 ABOUT
if select == "ABOUT":
    st.image(image)
    st.text("Computational analysis of nucleic acids structures based on Graph Neural Networks")
    st.text("Simulation-Driven Structure Design Laboratory, Seoul National University")
    st.write('\n')
    st.markdown('<p style="font-size: 32px;">Dna Origami Graph Neural Networks</p>', unsafe_allow_html=True)
    st.text("Deep-SNUPI is a graph neural networks model to predict the three-dimensional shape of DNA origami assemblies.") 
    st.text("It was trained by hybrid data-driven and physics-informed approach.")
    
#OPTION 2 ENSEMBLE MODEL PREDICT
elif select == "GNN predict": 
    # File upload
    col1, col2 = st.columns(2)
    col1.write("###")
    col1.markdown("Please upload input file or select from `dataset` folder")
    input_type = col1.selectbox('Input options',(None, "Select samples", "Upload data")) 
    
    dna = None
    
    # select samples from dataset             
    if input_type == "Select samples":
        user_input = False
        col1.markdown("👈 Click the arrow on left to select processed input data")
        dataset = st.sidebar.selectbox('Dataset',("Training set", "Test set",)) 
        
        # Training set list
        if dataset == "Training set":
            select_processed_file = st.sidebar.selectbox('Training set',
                                    (None,
                                    "DX_Honeycomb_42bp",
                                    "DX_Cross_63bp",
                                    "DX_Arrowhead_63bp",
                                    "DX_Annulus_42bp",
                                    "DX_Sided_Polygon_126bp",
                                    "2006_Rothemund_square",
                                    "2009_Nature_railedbridge",
                                    "2009_Science_Spiral_JY",
                                    "2012_NAR_32hb_21xx",
                                    "2012_NAR_32hb_42xx",
                                    "2012_NAR_32hb_63xx",
                                    "2012_NAR_32hb_84xx",
                                    "2012_NAR_A",
                                    "2012_NAR_S",
                                    "2017_NatComm_10_L3R1_2hb_168ds",
                                    "2019_ACS_Nano_Spring_01_6HB",
                                    "2019_ACS_Nano_Spring_02_12HB",
                                    "2019_ACS_Nano_Spring_03_24HB",
                                    "2019_ACSnano_C_08_A01_6hb_twisted_1IH0L6_nBT_14B",
                                    "2019_NAR_8_2INS_FLEX_2",
                                    "2020_AngChem_8SQ_All_02T",
                                    "2020_NatComm_TwistTower_TWCR_mod_delBPHJ",
                                    "2021_ACSnano_Sq_10HB_Hollow_13_1id_3gap_twcorr",
                                    "2021_ACSnano_Sq_12HB_0id_0gap_twcorr_scaffconn",
                                    "Gyroelon_Penta_Pyramid_6HB_42bp",
                                    "Triang_Bipyramid_6HB_126bp",
                                    "Penta_Bipyramid_6HB_42bp",
                                    "Square_Gyrobicupola_6HB_42bp",
                                    )) 
        # Test set list    
        elif dataset == "Test set":
            select_processed_file = st.sidebar.selectbox('Test set',
                                    (None,
                                    "2009_JACS_8Layer",
                                    "2009_NAR_6x10",
                                    "2009_Nature_genie_bottle",
                                    "2009_Science_bent_60",
                                    "2009_Science_gear_90",
                                    "2017_Nature_Triangle",
                                    "2017_Nature_V_22_55",
                                    "2017_Nature_V_Emboss_01",
                                    "2017_NatComm_L3_420ds",
                                    "2017_NatComm_M1_2hb_252ds",
                                    "2017_NatComm_M1M3_2hb_252ds",
                                    "2017_NatComm_M1M3_2hb_168ds",
                                    "2017_NatComm_M2R1_2hb_12nt",
                                    "2017_NatComm_L3R1_2hb_triangle",
                                    "2017_NatComm_M2_315ds_75deg",
                                    "2017_NatComm_M2_441ds_120deg",
                                    "2018_NatComm_Curved_Q",
                                    "2019_NAR_2_Flexible",
                                    "2019_ACSNano_12SQ",
                                    "2019_ACSNano_Ins_H0L6_02",
                                    "2019_ACSNano_Ins_H0L6_18",
                                    "2019_ACSNano_Ins_H6L0_04",
                                    "2019_ACSNano_Ins_H3L3_06",
                                    "2019_ACSNano_Tetrahedron_84bp",
                                    "2019_ACSNano_Cube_84bp",
                                    "2019_ACSNano_Cubeocta_84bp",
                                    "2019_ACSNano_Trunc_Tetra_63bp",
                                    "2019_ACSNano_Triang_Bipyramid_63bp",
                                    "2019_ACSNano_Penta_Bipyramid_105bp",
                                    "2019_ACSNano_Rhom_Dodeca_63bp",
                                    "2019_ACSNano_Tria_Tetra_84bp",
                                    "2019_ACSNano_Twisted_Tri_Prism_42bp",
                                    "2019_SciAdv_04_Wheel_DX_73bp",
                                    "2019_SciAdv_06_Rhombic_Tiling_DX_42bp",
                                    "2019_SciAdv_13_Hexagonal_Tiling_DX_63bp",
                                    "2019_SciAdv_14_Prismatic_Penta_Tiling_DX_52bp",
                                    "2019_SciAdv_16_4_Sided_Polygon_DX_73bp",
                                    "2019_SciAdv_16_4_Sided_Polygon_DX_105bp",
                                    "2019_SciAdv_17_5_Sided_Polygon_DX_73bp",
                                    "2019_SciAdv_18_6_Sided_Polygon_DX_115bp",
                                    "2019_SciAdv_19_L_Shape_42bp_DX_52bp",
                                    "2020_NatComm_Pointer_v2",
                                    "2020_NatComm_Dumbell_v2",
                                    "2020_NatComm_HB_v3",
                                    "2020_NatComm_6HBv3",
                                    "2020_AngChem_8SQ_08T",
                                    "2021_ACSNano_Sq_12HB_2gap",
                                    )) 
        
        if select_processed_file is not None:
            file_name = select_processed_file
            try:
                dna = get_input("./dataset/origami/test/" + select_processed_file)
            except:
                dna = get_input("./dataset/origami/train/" + select_processed_file)
                
    # select upload data from users            
    elif input_type == "Upload data":
        file_upload = col1.expander(label="Upload a design file")
        uploaded_file = file_upload.file_uploader("(Please upload your own input file from SNUPI)")
        col1.markdown('***')
        # Save it as temp file
        dna = None
        user_input = True
        file_name = None
        if uploaded_file:
            user_input = True
            try:
                temp_filename = "./dataset/user_input/temp.mat"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())   
                dna = get_input(temp_filename)
            except: 
                temp_filename = "./dataset/user_input/temp.pt"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())   
                dna = torch.load(temp_filename).to("cpu")  
            file_name = uploaded_file.name
    num_refine_steps = st.sidebar.number_input('Number of self-refinement steps', min_value=200, max_value=1000, step=100)
    
    if dna is not None:
        
        # Initial evaluation structural energies
        PE_init, SE_init, EE_init = total_PE(dna.x, dna)
        
        # visualize initial configuration
        fig_init = draw_DnaOrigami(dna, dna.x[:,0:6], SE_init, EE_init)
        fig_init.update_layout(height=400)
        col2.plotly_chart(fig_init, use_container_width=True, height=400) 
        
        # Trial prediction Trial prediction by Ensemble model
        clicked_predict = st.button('Predict')
        clicked_refinement = st.button('Predict with Self-refinement (GPU recommend)')
        y_trial, rmsd_trial, best_model_name, runtime = get_model_prediction(dna, user_input, file_name=file_name, device=device) 
         
        # trial prediction
        if clicked_predict:
            st.write("Prediction on " + device_name)
            PE_trial, SE_trial, EE_trial = total_PE(y_trial, dna)
            print("Best model performance: " + best_model_name)
            print("Init  : Strain energy = %.2e[pNnm] | elec. energy = %.2e[pNnm]" %(SE_init, EE_init))
            print("Trial : Strain energy = %.2e[pNnm] | elec. energy = %.2e[pNnm]" %(SE_trial, EE_trial))
            fig_trial = draw_DnaOrigami(dna, y_trial[:, 0:6], SE_trial, EE_trial)
            fig_trial.update_layout(height=800)
            if rmsd_trial is not None:
                st.write("Predicted configuration: RMSD = %.1f[nm] (runtime = %.1fs)" %(rmsd_trial, runtime))
            else:
                st.write("Predicted configuration: (runtime = %.1fs)" %(runtime))
            st.plotly_chart(fig_trial, use_container_width=True, height=800) 
            
        if clicked_refinement:
            st.write("Self-refinement processing on " + device_name)
            try:
                y_refine, rmsd_refine, run_time = self_refinement(dna, best_model_name, lr=5e-5, num_steps=num_refine_steps, 
                                                     file_name=file_name, device=device)
            except:
                y_refine, rmsd_refine, run_time = self_refinement(dna, best_model_name, num_steps=num_refine_steps, file_name=file_name, device='cpu')
            
            if rmsd_refine is not None:
                st.write("Refinement configuration: RMSD = %.1f[nm] (runtime = %.1fs)" %(rmsd_refine, run_time))
            else:
                st.write("Refinement configuration: (runtime = %.1fs)" %(run_time))
                
            PE_refine, SE_refine, EE_refine = total_PE(y_refine, dna)    
            fig_refine = draw_DnaOrigami(dna, y_refine[:,0:6], SE_refine, EE_refine)
            fig_refine.update_layout(height=800)
            print("Best model performance: " + best_model_name)
            print("Init  : Strain energy = %.2e[pNnm] | elec. energy = %.2e[pNnm]" %(SE_init, EE_init))
            print("Refine: Strain energy = %.2e[pNnm] | elec. energy = %.2e[pNnm]" %(SE_refine, EE_refine))
            st.plotly_chart(fig_refine, use_container_width=True, height=800)


    

  
        
        
            
        
    
    
    
    
