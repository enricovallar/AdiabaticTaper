#%%
"""
Waveguide Simulations using Lumerical MODE Solutions
====================================================

This script performs waveguide simulations using Lumerical MODE Solutions and processes the resulting data.

Modules
-------

- **waveguides_simulations**:
    - Contains `GeomBuilder` and `WaveguideModeFinder` classes for building geometries and finding waveguide modes.
- **importlib.util**:
    - Used for dynamic import of the Lumerical API.
- **numpy**:
    - Provides support for large, multi-dimensional arrays and matrices.
- **time**:
    - Provides various time-related functions.
- **pickle**:
    - Used for serializing and de-serializing Python object structures.
- **os**:
    - Provides a way of using operating system-dependent functionality.
- **matplotlib.pyplot**:
    - Used for plotting graphs and visualizations.
- **analysis_wg**:
    - Contains `Analysis_wg` class for analyzing waveguide data and additional analysis functions for waveguide data.

Constants
---------

- `PATH_MODELS (str)`: Path to save the waveguide models.
- `FILE_NAME (str)`: Base name for the waveguide model files.
- `PATH_DATA (str)`: Path to save the simulation data.
- `DATA_FILE_NAME (str)`: Name of the file to save the simulation data.

Variables
---------

- `height_InP (float)`: Height of the InP layer.
- `height_SiN (float)`: Height of the SiN layer.
- `width_start (int)`: Starting value for waveguide widths in nm.
- `width_stop (int)`: Stopping value for waveguide widths in nm.
- `width_step (int)`: Step size for waveguide widths in nm.
- `width_array (numpy.ndarray)`: Array of waveguide widths.
- `height_start (int)`: Starting value for waveguide heights in nm.
- `height_end (int)`: Ending value for waveguide heights in nm.
- `height_step (int)`: Step size for waveguide heights in nm.
- `height_array (numpy.ndarray)`: Array of waveguide heights.
- `data_points (list)`: List to store the simulation results.

Workflow
--------

1. **Setup**:
    - Set up paths and constants.
    - Create directories if they do not exist.
    - Define waveguide dimensions.
2. **Simulation**:
    - Loop through each combination of width and height to perform simulations.
    - Save the simulation results.
3. **Data Analysis**:
    - Load the saved simulation data.
    - Analyze the loaded data to extract relevant information.
4. **Data Visualization**:
    - Plot the results and save the plots.

Usage
-----

Run the script to perform waveguide simulations, save the results, and generate plots for analysis. The script is designed to be run as a standalone module.
"""

if __name__ == "__main__":


    #_____________________________________________________  
    #  1) REALIZE MODELS AND SIMULATE. SAVE DATA FOR LATER. 
    #_____________________________________________________
    import configuration
    from waveguides_simulations import GeomBuilder
    from waveguides_simulations import WaveguideModeFinder
    import importlib.util
    #The default paths for windows
    spec_win = importlib.util.spec_from_file_location('lumapi', configuration.LUMERICAL_API_PATH)
    #Functions that perform the actual loading
    lumapi = importlib.util.module_from_spec(spec_win) 
    spec_win.loader.exec_module(lumapi)
    import numpy as np
    import time
    import pickle
    import os
    import matplotlib.pyplot as plt


    # Path to save the waveguide models
    PATH_MODELS = rf"D:\WG\models_mg_inputs"
    FILE_NAME   = rf"input_wg_height_width"
    # Path to save the simulation data
    PATH_DATA   = rf"D:\WG\data"
    DATA_FILE_NAME = "wg_input_data.pickle"

    try:
        os.mkdir(PATH_MODELS)
    except:
        pass


    # Define default height of the InP and SiN layers in meters
    height_InP = 313e-9
    height_SiN = 350e-9

    # Define the range and step size for waveguide widths in nm
    width_start = 250
    width_stop  = 1000
    width_step = 25
    width_array = np.arange(width_start, width_stop+width_step, width_step)*1e-9


    # Define the range and step size for waveguide heights in nm
    height_start = 150
    height_end   = 1000
    height_step = 25
    height_array = np.arange(height_start, height_end, height_step)*1e-9

    print(f"width_array: {width_array}")
    print(f"heighr_array: {height_array}")



    # Perform simulations for each combination of width and height
    from analysis_wg import Analysis_wg

    data_points = []
    i=0
    j=0

    # Opening Lumerical MODE model 
    env = lumapi.MODE()
    env.save(f"{PATH_MODELS}/{FILE_NAME}_0_0")

    for j,height in enumerate(height_array):
        for i,width in enumerate(width_array):
            
            

            layoutmode_ = env.layoutmode()
            if layoutmode_ == 0:
                env.eval("switchtolayout;")
            env.deleteall()
            print(f"Starting with width: {width}")

            # define the geometry here 
            geom = GeomBuilder(env)
            geom.input_wg(width_top = width, width_bottom=width, height_top=height) # <------------DEFINE THE GEOMETRY

            # back to model
            env.groupscope("::model")

            # define the simulation region and mesh
            sim = WaveguideModeFinder(env)
            sim.add_simulation_region(width_simulation = 4* width, height_simulation=4*(height+height_SiN), number_of_trial_modes=20)
            sim.add_mesh(width_mesh= 2*width, height_mesh=2*(height+height_SiN), N=300)
            
            # save the model before simulation
            print("Saving model before simulation...")
            env.save(f"{PATH_MODELS}/{FILE_NAME}_{j}_{i}")
            
            # run the simulation
            print("Simulating...")
            env.run()
            found_modes = int(env.findmodes())
            print("Simulation finished!")
            print(f"{found_modes} modes found")      
            #print(f"I now found {found_modes} modes")
            #print(f"In total I have {found_modes} modes")
            
            # save the model after simulation
            print("Saving model after simulation...")
            env.save(f"{FILE_NAME}_{j}_{i}")
            
            
            # append the new data to the list
            data_points.append( {
                "width" : width, 
                "height": height, 
                "found_modes": found_modes, 
            })

    #close environment
    env.close()

    # save the data for later use
    if os.path.exists(f"{PATH_DATA}\{DATA_FILE_NAME}"):
        os.remove(f"{PATH_DATA}\{DATA_FILE_NAME}")

    with open(f"{PATH_DATA}\{DATA_FILE_NAME}", 'wb') as file:
        pickle.dump({"data_points": data_points, 
                    "height_array": height_array, 
                    "width_array": width_array}, 
                    file)   

    print("data_saved")



    #%%


    #___________________________________________________
    #  2) LOAD DATA AND ANALYZE. Extract relevant data
    #___________________________________________________
    import pickle
    import os
    import importlib.util
    import configuration
    #The default paths for windows
    spec_win = importlib.util.spec_from_file_location('lumapi', configuration.LUMERICAL_API_PATH)
    #Functions that perform the actual loading
    lumapi = importlib.util.module_from_spec(spec_win) # 
    spec_win.loader.exec_module(lumapi)
    import numpy as np
    from analysis_wg import Analysis_wg

    #specify the paths
    PATH_MODELS = rf"D:\WG\models_mg_inputs"
    FILE_NAME   = rf"input_wg_height_width"
    PATH_DATA   = rf"D:\WG\data"
    DATA_FILE_NAME = "wg_input_data.pickle"

    #load the data
    with open(f"{PATH_DATA}\{DATA_FILE_NAME}1", 'rb') as file:
        loaded_data = pickle.load(file)
        file.close()

    data_points = loaded_data["data_points"]
    width_array = loaded_data["width_array"]
    height_array = loaded_data["height_array"]
    height_array = np.arange(height_start, height_end, height_step)*1e-9

    #open Lumerical MODE envirenment
    env = lumapi.MODE()
    env.load(f"{PATH_MODELS}\{FILE_NAME}_0_0")

    # analyze the data
    k = 0
    for j, height in enumerate(height_array):
        for i, width in enumerate(width_array):
            print(f"Calculating for h={height/1e-9:.0f}, w={width/1e-9:.0f}")
            modes_data = []

            #loop for every mode
            for m in range(data_points[k]["found_modes"]):
                print(f"mode {m}")
                env.load(f"{FILE_NAME}_{j}_{i}")
                mode_name = "FDE::data::mode" + str(m+1)
                mode_data = Analysis_wg.extract_data(env, mode_name)
                Analysis_wg.calculate_poynting_vector(mode_data)
                Analysis_wg.purcell_factor(mode_data, 1550e-9)
                modes_data.append(mode_data)

            for m in range(data_points[k]["found_modes"]):
                Analysis_wg.calculate_beta_factor(modes_data)
            
            # Dataset to big. Store only relevant data. 
            modes_data_to_store =[]
            for m in range(np.min((data_points[k]["found_modes"],4))):
                modes_data_to_store.append(
                    {
                        "z": modes_data[m]["z"],
                        "y": modes_data[m]["y"],
                        "beta": modes_data[m]["beta"],
                        "E2": modes_data[m]["E2"],
                        "te_fraction": modes_data[m]["te_fraction"],
                        "neff": modes_data[m]["neff"]
                    })
            data_points[k]["modes"] = modes_data_to_store
            k+=1
            print("completed.")
    
    #save the data
    PATH_MODELS = rf"D:\WG\models_mg_inputs"
    FILE_NAME   = rf"input_wg_height_width"
    PATH_DATA   = rf"D:\WG\data"
    DATA_FILE_NAME = "wg_input_data.pickle"

    if os.path.exists(f"{PATH_DATA}\{DATA_FILE_NAME}"):
        os.remove(f"{PATH_DATA}\{DATA_FILE_NAME}")



    with open(f"{PATH_DATA}\{DATA_FILE_NAME}1", 'wb') as file:
        pickle.dump( pickle.dump(
            {"data_points": data_points, 
            "height_array": height_array, 
            "width_array": width_array}, 
            file))



    # %%
    #______________LOAD DATA_________________________________________________________________________
    import pickle
    import os
    import importlib.util
    import configuration
    #The default paths for windows
    spec_win = importlib.util.spec_from_file_location('lumapi', configuration.LUMERICAL_API_PATH)
    #Functions that perform the actual loading
    lumapi = importlib.util.module_from_spec(spec_win) # 
    spec_win.loader.exec_module(lumapi)
    import numpy as np


    PATH_MODELS = rf"D:\WG\models_mg_inputs"
    FILE_NAME   = rf"input_wg_height_width"
    PATH_DATA   = rf"D:\WG\working_data"
    DATA_FILE_NAME = "wg_input_data.pickle"
    NAME_SPEC="_glass"


    with open(f"{PATH_DATA}\{DATA_FILE_NAME}", 'rb') as file:
        loaded_data = pickle.load(file)
    print("data_loaded")

    try:
        data_points = loaded_data["data_points"]
        width_array = loaded_data["width_array"]
        height_array = loaded_data["height_array"]
    except:
        data_points = loaded_data # for backward compatibility



    #_______DATA_TO_PLOT_________________

    # 1) Find the modes with the highest neff
    from analysis_wg import Analysis_wg
    widths, heights, modes = Analysis_wg.find_te_modes_with_highest_neff(data_points)

    plottable_results = []
    # 2) Extract the beta factor inside the  InP waveguide
    for mode in modes:
        plottable_results.append(Analysis_wg.get_beta_at_position(mode, 
                                                                y0=0, 
                                                                z0=mode["height"]/2, 
                                                                y_span=mode["width"]*0.9,
                                                                z_span=mode["height"]*0.9
                                                                ))


    #____PLOTS___________________________
    import matplotlib.pyplot as plt
    title = "InP over SiN in Glass"

    # ____PLOT BETA MAP________________________
    fig, ax = plt.subplots(figsize=(8, 6), dpi = 300)
    fig.tight_layout()
    # 2D plot of the beta factor with respect to the waveguide width and height
    # The maximum beta factor is marked with a red dot
    # The corresponding mode is  returned with the target values and the position 
    # of the maximum beta factor inside the corresponding waveguide
    max_width, max_height, max_y, max_z, corresponding_mode = Analysis_wg.map_beta(
        plottable_results, 
        modes, 
        ax, 
        title
    )
    # Draw a line to indicate the cutoff width in the same figure
    Analysis_wg.plot_cutoff_line(data_points, ax)
    print(f"Max $\\beta$-factor occurs at Width: {max_width} µm, Height: {max_height} µm, QD Position (y, z): ({max_y} nm, {max_z} nm)")
    # saving figure
    plt.savefig(fr"{PATH_DATA}\map_beta{NAME_SPEC}.png", dpi=300, bbox_inches='tight')

    # ____PLOT TE FRACTION MAP________________________
    fig, ax = plt.subplots(figsize=(8, 6), dpi = 300)
    fig.tight_layout()
    # 2D plot of the TE fraction with respect to the waveguide width and height
    Analysis_wg.map_TE_fraction(plottable_results, ax, title)
    Analysis_wg.plot_cutoff_line(data_points, ax)
    plt.savefig(fr"{PATH_DATA}\map_TE_fraction{NAME_SPEC}.png", dpi=300, bbox_inches='tight')


    #__PLOT BETA___________________________
    fig, ax = plt.subplots(figsize=(8, 6), dpi = 300)
    # 2D plot of the beta factor with respect to the position inside the waveguide
    # The position of the maximum beta factor is marked with a red dot
    # The previouly found mode is used to find the beta factor
    Analysis_wg.plot_beta_vs_yz(
        corresponding_mode, ax, title,  
        y_span_plot=2,
        z_span_plot=2, 
        y0 = 0, 
        z0 = 313e-9/2,
        y_span_find_beta= 0.9,
        z_span_find_beta= 0.9,
        height_bot=350e-9, 
        colormap='inferno', 
        top_rect_color='blue', 
        bottom_rect_color='blue')
    # saving figure
    plt.savefig(fr"{PATH_DATA}\beta{NAME_SPEC}.png", dpi=300, bbox_inches='tight')
    plt.show()

    #_________PLOT FIELD___________________
    # 2D plot of the electric field intensity with respect to the position inside the waveguide
    # The previously found mode is used to plot the electric field intensity
    fig, ax = plt.subplots(figsize=(8, 6), dpi = 300)
    Analysis_wg.plot_electric_field(
        corresponding_mode, 
        ax, 
        title,  
        y_span=2, 
        z_span=2, 
        height_bot=350e-9, 
        colormap='jet', 
        top_rect_color='black', 
        bottom_rect_color='black')
    # saving figure

    plt.savefig(fr"{PATH_DATA}\field_intensity{NAME_SPEC}.png", dpi=300, bbox_inches='tight')
    plt.show()
        
    # %%
