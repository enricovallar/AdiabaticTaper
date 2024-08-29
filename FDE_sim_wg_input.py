#%%

from waveguides_simulations import GeomBuilder
from waveguides_simulations import WaveguideModeFinder
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as plt



PATH_MODELS = rf"D:\WG\models_mg_inputs"
FILE_NAME   = rf"input_wg_height_width"
PATH_DATA   = rf"D:\WG\data"
DATA_FILE_NAME = "wg_input_data.pickle"

try:
    os.mkdir(PATH_MODELS)
except:
    pass


height_InP = 313e-9
height_SiN = 350e-9


width_start = 250
width_stop  = 1000
width_step = 25
width_array = np.arange(width_start, width_stop+width_step, width_step)*1e-9


height_start = 150
height_end   = 1000
height_step = 25
height_array = np.arange(height_start, height_end, height_step)*1e-9

print(f"width_array: {width_array}")
print(f"heighr_array: {height_array}")


from analysis_wg import Analysis_wg

data_points = []
i=0
j=0

for j,height in enumerate(height_array):
    for i,width in enumerate(width_array):
        
        env = lumapi.MODE()  

        layoutmode_ = env.layoutmode()
        if layoutmode_ == 0:
            env.eval("switchtolayout;")
        env.deleteall()
        print(f"Starting with width: {width}")
        geom = GeomBuilder(env)
        geom.input_wg(width_top = width, width_bottom=width, height_top=height) # <------------DEFINE THE GEOMETRY

        
        env.groupscope("::model")

        sim = WaveguideModeFinder(env)
        sim.add_simulation_region(width_simulation = 4* width, height_simulation=4*(height+height_SiN), number_of_trial_modes=20)
        sim.add_mesh(width_mesh= 2*width, height_mesh=2*(height+height_SiN), N=300)
        


        print("Saving model before simulation...")
        env.save(f"{PATH_MODELS}/{FILE_NAME}_{j}_{i}")
        print("Simulating...")

        env.run()
        found_modes = int(env.findmodes())
        print("Simulation finished!")
        print(f"{found_modes} modes found")
        
        print(f"I now found {found_modes} modes")
        print(f"In total I have {found_modes} modes")

        print("Saving model after simulation...")
        env.save(f"{FILE_NAME}_{j}_{i}")
        env.close()

        data_points.append( {
            "width" : width, 
            "height": height, 
            "found_modes": found_modes, 
        })

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
import pickle
import os
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np

PATH_MODELS = rf"D:\WG\models_mg_inputs"
FILE_NAME   = rf"input_wg_height_width"
PATH_DATA   = rf"D:\WG\data"
DATA_FILE_NAME = "wg_input_data.pickle"




width_start = 250
width_stop  = 1000
width_step = 25
width_array = np.arange(width_start, width_stop+width_step, width_step)*1e-9


height_start = 150
height_end   = 1000
height_step = 25
height_array = np.arange(height_start, height_end, height_step)*1e-9


env = lumapi.MODE()
env.load(f"{PATH_MODELS}\{FILE_NAME}_0_0")

data_points=[]
k = 0
for j, height in enumerate(height_array):
    for i, width in enumerate(width_array):

        env.load(f"{FILE_NAME}_{j}_{i}")
        found_modes =[]
        mode_counter = 0
        while True:
            count = mode_counter+1
            try:
                _ = env.getdata(f"FDE::data::mode{count}")
                mode_counter = count
            except:
                break
        found_modes = mode_counter
        data_points.append( {
            "width" : width, 
            "height": height, 
            "found_modes": found_modes, 
        })
        print(f"for w={width/1e-9: .0f} and h={height/1e-9:.0f} I found {found_modes} modes")
#%%

#___________________________________________________
import pickle
import os
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np

PATH_MODELS = rf"D:\WG\models_mg_inputs"
FILE_NAME   = rf"input_wg_height_width"
PATH_DATA   = rf"D:\WG\data"
DATA_FILE_NAME = "wg_input_data.pickle"
width_start = 250
width_stop  = 1000
width_step = 25
width_array = np.arange(width_start, width_stop+width_step, width_step)*1e-9


height_start = 150
height_end   = 1000
height_step = 25
height_array = np.arange(height_start, height_end, height_step)*1e-9
env = lumapi.MODE()
env.load(f"{PATH_MODELS}\{FILE_NAME}_0_0")


with open(f"{PATH_DATA}\{DATA_FILE_NAME}1", 'rb') as file:
    loaded_data = pickle.load(file)
    file.close()
data_points = loaded_data
from analysis_wg_2 import Analysis_wg

k = 0
for j, height in enumerate(height_array):
    for i, width in enumerate(width_array):
        print(f"Calculating for h={height/1e-9:.0f}, w={width/1e-9:.0f}")
        modes_data = []
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
   


    


#%%
PATH_MODELS = rf"D:\WG\models_mg_inputs"
FILE_NAME   = rf"input_wg_height_width"
PATH_DATA   = rf"D:\WG\data"
DATA_FILE_NAME = "wg_input_data.pickle"

if os.path.exists(f"{PATH_DATA}\{DATA_FILE_NAME}"):
    os.remove(f"{PATH_DATA}\{DATA_FILE_NAME}")



with open(f"{PATH_DATA}\{DATA_FILE_NAME}1", 'wb') as file:
    pickle.dump( data_points, file)



# %%
import pickle
import os
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np

PATH_MODELS = rf"D:\WG\models_mg_inputs"
FILE_NAME   = rf"input_wg_height_width"
PATH_DATA   = rf"D:\WG\working_data"
DATA_FILE_NAME = "wg_input_data.pickle"



with open(f"{PATH_DATA}\{DATA_FILE_NAME}", 'rb') as file:
    data_points = pickle.load(file)


#%%
from analysis_wg_2 import Analysis_wg
widths, heights, modes = Analysis_wg.find_te_modes_with_highest_neff(data_points)

plottable_results = []
for mode in modes:
    plottable_results.append(Analysis_wg.get_beta_at_position(mode, 
                                                              y0=0, 
                                                              z0=mode["height"]/2, 
                                                              y_span=mode["width"]*0.9,
                                                              z_span=mode["height"]*0.9
                                                            ))

# %%
#____PLOTS___________________________
import matplotlib.pyplot as plt


# ____PLOT MAP________________________
fig, ax = plt.subplots(figsize=(8, 6), dpi = 300)
fig.tight_layout()
max_width, max_height, max_y, max_z, corresponding_mode = Analysis_wg.find_max_beta_and_mode(plottable_results, modes, ax)
Analysis_wg.plot_cutoff_line(data_points, ax)
print(f"Max $\\beta$-factor occurs at Width: {max_width} µm, Height: {max_height} µm, QD Position (y, z): ({max_y} nm, {max_z} nm)")
print(f"Corresponding Mode: {corresponding_mode}")
plt.savefig(fr"{PATH_DATA}\map.png", dpi=300, bbox_inches='tight')
#__PLOT BETA___________________________
fig, ax = plt.subplots(figsize=(8, 6), dpi = 300)
Analysis_wg.plot_beta_vs_yz(
    corresponding_mode, ax, y_span=2, z_span=2, height_bot=350e-9, 
    colormap='inferno', top_rect_color='blue', bottom_rect_color='blue')
plt.savefig(fr"{PATH_DATA}\beta.png", dpi=300, bbox_inches='tight')
plt.show()

#_________PLOT FIELD___________________
fig, ax = plt.subplots(figsize=(8, 6), dpi = 300)
Analysis_wg.plot_electric_field(
    corresponding_mode, ax, y_span=2, z_span=2, height_bot=350e-9, 
    colormap='jet', top_rect_color='black', bottom_rect_color='black')
plt.savefig(fr"{PATH_DATA}\field_intensity.png", dpi=300, bbox_inches='tight')
plt.show()
    
# %%
