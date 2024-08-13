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



PATH_MODELS = "models_wg_input"
FILE_NAME   = "wg_width_"

try:
    os.mkdir(PATH_MODELS)
except:
    pass



# %%

height_InP = 313e-9
height_SiN = 350e-9


width_start = 250
width_stop  = 1000
width_step = 25
width_array = np.arange(width_start, width_stop+width_step, width_step)*1e-9
print(width_array)


#%%
from analysis_wg import Analysis_wg

found_modes = []
i=0
#%%

for i,width in enumerate(width_array):
    #%%
    env = lumapi.MODE()  

    layoutmode_ = env.layoutmode()
    if layoutmode_ == 0:
        env.eval("switchtolayout;")
    env.deleteall()
    print(f"Starting with width: {width}")
    geom = GeomBuilder(env)
    geom.input_wg(width_top = width, width_bottom=width) # <------------DEFINE THE GEOMETRY

    
    env.groupscope("::model")

    sim = WaveguideModeFinder(env)
    sim.add_simulation_region(width_simulation = 4*width, height_simulation = 4*(height_InP*height_SiN), number_of_trial_modes=20)
    sim.add_mesh(width_mesh= 2*width, height_mesh=2*(height_InP+height_SiN), N=300)

    print("Saving model before simulation...")
    env.save(f"{PATH_MODELS}/{FILE_NAME}{i}")
    print("Simulating...")

    env.run()
    found_modes_ = int(env.findmodes())
    print("Simulation finished!")
    print(f"{found_modes_} modes found")
    found_modes.append(found_modes_)
    print(f"I now found {found_modes_} modes")
    print(f"In total I have {found_modes} modes")

    print("Saving model after simulation...")
    env.save(f"{FILE_NAME}{i}")
    #env.close()
# %%
