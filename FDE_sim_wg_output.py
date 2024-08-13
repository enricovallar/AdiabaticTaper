from AdiabaticTaper.waveguides_simulations import GeomBuilder
from AdiabaticTaper.waveguides_simulations import WaveguideModeFinder
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


width = 500e-6
PATH_FOLDER = "output_waveguide"

try:
    os.mkdir(PATH_FOLDER)
except:
    pass



# %%
width_start = 100
width_stop  = 1000
width_step = 50
width_array = np.arange(width_start, width_stop+width_step, width_step)*1e-9
print(width_array)


#%%
from AdiabaticTaper.analysis_wg import Analysis_wg
PATH = "output_waveguide"
found_modes = []
i=0
#%%

for i,width in enumerate(width_array):
    
    env = lumapi.MODE()  

    layoutmode_ = env.layoutmode()
    if layoutmode_ == 0:
        env.eval("switchtolayout;")
    env.deleteall()
    print(f"Starting with width: {width}")
    geom = GeomBuilder(env)
    geom.output_wg(width = width)

    
    env.groupscope("::model")

    sim = WaveguideModeFinder(env)
    sim.add_simulation_region(width_simulation = 4*width, height_simulation = 5*313e-9, number_of_trial_modes=20)
    sim.add_mesh(width_mesh= 2*width, height_mesh=2*313e-9, N=300)

    print("Saving model before simulation...")
    env.save(PATH + "/output_modes_"+str(i))
    print("Simulating...")

    env.run()
    found_modes_ = int(env.findmodes())
    print("Simulation finished!")
    print(f"{found_modes_} modes found")
    found_modes.append(found_modes_)
    print(f"I now found {found_modes_} modes")
    print(f"In total I have {found_modes} modes")

    print("Saving model after simulation...")
    env.save("output_modes_"+str(i))
    env.close()