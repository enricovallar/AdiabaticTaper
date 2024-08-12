#%%
#Importing
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np
from AdiabaticTaper.waveguides_simulations import GeomBuilder
from AdiabaticTaper.waveguides_simulations import WaveguideModeFinder
import pickle 
import os
from AdiabaticTaper.analysis_wg import Analysis_wg
import matplotlib.pyplot as plt

#Realisation
env = lumapi.MODE()  

layoutmode_ = env.layoutmode()
if layoutmode_ == 0:            
    env.eval("switchtolayout;")
env.deleteall()
#print(f"Starting with width: {width}")
geom = GeomBuilder(env)
geom.input_wg()
geom.taper_in()
geom.taper_out()
geom.output_wg()
while True:
    pass
# %%
