#%%
#Importing
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np
from waveguides_simulations import GeomBuilder
from waveguides_simulations import WaveguideModeFinder
import pickle 
import os
from analysis_wg import Analysis_wg
import matplotlib.pyplot as plt


from waveguides_simulations import taper_geometry

env = lumapi.MODE()
width_in   = 500e-9
width_out = 1000e-9
width_tip  = 50e-9
m_in = 0.8
m_out = 7
length_taper = 19e-6
taper_geometry(env=env, 
               width_in = width_in, width_out=width_out, width_tip=width_tip, 
               m_in= m_in, m_out=m_out, 
               length_taper=length_taper
               )
# %%
