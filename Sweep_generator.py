#Import Modules
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import pickle 
import os
import numpy as np 
from analysis_wg import Analysis_wg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from taperDesigner import TaperDesigner

#Path to stored folder
PATH_WIDTH = "../Sweeping_models/Width_sweep"
PATH_HEIGHT = '../Sweeping_models/Height_sweep'
ratio_array = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]


try:
    os.mkdir(PATH_WIDTH)
except:
    pass

try:
    os.mkdir(PATH_HEIGHT)
except:
    pass

class sweep_generation:

    @staticmethod
    def width_sweep(path: str = PATH_WIDTH,
                    ratio_array: list = ratio_array
                    ):
        for i in range(len(ratio_array)):
            env = lumapi.MODE()
            taper = TaperDesigner(env, mul_w = ratio_array[i])
            print(f"Saving model {i+1}...")
            taper._env.save(f'{path}/width_sweep_{i+1}')
            taper._env.close()
    
    @staticmethod
    def height_sweep(path: str = PATH_HEIGHT,
                     ratio_array: list = ratio_array
                     ):
        for i in range(len(ratio_array)):
            env = lumapi.MODE()
            taper = TaperDesigner(env, mul_h = ratio_array[i])
            print(f"Saving model {i+1}...")
            taper._env.save(f'{path}/height_sweep_{i+1}')
            taper._env.close()

    @staticmethod
    def 

#%%
#Import Modules
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import pickle 
import os
import numpy as np 
from analysis_wg import Analysis_wg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from taperDesigner import TaperDesigner
from Sweep_generator import sweep_generation

sweep_generation.width_sweep()
sweep_generation.height_sweep()
        



# %%
