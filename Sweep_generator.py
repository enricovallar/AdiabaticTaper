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
PATH_WIDTH = "../Sweeping_models/Sim_width"
PATH_HEIGHT = '../Sweeping_models/Sim_height'
PATH_CONVERGENCE_DATA = '../Sweeping_models'
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
    def sim_width(path: str = PATH_WIDTH,
                    ratio_array: list = ratio_array
                    ):
        for i in range(len(ratio_array)):
            env = lumapi.MODE()
            taper = TaperDesigner(env, mul_w = ratio_array[i])
            print(f"Saving model {i}...")
            taper._env.save(f'{path}/sim_width_{i}')
            taper._env.close()
    
    @staticmethod
    def sim_height(path: str = PATH_HEIGHT,
                     ratio_array: list = ratio_array
                     ):
        for i in range(len(ratio_array)):
            env = lumapi.MODE()
            taper = TaperDesigner(env, mul_h = ratio_array[i])
            print(f"Saving model {i}...")
            taper._env.save(f'{path}/sim_height_{i}')
            taper._env.close()

    @staticmethod
    def convergence_test(path: str = PATH_WIDTH,
                         ratio_array: list = ratio_array
                         ):
        res = []
        for i in range(len(ratio_array)):
            env = lumapi.MODE()
            env.load(f'{path}/sim_width_{i}')
            env.eval('analysis;')
            env.emepropagate()

            S = env.getresult('EME', 'user s matrix')

            S21 = S[1][0]

            T = np.abs(S21)**2
            print(f'Transmission coeff for {i} model is {T}')
            res.append(T)
        plt.plot(ratio_array,res,marker='o')
        plt.xlabel('Ratio of simulation region to the device')
        plt.ylabel('Transmission coeffiecient')
        plt.title('Convergence test on simulation region')

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

sweep_generation.sim_width()
sweep_generation.sim_height()
#sweep_generation.convergence_test()
        



# %%
