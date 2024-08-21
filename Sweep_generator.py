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
PATH_CELL_NUMBER = '../Sweeping_models/Cell_number'
PATH_MESH_WIDTH = '../Sweeping_models/Mesh_width'
PATH_MESH_HEIGHT = '../Sweeping_models/Mesh_height'
PATH_MESH_RESOLUTION_WIDTH = '../Sweeping_models/Mesh_resolution_width'
PATH_MESH_RESOLUTION_HEIGHT = '../Sweeping_models/Mesh_resolution_height'
PATH_CONVERGENCE_DATA = '../Sweeping_models'
sim_size_ratio = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
cell_number = [5,10,15,20,25,30,35,40,45,50]
mesh_size_ratio = [1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]
mesh_resolution = [5e-9,10e-9,15e-9,20e-9,25e-9,30e-9,35e-9,40e-9,45e-9,50e-9]
sim_size_ratio_FOTONANO_2 = [1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8,8.25,8.5,8.75,9]
PATH_FOTONANO_2 = '../Sweeping_models/Sim_width_FOTONANO_2'

#Create directories
try:
    os.mkdir(PATH_WIDTH)
except:
    pass
try:
    os.mkdir(PATH_CELL_NUMBER)
except:
    pass
try:
    os.mkdir(PATH_MESH_HEIGHT)
except:
    pass
try:
    os.mkdir(PATH_MESH_WIDTH)
except:
    pass
try:
    os.mkdir(PATH_MESH_RESOLUTION_HEIGHT)
except:
    pass
try:
    os.mkdir(PATH_MESH_RESOLUTION_WIDTH)
except:
    pass
try:
    os.mkdir(PATH_HEIGHT)
except:
    pass

#Defining class
class sweep_generation:

    @staticmethod
    def sim_width(path: str = PATH_WIDTH,
                    ratio_array: list = sim_size_ratio,
                    ):
        env = lumapi.MODE()
        env.save(f'{path}/sim_width_0')
        for i in range(len(ratio_array)):
            taper = TaperDesigner(env, mul_h=4, width_in=400e-9, width_out=1e-6, mul_w = ratio_array[i], dz=5e-9, dy=5e-9)
            print(f"Saving model {i} with simulation width ratio {ratio_array[i]}")
            taper._env.save(f'sim_width_{i}')
            taper._env.deleteall()
        env.close()
    
    @staticmethod
    def sim_height(path: str = '../Sweeping_models/Sim_height_FOTONANO',
                     ratio_array: list = sim_size_ratio_FOTONANO_2
                     ):
        env = lumapi.MODE()
        env.save(f'{path}/sim_height_0')
        for i in range(len(ratio_array)):
            taper = TaperDesigner(env, width_in=400e-9, width_out=1e-6, mul_w=4, mul_h = ratio_array[i], dx=20e-9, dy=20e-9)
            print(f"Saving model {i} with simulation height ratio {ratio_array[i]}")
            taper._env.save(f'sim_height_{i}')
            taper._env.deleteall()
        env.close()

    @staticmethod
    def cell_number(path: str = PATH_CELL_NUMBER,
                    cell_number: list = cell_number
                    ):
        env = lumapi.MODE()
        env.save(f'{path}/cell_number_0')
        for i in range(len(cell_number)):
            taper = TaperDesigner(env, cell_number = cell_number[i])
            print(f'Saving model {i+1} with {cell_number[i]} cells')
            taper._env.save(f'cell_number_{i+1}')
            taper._env.deleteall()
        env.close()

    @staticmethod
    def mesh_width(path: str = PATH_MESH_WIDTH,
                   mesh_size_ratio: list = mesh_size_ratio
                   ):
        env = lumapi.MODE()
        env.save(f'{path}/mesh_width_0')
        for i in range(len(mesh_size_ratio)):
            taper = TaperDesigner(env, mul_w_mesh=mesh_size_ratio[i])
            print(f'Saving model {i+1} with mesh width ratio {mesh_size_ratio[i]}')
            taper._env.save(f'mesh_width_{i+1}')
            taper._env.deleteall()
        env.close()

    @staticmethod
    def mesh_height(path: str = PATH_MESH_HEIGHT,
                    mesh_size_ratio: list = mesh_size_ratio
                    ):
        env = lumapi.MODE()
        env.save(f'{path}/mesh_height_0')
        for i in range(len(mesh_size_ratio)):
            taper = TaperDesigner(env, mul_h_mesh=mesh_size_ratio[i])
            print(f'Saving model {i+1} with mesh height ratio {mesh_size_ratio[i]}')
            taper._env.save(f'mesh_height_{i+1}')
            taper._env.deleteall()
        env.close()

    @staticmethod
    def mesh_resolution_width(path: str = PATH_MESH_RESOLUTION_WIDTH,
                              mesh_resolution: list = mesh_resolution
                              ):
        env = lumapi.MODE()
        env.save(f'{path}/mesh_resolution_width_0')
        for i in range(len(mesh_resolution)):
            taper = TaperDesigner(env,dy=mesh_resolution[i])
            print(f'Saving model {i+1} with mesh width resolution {mesh_resolution[i]}')
            taper._env.save(f'mesh_resolution_width_{i+1}')
            taper._env.deleteall()
        env.close()

    @staticmethod
    def mesh_resolution_height(path: str = PATH_MESH_RESOLUTION_HEIGHT,
                              mesh_resolution: list = mesh_resolution
                              ):
        env = lumapi.MODE()
        env.save(f'{path}/mesh_resolution_height_0')
        for i in range(len(mesh_resolution)):
            taper = TaperDesigner(env,dz=mesh_resolution[i])
            print(f'Saving model {i+1} with mesh height resolution {mesh_resolution[i]}')
            taper._env.save(f'mesh_resolution_height_{i+1}')
            taper._env.deleteall()
        env.close()


    @staticmethod
    def convergence_test(path: str = PATH_WIDTH,
                         ratio_array: list = sim_size_ratio
                         ):
        res = []
        for i in range(2,len(ratio_array)):
            env = lumapi.MODE()
            env.load(f'{path}/sim_width_{i}')
            env.eval('analysis;')
            env.run()
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

if __name__=='__main__':
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
    sim_size_ratio_FOTONANO_2 = [1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8,8.25,8.5,8.75,9]
    PATH_FOTONANO_2 = '../Sweeping_models/Sim_width_FOTONANO_2'

    #sweep_generation.sim_width(path=PATH_FOTONANO_2, ratio_array=sim_size_ratio_FOTONANO_2)
    #sweep_generation.sim_height()
    #sweep_generation.cell_number()
    #sweep_generation.mesh_height()
    #sweep_generation.mesh_width()
    #sweep_generation.mesh_resolution_height()
    #sweep_generation.mesh_resolution_width()
    sweep_generation.convergence_test(path=PATH_FOTONANO_2, ratio_array=sim_size_ratio_FOTONANO_2)

            



    # %%
