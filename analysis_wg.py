import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


class Analysis_wg:
    
    @staticmethod
    def extract_data(env, mode):
        neff = env.getdata(mode, "neff")
        te_fraction = env.getdata(mode, "TE polarization fraction")
    
        x = np.squeeze(env.getdata(mode,"x"))
        y = np.squeeze(env.getdata(mode,"y"))
        z = np.squeeze(env.getdata(mode,"z"))

        E2 = np.squeeze(env.getelectric(mode))
        H2 = np.squeeze(env.getmagnetic(mode))

        Ex = np.squeeze(env.getdata(mode, "Ex"))
        Ey = np.squeeze(env.getdata(mode, "Ey"))
        Ez = np.squeeze(env.getdata(mode, "Ez"))

        data = { 
            "neff" : neff,
            "te_fraction" : te_fraction,
           
            "x": x,
            "y": y, 
            "z": z, 

            "Ex": Ex, 
            "Ey": Ey,
            "Ez": Ez, 

            "E2": E2,
            "H2": H2
        }

        return data
    
    @staticmethod
    def plot_field(ax, data, title):
        

        ax.set_xlabel("x (\u00B5m)")
        ax.set_ylabel("y (\u00B5m)")
        
        ax.pcolormesh(data["y"]*1e6,data["z"]*1e6,np.transpose(data["E2"]),shading = 'gouraud',cmap = 'jet', norm=Normalize(vmin=0, vmax=1))
        ax.set_title(title)

        

    @staticmethod
    def plot_neff( data_array, width_array, title, subtitle=None):
        for data, width in zip(data_array, width_array):
            for mode in data:
                print(mode)
                try:
                    f = mode["te_fraction"]
                    color = (f, 0, 1-f)
                    plt.scatter(width, mode["neff"].real, c=color)
                except:
                    plt.scatter(width, None)
                
        plt.grid()
        plt.xlabel("width")
        plt.ylabel("$n_{eff}$")
        plt.title(title, pad=40)
        plt.suptitle(subtitle, y=0.97, fontsize=10)
        plt.show()
            

if __name__ == "__main__":
#%%
    import pickle 
    import os
    import numpy as np 
    from AdiabaticTaper.analysis_wg import Analysis_wg
    import matplotlib.pyplot as plt
    
    DATAFILE_PATH = '..\input_wg_data_ok.pickle'
    with open(DATAFILE_PATH, 'rb') as file:
        loaded_data = pickle.load(file)
    
    width_array = loaded_data["width_array"]
    found_modes = loaded_data["found_modes"]
    data_array  = loaded_data["data_array"]


    title = "Effective index"
    subtitle = "$313\; nm$ thick $InP$ over $350\; nm$ thick $Si_3N_4$ with the same size. \n Color is related to TE polarization fraction, red is TE and blue is TM."
    Analysis_wg.plot_neff(data_array, width_array, title, subtitle)
    

    for data, width in zip(data_array, width_array):
        figure, axs = plt.subplots(3,3)
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        fig_title = f"Electric field intensity in $InP$ over $Si_3N_4$\n $Width={width/1e-9:.0f}nm$"
        figure.suptitle(fig_title, fontsize=14, fontweight="bold")
        
        i =0
        for mode, ax in zip(data, axs.flatten()):
            i+=1
            title =  f"$mode={i}$ with $n_{{eff}} = {np.squeeze(mode['neff'].real):.2f}$"
            Analysis_wg.plot_field(ax,mode, title)

    

# %%
