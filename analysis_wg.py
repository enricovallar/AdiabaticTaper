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
        ax.set_title(title, fontsize = 10)


    @staticmethod
    def plot_field_zoomed(ax, data, title, zoom_factor=0.75):
        """
        Plots the zoomed-in field data.

        Parameters:
        - ax: Matplotlib Axes object to plot on.
        - data: Dictionary containing field data including 'y', 'z', and 'E2'.
        - title: The title for the plot.
        - zoom_factor: Fraction of the data to display, centered around the middle. Default is 0.75.
        """
        # Extract the original x (data["y"]) and y (data["z"]) ranges
        y_range = data["y"] * 1e6  # Convert to micrometers
        z_range = data["z"] * 1e6  # Convert to micrometers

        # Calculate the center indices
        y_center = len(y_range) // 2
        z_center = len(z_range) // 2

        # Define the zoom range based on the zoom factor
        y_zoom_range = y_range[int(y_center - len(y_range)*(zoom_factor/2)):int(y_center + len(y_range)*(zoom_factor/2))]
        z_zoom_range = z_range[int(z_center - len(z_range)*(zoom_factor/2)):int(z_center + len(z_range)*(zoom_factor/2))]

        # Also zoom in on the corresponding E2 data
        E2_zoom = np.transpose(data["E2"])[
            int(y_center - len(y_range)*(zoom_factor/2)):int(y_center + len(y_range)*(zoom_factor/2)),
            int(z_center - len(z_range)*(zoom_factor/2)):int(z_center + len(z_range)*(zoom_factor/2))
        ]

        # Plot the zoomed data
        ax.set_xlabel("x (\u00B5m)")
        ax.set_ylabel("y (\u00B5m)")
        
        ax.pcolormesh(y_zoom_range, z_zoom_range, E2_zoom, shading='gouraud', cmap='jet', norm=Normalize(vmin=0, vmax=1))
        ax.set_title(title, fontsize=10)

        # Set the aspect ratio to be equal
        ax.set_aspect('equal', 'box')

        

        

    @staticmethod
    def plot_neff( data_array, width_array, title, subtitle=None):
        plt.figure(figsize=(10, 6))
        for data, width in zip(data_array, width_array):
            for mode in data:
                try:
                    f = mode["te_fraction"]
                    color = (f, 0, 1-f)
                    plt.scatter(width/1e-9, mode["neff"].real, color=color)
                except:
                    plt.scatter(width, None)
                
        plt.grid()
        plt.xlabel("width [nm]")
        plt.ylabel("$n_{eff}$")
        plt.title(f"{title}\n{subtitle}")
            

if __name__ == "__main__":
#%%
    import pickle 
    import os
    import numpy as np 
    from analysis_wg import Analysis_wg
    import matplotlib.pyplot as plt
    
    DATAFILE_PATH = '..\input_wg_data_ok.pickle'
    with open(DATAFILE_PATH, 'rb') as file:
        loaded_data = pickle.load(file)
    
    width_array = loaded_data["width_array"]
    found_modes = loaded_data["found_modes"]
    data_array  = loaded_data["data_array"]


    title = "Effective index"
    subtitle = "$313\; nm$ thick $InP$ over $350\; nm$ thick $Si_3N_4$ with the same width. \n Dot color is related to TE polarization fraction, red is TE and blue is TM."
    Analysis_wg.plot_neff(data_array, width_array, title, subtitle)
    
    #%%

    plt.ion()
    for data, width in zip(data_array, width_array):
        figure, axs = plt.subplots(2,2, constrained_layout = True)
        #plt.tight_layout(rect=[0, 0, 1, 0.85])
        
        fig_title = f"Electric field intensity in $InP$ over $Si_3N_4$\n $Width={width/1e-9:.0f}nm$"
        figure.suptitle(fig_title, fontsize=14, fontweight="bold")
        
        i =0
        for mode, ax in zip(data[0:4], axs.flatten()):
            i+=1
            title =  f"$mode\; {i}$: $n_{{eff}} = {np.squeeze(mode['neff'].real):.2f}$, $f_{{TE}}={mode['te_fraction']*100:.0f}\%$"
            Analysis_wg.plot_field_zoomed(ax,mode, title, zoom_factor = 60)
        plt.show()
        plt.pause(0.1)
    plt.ioff()

        
    
    

# %%
