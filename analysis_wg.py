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

        Hx = np.squeeze(env.getdata(mode, "Hx"))
        Hy = np.squeeze(env.getdata(mode, "Hy"))
        Hz = np.squeeze(env.getdata(mode, "Hz"))

        data = { 
            "neff" : neff,
            "te_fraction" : te_fraction,
           
            "x": x,
            "y": y, 
            "z": z, 

            "Ex": Ex, 
            "Ey": Ey,
            "Ez": Ez, 

            "Hx": Hx, 
            "Hy": Hy,
            "Hz": Hz, 

            "E2": E2,
            "H2": H2, 
        }

        return data
    
    @staticmethod
    def plot_field(ax, data, title):
        
    
        ax.set_xlabel("x (\u00B5m)")
        ax.set_ylabel("z (\u00B5m)")
        
        ax.pcolormesh(data["y"]*1e6,data["z"]*1e6,np.transpose(data["E2"]),shading = 'gouraud',cmap = 'jet', norm=Normalize(vmin=0, vmax=1))
        ax.set_title(title, fontsize = 10)


    @staticmethod
    def plot_field(ax, data, title, y_span=None, z_span=None):
        """
        Plots the electric field intensity on a given axis with customizable window size centered at zero.

        Parameters:
        - ax: The axis to plot on.
        - data: A dictionary containing 'y', 'z', and 'E2' keys.
        - title: The title of the plot.
        - y_span: The width of the window in the y-direction in micrometers. Default is None (auto).
        - z_span: The height of the window in the z-direction in micrometers. Default is None (auto).
        """
        # Set axis labels
        ax.set_xlabel("y (\u00B5m)")
        ax.set_ylabel("z (\u00B5m)")
        
        # Calculate limits centered at zero
        if y_span is not None:
            xlim = (-y_span / 2, y_span / 2)
            ax.set_xlim(xlim)
        
        if z_span is not None:
            ylim = (-z_span / 2, z_span / 2)
            ax.set_ylim(ylim)
        
        # Plot the data
        pcm = ax.pcolormesh(data["y"]*1e6, data["z"]*1e6, np.transpose(data["E2"]), 
                            shading='gouraud', cmap='jet', norm=Normalize(vmin=0, vmax=1))
        
        # Set the title
        ax.set_title(title, fontsize=10)
        
        return pcm


        

        

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
            
