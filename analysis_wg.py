import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import importlib.util

# Load lumapi module (ensure this path is correct)
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
lumapi = importlib.util.module_from_spec(spec_win)
spec_win.loader.exec_module(lumapi)


from mpl_toolkits.mplot3d import Axes3D

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
    def calculate_poynting_vector(data):
        """
        Calculate both the complex Poynting vector and its time-averaged real part from the electric and magnetic field components.

        Parameters:
            data (dict): A dictionary containing the electric and magnetic field components.

        Returns:
            dict: A dictionary with the complex Poynting vector components, the time-averaged real part, and their magnitudes.
        """
        # Extract electric and magnetic field components
        Ex = data["Ex"]
        Ey = data["Ey"]
        Ez = data["Ez"]
        
        Hx = data["Hx"]
        Hy = data["Hy"]
        Hz = data["Hz"]
        
        # Calculate the complex Poynting vector components
        Sx_complex = Ey * np.conj(Hz) - Ez * np.conj(Hy)
        Sy_complex = Ez * np.conj(Hx) - Ex * np.conj(Hz)
        Sz_complex = Ex * np.conj(Hy) - Ey * np.conj(Hx)
        
        # Calculate the time-averaged (real part) Poynting vector components
        Sx_real = 0.5 * np.real(Sx_complex)
        Sy_real = 0.5 * np.real(Sy_complex)
        Sz_real = 0.5 * np.real(Sz_complex)
        
        # Calculate the magnitudes of the Poynting vector
        S_magnitude_complex = np.sqrt(np.abs(Sx_complex)**2 + np.abs(Sy_complex)**2 + np.abs(Sz_complex)**2)
        S_magnitude_real = np.sqrt(Sx_real**2 + Sy_real**2 + Sz_real**2)
        
        # Store the results in a dictionary
        poynting_vector = {
            "Sx_complex": Sx_complex,
            "Sy_complex": Sy_complex,
            "Sz_complex": Sz_complex,
            "S_magnitude_complex": S_magnitude_complex,
            "Sx_real": Sx_real,
            "Sy_real": Sy_real,
            "Sz_real": Sz_real,
            "S_magnitude_real": S_magnitude_real
        }
        
        return poynting_vector


    def purcell_factor(mode_data):
        """
        Calculate the Purcell factors (gamma_y, gamma_z) using the Ey and Ez components of the electric field.

        Parameters:
            mode_data (dict): A dictionary containing the mode data.

        Returns:
            tuple: A tuple (gamma_y, gamma_z) with the Purcell factors calculated using Ey and Ez.
        """
        # Calculate the Poynting vector
        poynting_vector = Analysis_wg.calculate_poynting_vector(mode_data)
        
        # Extract necessary components
        Ey = mode_data["Ey"]
        Ez = mode_data["Ez"]
        Sx_complex = poynting_vector["Sx_complex"]
        
        # Extract y and z arrays from mode_data
        y = mode_data["y"]
        z = mode_data["z"]
        
        # Calculate grid spacings assuming a uniform grid
        dy = np.abs(y[1] - y[0])
        dz = np.abs(z[1] - z[0])
        
        # Perform the integration of Sx_complex over the y-z plane
        integrate_Sx_complex = np.sum(Sx_complex) * dy * dz
        
        # Calculate the Purcell factors for Ey and Ez
        gamma_y = (np.abs(Ey)**2) / integrate_Sx_complex
        gamma_z = (np.abs(Ez)**2) / integrate_Sx_complex
        
        # Return the dictionary
        return {"gamma_y": gamma_y, "gamma_z": gamma_z}
    
    @staticmethod
    def collect_purcell(data_array):
        for data in data_array:
            for mode in data:
                mode["purcell factors"] = Analysis_wg.purcell_factor(mode)



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
    def plot_purcell(ax, data, title, y_span=None, z_span=None, k="y"):
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
        
        purcell_dictionary = data["purcell factors"]
        # Plot the data
        pcm = ax.pcolormesh(data["y"]*1e6, data["z"]*1e6, np.transpose(np.abs(purcell_dictionary[f"gamma_{k}"])), 
                            shading='gouraud', cmap='jet', norm=Normalize(vmin=0, vmax=1))
        
        # Set the title
        ax.set_title(title, fontsize=10)
        
        return pcm

    @staticmethod
    def plot_neff(data_array, width_array, title, subtitle=None):
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
        plt.title(f"{title}\n{subtitle}" if subtitle else title)
        
    
   



