import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from scipy.integrate import simps

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
        Sx_real = np.real(Sx_complex)
        Sy_real = np.real(Sy_complex)
        Sz_real = np.real(Sz_complex)
        
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


    def purcell_factor(mode_data, lam0):
        """
        Calculate the Purcell factors (gamma_y, gamma_z) using the Ey and Ez components of the electric field.

        Parameters:
            mode_data (dict): A dictionary containing the mode data.

        Returns:
            dict: A dictionary with the Purcell factors calculated using Ey and Ez components.
        """
        # Calculate the Poynting vector
        poynting_vector = Analysis_wg.calculate_poynting_vector(mode_data)
        
        # Constants
        pi = np.pi
        c = 3e8  # Speed of light in vacuum (m/s)
        epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)
        
        # Calculate k0
        k0 = 2 * pi / lam0
        
        # Constant part of the expression
        constant_part = (3 * pi * c * epsilon_0) / k0**2

        # Extract necessary components
        Ey = mode_data["Ey"]
        Ez = mode_data["Ez"]
        Sx = poynting_vector["Sx_real"]
        
        # Extract y and z arrays from mode_data
        y = mode_data["y"]
        z = mode_data["z"]
        
        # Calculate grid spacings in a non  uniform grid
        # Integrate along the x-axis for each y slice
        integrate_Sx = simps(Sx, x=y, axis=1)

        # Now integrate the result along the y-axis
        integrate_Sx = simps(integrate_Sx, x=z)
        
        
        # Calculate the Purcell factors for Ey and Ez
        gamma_y = (np.abs(Ey)**2) / integrate_Sx*constant_part
        gamma_z = (np.abs(Ez)**2) / integrate_Sx*constant_part
        
        # Return the dictionary
        return {"gamma_y": gamma_y, "gamma_z": gamma_z}
    
    @staticmethod
    def collect_purcell(data_array, lam0=1550e-9):
        """
        Calculate and collect Purcell factors for each mode in a data array.
        
        Parameters:
            data_array (list): A list of dictionaries, each representing mode data.

        Returns:
            None: This function modifies the input list in place, adding Purcell factors to each mode.
        """
        for data in data_array:
            for mode in data:
                mode["purcell_factors"] = Analysis_wg.purcell_factor(mode, lam0)



    @staticmethod
    def plot_field(ax, data, title, y_span=None, z_span=None, normalize=False):
        """
        Plots the electric field intensity on a given axis with customizable window size centered at zero.

        Parameters:
        - ax: The axis to plot on.
        - data: A dictionary containing 'y', 'z', and 'E2' keys.
        - title: The title of the plot.
        - y_span: The width of the window in the y-direction in micrometers. Default is None (auto).
        - z_span: The height of the window in the z-direction in micrometers. Default is None (auto).
        - normalize: Boolean flag to normalize the plot. Default is False.
        
        Returns:
        - pcm: The plot colormesh object for further customization.
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
        
        if normalize:
            # Plot the data
            pcm = ax.pcolormesh(data["y"]*1e6, data["z"]*1e6, np.transpose(data["E2"]), 
                                shading='gouraud', cmap='jet', norm=Normalize(vmin=0, vmax=1))
        else:
            pcm = ax.pcolormesh(data["y"]*1e6, data["z"]*1e6, np.transpose(data["E2"]), 
                                shading='gouraud', cmap='jet')

        
        # Set the title
        ax.set_title(title, fontsize=10)
        
        return pcm

    @staticmethod
    def plot_purcell(ax, data, title, purcell_key="purcell_factors", y_span=None, z_span=None, k="y", normalize=False):
        """
        Plots the Purcell factor on a given axis with customizable window size centered at zero.

        Parameters:
        - ax: The axis to plot on.
        - data: A dictionary containing 'y', 'z', and Purcell factors.
        - title: The title of the plot.
        - purcell_key: The key to access the Purcell factors in the data dictionary. Default is 'purcell_factors_normalized'.
        - y_span: The width of the window in the y-direction in micrometers. Default is None (auto).
        - z_span: The height of the window in the z-direction in micrometers. Default is None (auto).
        - k: The key to select the Purcell factor direction ('y' or 'z'). Default is 'y'.
        - normalize: Boolean flag to normalize the plot. Default is False.
        
        Returns:
        - pcm: The plot colormesh object for further customization.
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
        
        
        purcell_dictionary = data[purcell_key]
        # Plot the data
        if normalize:
            pcm = ax.pcolormesh(data["y"]*1e6, data["z"]*1e6, np.transpose(purcell_dictionary[f"gamma_{k}"]), 
                                shading='gouraud', cmap='jet', norm=Normalize(vmin=0, vmax=1))
        else:
            pcm = ax.pcolormesh(data["y"]*1e6, data["z"]*1e6, np.transpose(purcell_dictionary[f"gamma_{k}"]), 
                                shading='gouraud', cmap='jet')

        
        # Set the title
        ax.set_title(title, fontsize=10)

        # Add the colorbar
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Purcell factor')
        
        return pcm
       
    
    

    @staticmethod
    def calculate_beta_factor(data_array):
        """
        Calculate the beta factor for each mode in the data array.

        Parameters:
        - data_array: List of mode data dictionaries.

        Returns:
        - None: This function modifies the input list in place, adding beta factors to each mode.
        """
        for data in data_array:
            P_y = []
            P_z = []
            for mode in data:
                purcell_factors = mode["purcell_factors"]
                gamma_y = purcell_factors["gamma_y"]
                gamma_z = purcell_factors["gamma_z"]
                P_y.append(gamma_y)
                P_z.append(gamma_z)
            
            for i, (Pn_y, Pn_z, mode) in enumerate(zip(P_y, P_z, data)):
                
                P_others_y = P_y[:i] + P_y[i+1:]

                # Sum the arrays in P_others
                P_sum_y = np.sum(P_others_y, axis=0)  # This sums P2 + P3 + ... + PN element-wise

                # Calculate B using the given formula
                beta_y = 1 / (1 + (1 + P_sum_y) / Pn_y)

                P_others_z = P_z[:i] + P_z[i+1:]

                # Sum the arrays in P_others
                P_sum_z = np.sum(P_others_z, axis=0)  # This sums P2 + P3 + ... + PN element-wise

                # Calculate B using the given formula
                beta_z = 1 / (1 + (1 + P_sum_z) / Pn_z)

                beta_dictionary = {
                    "beta_y" : beta_y,
                    "beta_z" : beta_z
                }
                
                mode["beta_factors"] = beta_dictionary
                



    @staticmethod
    def plot_beta(ax, data, title, y_span=None, z_span=None, k="y", normalize=True):
        """
        Plots the beta factor on a given axis with customizable window size centered at zero.

        Parameters:
        - ax: The axis to plot on.
        - data: A dictionary containing 'y', 'z', and beta factors.
        - title: The title of the plot.
        - y_span: The width of the window in the y-direction in micrometers. Default is None (auto).
        - z_span: The height of the window in the z-direction in micrometers. Default is None (auto).
        - k: The key to select the beta factor direction ('y' or 'z'). Default is 'y'.
        - normalize: Boolean flag to normalize the plot. Default is True.
        
        Returns:
        - pcm: The plot colormesh object for further customization.
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
        
        
        beta_dictionary = data["beta_factors"]
        # Plot the data
        if normalize:
            pcm = ax.pcolormesh(data["y"]*1e6, data["z"]*1e6, np.transpose(beta_dictionary[f"beta_{k}"]), 
                                shading='gouraud', cmap='jet', norm=Normalize(vmin=0, vmax=1))
        else:
            pcm = ax.pcolormesh(data["y"]*1e6, data["z"]*1e6, np.transpose(beta_dictionary[f"beta_{k}"]), 
                                shading='gouraud', cmap='jet')

        
        # Set the title
        ax.set_title(title, fontsize=10)

        # Add the colorbar
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Beta')
        
        return pcm



    @staticmethod
    def integrate_beta_in_region(data_array, y_0, z_0, y_span, z_span):
        """
        Integrate the beta factor over a specified region of space for each mode in the data array.

        Parameters:
            data_array (list): A list of dictionaries where each dictionary contains mode data, 
                            including 'y', 'z', and 'beta_factors'.
            y_0 (float): Center of the region in the y-direction (in meters).
            z_0 (float): Center of the region in the z-direction (in meters).
            y_span (float): Width of the region in the y-direction (in meters).
            z_span (float): Height of the region in the z-direction (in meters).

        Returns:
            None: The method adds the integrated beta values for y and z directions to each mode's dictionary within the data array.
        """
        # Iterate over each data dictionary in the data_array
        for data in data_array:
            for mode in data:
                # Extract y and z arrays
                y = mode["y"]
                z = mode["z"]
                beta_y = mode["beta_factors"]["beta_y"]
                beta_z = mode["beta_factors"]["beta_z"]

                # Define the region limits
                y_min = y_0 - y_span / 2
                y_max = y_0 + y_span / 2
                z_min = z_0 - z_span / 2
                z_max = z_0 + z_span / 2

                # Find indices that are within the specified region
                y_indices = np.where((y >= y_min) & (y <= y_max))[0]
                z_indices = np.where((z >= z_min) & (z <= z_max))[0]
                y_ = y[y_indices]
                z_ = z[z_indices]

                # Extract the subarray of beta within the specified range
                beta_subarray_y = beta_y[np.ix_(y_indices, z_indices)]
                beta_subarray_z = beta_z[np.ix_(y_indices, z_indices)]

                

                # Integrate beta over the specified region
                beta_integral_y = simps(beta_subarray_y, x=z_, axis=1)
                beta_integral_y = simps(beta_integral_y, x=y_ )


                beta_integral_z = simps(beta_subarray_z, x=z_, axis=1)
                beta_integral_z = simps(beta_integral_z, x=y_)
                


                # Add the integrated beta values to the mode's dictionary
                mode["beta_integrals"] = {
                    "beta_integral_y": beta_integral_y,
                    "beta_integral_z": beta_integral_z
                }



            
    @staticmethod
    def normalize_beta_integrals(data_array):
        """
        Normalize the beta integrals over a specified set of data.

        Parameters:
            data_array (list): A list of dictionaries where each dictionary contains mode data, 
                               including 'beta_integrals' calculated by the integrate_beta_in_region method.

        Returns:
            None: The method normalizes the beta integrals in place and adds the normalized values to each mode's dictionary.
        """
        # Initialize variables to store the maximum beta integrals
        beta_integral_max = 0

        # First pass: find the global maximum beta integral value
        for data in data_array:
            for mode in data: 
                beta_integrals = mode["beta_integrals"]
                beta_integral_max = max(beta_integral_max, beta_integrals["beta_integral_y"])
                beta_integral_max = max(beta_integral_max, beta_integrals["beta_integral_z"])

        # Second pass: normalize the beta integrals
        for data in data_array: 
            for mode in data: 
                beta_integrals = mode["beta_integrals"]
                beta_integral_y_normalized = beta_integrals["beta_integral_y"] / beta_integral_max
                beta_integral_z_normalized = beta_integrals["beta_integral_z"] / beta_integral_max
                beta_integrals_normalized = {
                    "beta_integral_y_normalized": beta_integral_y_normalized,
                    "beta_integral_z_normalized": beta_integral_z_normalized
                }
                mode["beta_integrals_normalized"] = beta_integrals_normalized


    @staticmethod
    def plot_neff(data_array, width_array, title, subtitle=None):
        """
        Plots the effective index (neff) as a function of width for a series of modes.
        The size of the plot markers is scaled by the normalized beta_integral, and the TE fraction 
        determines whether to use beta_integral_y_normalized or beta_integral_z_normalized.

        Parameters:
        - data_array: List of mode data dictionaries.
        - width_array: List of widths corresponding to the modes.
        - title: The main title for the plot.
        - subtitle: Optional subtitle for the plot. Default is None.
        """
        plt.figure(figsize=(10, 6))
        
        for data, width in zip(data_array, width_array):
            for mode in data:
                try:
                    # Determine whether to use beta_integral_y_normalized or beta_integral_z_normalized
                    if mode["te_fraction"] > 0.5:
                        beta_integral_normalized = mode["beta_integrals_normalized"]["beta_integral_y_normalized"]
                    else:
                        beta_integral_normalized = mode["beta_integrals_normalized"]["beta_integral_z_normalized"]
                    
                    # Scale marker size based on normalized beta_integral
                    marker_size = beta_integral_normalized * 100  # Adjust this factor for better scaling
                    
                    # Determine color based on TE fraction
                    f = mode["te_fraction"]
                    color = (f, 0, 1-f)
                    
                    # Plot the point
                    plt.scatter(width/1e-9, mode["neff"].real, color=color, s=marker_size, alpha=0.7)
                    
                except KeyError as e:
                    print(f"Missing data in mode: {e}")
                    plt.scatter(width/1e-9, None, color='gray')

        plt.grid()
        plt.xlabel("width [nm]")
        plt.ylabel("$n_{eff}$")
        plt.title(f"{title}\n{subtitle}" if subtitle else title)




    #Function to draw contour
    @staticmethod
    def draw_contour(ax,
                    height = [313e-9],
                    width = 550e-9
                    ):
        for h in height:
            height[height.index(h)]=h/1e-6
        width=width/1e-6
        if len(height)==1:
            ax.add_patch(Rectangle((-width/2,-height[0]/2),width,height[0],fill=None))
        elif len(height)==2:
            ax.add_patch(Rectangle((-width/2,0),width,height[0],fill=None))
            ax.add_patch(Rectangle((-width/2,-height[1]),width,height[1],fill=None))
        else:
            pass


    @staticmethod
    def normalize_beta_gradients(data_array):
        """
        Normalize the beta gradients (gradient_y and gradient_z) over a specified set of data,
        considering only the points within y_span equal to the width of the mode, centered at 0.

        Parameters:
            data_array (list): A list of dictionaries where each dictionary contains mode data,
                            including 'beta_gradients' calculated by the calculate_beta_gradient method.

        Returns:
            None: The method normalizes the beta gradients in place and adds the normalized values to each mode's dictionary.
        """
        # Initialize variable to store the maximum absolute gradient value
        max_gradient = 0

        # First pass: find the global maximum absolute gradient value within the specified y_span
        for data in data_array:
            for mode in data:
                if "beta_gradients" in mode:
                    gradients = mode["beta_gradients"]
                    y = mode["y"]
                    width = 0.95*mode["width"]

                    # Determine the range of y values that fall within y_span centered at 0
                    y_min = -width / 2
                    y_max = width / 2

                    # Find indices that fall within this range
                    y_indices = np.where((y >= y_min) & (y <= y_max))[0]

                    # Extract the relevant parts of the gradients
                    gradient_y_filtered = gradients["gradient_y"][y_indices]
                    gradient_z_filtered = gradients["gradient_z"][y_indices]

                    # Update max_gradient with the maximum value found within the filtered range
                    max_gradient = max(max_gradient, np.max(np.abs(gradient_y_filtered)))
                    max_gradient = max(max_gradient, np.max(np.abs(gradient_z_filtered)))

        # Second pass: normalize the beta gradients using the determined max_gradient
        for data in data_array:
            for mode in data:
                if "beta_gradients" in mode:
                    gradients = mode["beta_gradients"]
                    y = mode["y"]
                    width = 0.95*mode["width"]

                    # Determine the range of y values that fall within y_span centered at 0
                    y_min = -width / 2
                    y_max = width / 2

                    # Find indices that fall within this range
                    y_indices = np.where((y >= y_min) & (y <= y_max))[0]

                    # Extract and normalize the relevant parts of the gradients
                    gradient_y_normalized = gradients["gradient_y"]/ max_gradient
                    gradient_z_normalized = gradients["gradient_z"]/ max_gradient

                    gradients_normalized = {
                        "gradient_y": gradient_y_normalized,
                        "gradient_z": gradient_z_normalized, 
                        "max_gradient": max_gradient
                    }
                    mode["beta_gradients_normalized"] = gradients_normalized



    @staticmethod
    def plot_beta_gradient(ax, mode, z_position, title=None, normalize= False, xlim = None):
        """
        Plots the gradient of the beta factor (beta_y or beta_z) along the y-axis 
        at a specific z position for a given mode. The plot is determined by the TE fraction.
        The gradients are normalized with the same value.

        Parameters:
        - ax: The matplotlib axis on which to plot the data.
        - mode: A dictionary containing the mode data, including 'y', 'z', and 'beta_factors'.
        - z_position: The specific z position (in meters) at which to calculate and plot the gradient.
        - title: Optional title for the plot. Default is None.
        - ylim: Tuple (min, max) specifying the limits for the y-axis. Default is None.
        - xlim: Tuple (min, max) specifying the limits for the x-axis. Default is None.
        
        Returns:
        - None: The function directly plots on the provided axis.
        """
        if normalize:
            beta_gradients = mode["beta_gradients_normalized"]
            width = mode["width"]/1e-6
            ylim =(-1,1)
            ax.set_ylim(ylim)
        else: 
            beta_gradients = mode["beta_gradients"]
            ylim = (-mode["beta_gradients_normalized"]["max_gradient"],
                         mode["beta_gradients_normalized"]["max_gradient"])
            width = mode["width"]/1e-6
            ax.set_ylim(ylim)

        # Extract data
        y = mode["y"] * 1e6  # Convert y to micrometers

        if mode["te_fraction"] > 0.5:
            # Plot the gradient of beta_y in red
            gradient = beta_gradients["gradient_y"]
            color = 'red'
            label = f"Gradient of beta_y (z = {z_position * 1e6:.2f} µm)"
        else:
            # Plot the gradient of beta_z in blue
            gradient = beta_gradients["gradient_z"]
            color = 'blue'
            label = f"Gradient of beta_z (z = {z_position * 1e6:.2f} µm)"
        
        # Plot the gradient
        line, = ax.plot(y, gradient,"--" ,  color=color, label=label)

        # Set axis labels and title
        ax.set_xlabel("y (µm)")
        ax.set_ylabel("Gradient of $\\beta$")
        if title:
            ax.set_title(title)
        
        if xlim is not None: 
            ax.set_xlim(xlim)
        else:
            xlim = (-0.95*width/2, 0.95*width/2)

            ax.set_xlim(xlim)
            

        #ax.legend()
        ax.grid(True)
        return line

    @staticmethod
    def calculate_beta_gradient(data_array, z_):
        """
        Calculate the gradient of the beta factor along the y-axis at a specific z position (z_)
        and save the results directly into each mode's dictionary.

        Parameters:
            data_array (list): A list of dictionaries where each dictionary contains mode data, 
                            including 'y', 'z', and 'beta_factors'.
            z_ (float): The specific z position (in meters) at which to calculate the gradient.

        Returns:
            None: The method adds the gradient values to each mode's dictionary within the data array.
        """

        for data in data_array:
            for mode in data:
                y = mode["y"]
                z = mode["z"]
                beta_y = mode["beta_factors"]["beta_y"]
                beta_z = mode["beta_factors"]["beta_z"]

                # Find the index closest to the specified z_ position
                z_index = (np.abs(z - z_)).argmin()

                # Extract the beta values at the specified z position
                beta_y_at_z = beta_y[:, z_index]
                beta_z_at_z = beta_z[:, z_index]

                # Calculate the gradient with respect to y
                gradient_y = np.gradient(beta_y_at_z, y)
                gradient_z = np.gradient(beta_z_at_z, y)

                # Save the gradients in the mode dictionary
                mode["beta_gradients"] = {
                    "gradient_y": gradient_y,
                    "gradient_z": gradient_z,
                    "y": y,
                    "z_index": z_index,
                    "z_value": z[z_index]
                }


    @staticmethod
    def plot_beta_at_z(z_, ax, mode):
        """
        Plots the values of beta_y or beta_z at a specific value of z = z_ on the right y-axis.
        Plots beta_y if TE fraction > 0.5, otherwise plots beta_z.

        Parameters:
        - z_ (float): The specific z position (in meters) at which to plot the beta values.
        - ax: The matplotlib axis on which to plot the data. This will be the left axis, and a new right axis will be created.
        - mode: A dictionary containing the mode data, including 'y', 'z', and 'beta_factors'.

        Returns:
        - None: The function directly plots on the provided axis.
        """
        
        # Create a secondary y-axis on the right
        ax_right = ax.twinx()

        # Extract the beta factors and spatial coordinates
        y = mode["y"] * 1e6  # Convert y to micrometers
        z = mode["z"]
        beta_y = mode["beta_factors"]["beta_y"]
        beta_z = mode["beta_factors"]["beta_z"]
        
        # Find the index closest to the specified z_ value
        z_index = (np.abs(z - z_)).argmin()
        
        # Determine which beta factor to plot based on the TE fraction
        if mode["te_fraction"] > 0.5:
            beta_values = beta_y[:, z_index]
            color = 'red'
            label = f"beta_y at z = {z_ * 1e6:.2f} µm"
        else:
            beta_values = beta_z[:, z_index]
            color = 'blue'
            label = f"beta_z at z = {z_ * 1e6:.2f} µm"
        
        # Plot the beta values along the y-axis at z = z_ on the right y-axis
        line, = ax_right.plot(y, beta_values, color=color, label=label)
        
        # Set axis labels and title
        ax.set_xlabel("y (µm)")
        ax.set_title(f"$\\beta$-factor z = {z_ * 1e6:.2f} µm")
        
        ax_right.set_ylabel("$\\beta$-factor", color=color)
        ax_right.tick_params(axis='y', labelcolor=color)
        
        # Add legend and grid
        #ax_right.legend()
        ax_right.grid(True)

        ax_right.set_ylim((0,1))
        return line



    @staticmethod
    def get_beta_at_position(mode, y0, z0):
        y = mode["y"] * 1e6  # Convert y to micrometers
        z = mode["z"]
        beta_y = mode["beta_factors"]["beta_y"]
        beta_z = mode["beta_factors"]["beta_z"]
        
    



   
