import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from scipy.integrate import simps
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

import importlib.util

# Load lumapi module (ensure this path is correct)
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
lumapi = importlib.util.module_from_spec(spec_win)
spec_win.loader.exec_module(lumapi)


from mpl_toolkits.mplot3d import Axes3D

class Analysis_wg:
    
    @staticmethod
    def extract_data(env, mode_name):
        neff = env.getdata(mode_name, "neff")
        te_fraction = env.getdata(mode_name, "TE polarization fraction")
    
        x = np.squeeze(env.getdata(mode_name,"x"))
        y = np.squeeze(env.getdata(mode_name,"y"))
        z = np.squeeze(env.getdata(mode_name,"z"))

        Ex = np.squeeze(env.getdata(mode_name, "Ex"))
        Ey = np.squeeze(env.getdata(mode_name, "Ey"))
        Ez = np.squeeze(env.getdata(mode_name, "Ez"))

        Hx = np.squeeze(env.getdata(mode_name, "Hx"))
        Hy = np.squeeze(env.getdata(mode_name, "Hy"))
        Hz = np.squeeze(env.getdata(mode_name, "Hz"))
        
        E2 = np.squeeze(env.getelectric(mode_name))

        mode_data = { 
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
            "E2": E2
        }

        return mode_data
    
    @staticmethod
    def calculate_poynting_vector(mode_data):
        """
        Calculate both the complex Poynting vector and its time-averaged real part from the electric and magnetic field components.

        Parameters:
            data (dict): A dictionary containing the electric and magnetic field components.

        Returns:
            dict: A dictionary with the complex Poynting vector components, the time-averaged real part, and their magnitudes.
        """
        # Extract electric and magnetic field components
        Ey = mode_data["Ey"]
        Ez = mode_data["Ez"]
        
        Hy = mode_data["Hy"]
        Hz = mode_data["Hz"]
        
        # Calculate the complex Poynting vector components
        Sx = Ey * np.conj(Hz) - Ez * np.conj(Hy)

        # Store the results in a dictionary
        mode_data["Sx"] = Sx
        
        


    def purcell_factor(mode_data, lam0):
        """
        Calculate the Purcell factors (gamma_y, gamma_z) using the Ey and Ez components of the electric field.

        Parameters:
            mode_data (dict): A dictionary containing the mode data.

        Returns:
            dict: A dictionary with the Purcell factors calculated using Ey and Ez components.
        """
        
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
        Sx = mode_data["Sx"].real
        
        # Extract y and z arrays from mode_data
        y = mode_data["y"]
        z = mode_data["z"]
        
        # Calculate grid spacings in a non  uniform grid
        # Integrate along the x-axis for each y slice
        integrate_Sx = simps(Sx, x=y, axis=1)

        # Now integrate the result along the y-axis
        integrate_Sx = simps(integrate_Sx, x=z)
        
        
        # Calculate the Purcell factors for Ey and Ez
        gamma = (np.abs(Ey)**2) / integrate_Sx*constant_part
        mode_data["gamma"] = gamma
        
    


    @staticmethod
    def plot_field(ax, mode_data, title, y_span=None, z_span=None, normalize=False):
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
            pcm = ax.pcolormesh(mode_data["y"]*1e6, mode_data["z"]*1e6, np.transpose(mode_data["E2"]), 
                                shading='gouraud', cmap='jet', norm=Normalize(vmin=0, vmax=1))
        else:
            pcm = ax.pcolormesh(mode_data["y"]*1e6, mode_data["z"]*1e6, np.transpose(mode_data["E2"]), 
                                shading='gouraud', cmap='jet')

        
        # Set the title
        ax.set_title(title, fontsize=10)
        
        return pcm
      
   

    @staticmethod
    def calculate_beta_factor(modes_data):
        """
        Calculate the beta factor for each mode in the data array.

        Parameters:
        - data_array: List of mode data dictionaries.

        Returns:
        - None: This function modifies the input list in place, adding beta factors to each mode.
        """
        
        P = []
        for mode in modes_data: 
            P.append(mode["gamma"])
        
        for n, (Pn, mode) in enumerate(zip(P, modes_data)):
            
            P_others = P[:n] + P[n+1:]

            # Sum the arrays in P_others
            P_sum = np.sum(P_others, axis=0)  # This sums P2 + P3 + ... + PN element-wise

            # Calculate B using the given formula
            beta = 1 / (1 + (1 + P_sum) / Pn)                
            modes_data[n]["beta"] = beta
            



    
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

        #if mode["te_fraction"] > 0.5:
        # Plot the gradient of beta_y in red
        gradient = beta_gradients["gradient_y"]
        color = 'orange'
        label = f"Gradient of $\\beta$-factor at $z={z_position * 1e9:.0f}nm$)"
        # else:
        #     # Plot the gradient of beta_z in blue
        #     gradient = beta_gradients["gradient_z"]
        #     color = 'blue'
        #     label = f"Gradient of beta_z (z = {z_position * 1e6:.3f} µm)"
        
        # Plot the gradient
        line, = ax.plot(y, gradient,"--" ,  color=color, label=label)

        # Set axis labels and title
        ax.set_xlabel("y (µm)")
        ax.set_ylabel("Gradient of $\\beta$-factor", color= color)
        ax.tick_params(axis='y', labelcolor=color)
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
        #if mode["te_fraction"] > 0.5:
        beta_values = beta_y[:, z_index]
        color = 'red'
        label = f"$\\beta$-factor at $z = {z_ * 1e9:.0f} nm$"
        # else:
        #     beta_values = beta_z[:, z_index]
        #     color = 'blue'
        #     label = f"beta_z at z = {z_ * 1e9:.0f} µm"
        
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
    def get_beta_at_position(mode, y0=0, z0=313e-9/2, y_span=None, z_span=None):
                
        y = mode["y"]
        z = mode["z"]
        
        if y_span is None and z_span is None:
            # If no span is provided, find the closest single point
            y_index = (np.abs(y - y0)).argmin()
            z_index = (np.abs(z - z0)).argmin()
            beta = mode["beta"][y_index, z_index]
            y_ = y[y_index]
            z_ = z[z_index]
        else:
            # Determine the range of indices within the span
            if y_span is not None:
                y_min = y0 - y_span / 2
                y_max = y0 + y_span / 2
                y_indices = np.where((y >= y_min) & (y <= y_max))[0]
            else:
                y_indices = range(len(y))
            
            if z_span is not None:
                z_min = z0 - z_span / 2
                z_max = z0 + z_span / 2
                z_indices = np.where((z >= z_min) & (z <= z_max))[0]
            else:
                z_indices = range(len(z))
            
            # Extract the relevant beta values and find the maximum
            beta = np.max(mode["beta"][np.ix_(y_indices, z_indices)])
            
            # Find the position of this maximum beta
            max_index = np.unravel_index(np.argmax(mode["beta"][np.ix_(y_indices, z_indices)]), (len(y_indices), len(z_indices)))
            y_ = y[y_indices][max_index[0]]
            z_= z[z_indices][max_index[1]]

        result = {
            "z_target": z_,
            "y_target": y_,
            "beta_target": beta,
            "width": mode["width"],
            "height": mode["height"],
            "neff": mode["neff"],
            "te_fraction": mode["te_fraction"]
        }
        
            
        return result


 


    @staticmethod
    def find_te_modes_with_highest_neff(data_points):
        modes=[]
        widths =[]
        heights=[]

        for data_point in data_points:
            width = data_point["width"]
            height = data_point["height"]
            for mode in data_point["modes"]:
                if mode["te_fraction"]>0.5:
                    mode["width"] = width
                    mode["height"] = height
                    widths.append(width)
                    heights.append(height)
                    modes.append(mode)
                    break
        return (widths, heights, modes)
    

    def find_max_beta_and_mode(plottable_results, modes, ax, colormap = "inferno"):
        # Convert width and height to micrometers (µm) for axes, and to nanometers (nm) for legend
        width = np.array([result['width'] for result in plottable_results]) * 1e6  # For plotting (µm)
        height = np.array([result['height'] for result in plottable_results]) * 1e6  # For plotting (µm)
        beta = np.array([result['beta_target'] for result in plottable_results])
        y_values = np.array([result['y_target'] for result in plottable_results]) * 1e9  # Convert to nm
        z_values = np.array([result['z_target'] for result in plottable_results]) * 1e9  # Convert to nm
        
        # Create grid data for surface plot
        width_unique = np.unique(width)
        height_unique = np.unique(height)
        
        width_grid, height_grid = np.meshgrid(width_unique, height_unique)
        beta_grid = np.zeros_like(width_grid, dtype=float)

        # Fill beta_grid with corresponding beta values
        for i in range(len(width)):
            width_index = np.where(width_unique == width[i])[0][0]
            height_index = np.where(height_unique == height[i])[0][0]
            beta_grid[height_index, width_index] = beta[i]
        
        # Find the maximum beta value and its corresponding width, height, y, and z
        max_beta = np.max(beta_grid)
        max_index = np.unravel_index(np.argmax(beta_grid), beta_grid.shape)
        max_width = width_grid[max_index]
        max_height = height_grid[max_index]

        # Get the y and z corresponding to the maximum beta
        max_y = int(y_values[np.argmax(beta)])  # Convert to integer for whole nm
        max_z = int(z_values[np.argmax(beta)])  # Convert to integer for whole nm

        # Convert max_width and max_height to nanometers for the legend
        max_width_nm = int(max_width * 1e3)  # Convert µm to nm
        max_height_nm = int(max_height * 1e3)  # Convert µm to nm
        
        # Plotting
        contour = ax.contourf(width_grid, height_grid, beta_grid, cmap=colormap, levels=np.linspace(0, 1, 100))
        plt.colorbar(contour, ax=ax)
        
        # Highlight the maximum beta value
        ax.scatter(max_width, max_height, color='red', 
                label=rf"""Max $\beta$-factor: {max_beta:.2f}
    QD Position (y, z): ({max_y} nm, {max_z} nm)
    Waveguide (W, H): ({max_width_nm} nm, {max_height_nm} nm)""", 
                edgecolors='black')
        ax.legend()
        
        ax.set_xlabel('Width (µm)')
        ax.set_ylabel('Height (µm)')
        ax.set_title(r'$\beta$-factor mapping')

        # Find the corresponding mode in the modes array
        corresponding_mode = None
        for mode in modes:
            if mode['width'] * 1e6 == max_width and mode['height'] * 1e6 == max_height:
                corresponding_mode = mode
                break
        
        return max_width, max_height, max_y, max_z, corresponding_mode

    

   

    @staticmethod
    def plot_beta_vs_yz(
        mode, ax, y_span=2, z_span=2, height_bot=350e-9, colormap='inferno',
        top_rect_color='purple', bottom_rect_color='blue'):
        
        y = mode["y"] * 1e9  # Convert y to nanometers (nm)
        z = mode["z"] * 1e9  # Convert z to nanometers (nm)
        beta = mode["beta"]
        width = mode["width"] * 1e9  # Convert width to nanometers (nm)
        height = mode["height"] * 1e9  # Convert height to nanometers (nm)
        height_bot = height_bot * 1e9  # Convert height_bot to nanometers (nm)
        
        # Define the plotting range for y and z based on the spans
        y_min = -y_span * width / 2
        y_max = y_span * width / 2
        z_min = -z_span * (height + height_bot) / 2
        z_max = z_span * (height + height_bot) / 2
        
        # Determine the indices corresponding to the plotting range
        y_indices = np.where((y >= y_min) & (y <= y_max))[0]
        z_indices = np.where((z >= z_min) & (z <= z_max))[0]
        
        # Create a meshgrid for the selected y and z ranges
        y_grid, z_grid = np.meshgrid(y[y_indices], z[z_indices], indexing='ij')
        beta_grid = beta[np.ix_(y_indices, z_indices)]
        
        # Plot the beta values with the specified colormap and normalized from 0 to 1
        contour = ax.contourf(y_grid, z_grid, beta_grid, cmap=colormap, levels=np.linspace(0, 1, 100))
        plt.colorbar(contour, ax=ax, label=rf"$\beta$-factor")
        
        # Find the maximum beta value and its location
        max_beta = np.max(beta_grid)
        max_index = np.unravel_index(np.argmax(beta_grid, axis=None), beta_grid.shape)
        max_y = y_grid[max_index]
        max_z = z_grid[max_index]
        
        # Convert the QD position to nanometers and remove decimal digits
        max_y_nm = int(max_y)
        max_z_nm = int(max_z)
        
        # Print the position of the maximum beta for verification
        print(f"Max $\beta$-factor position: y = {max_y_nm} nm, z = {max_z_nm} nm")
        
        # Mark the maximum beta value
        ax.scatter(max_y, max_z, color='red', s=50, 
                label=rf"""Max $\beta$-factor: {max_beta:.2f}
    QD Position: (y={max_y_nm} nm, z={max_z_nm} nm)""")
        
        # Draw the rectangles with specified colors and thicker black lines
        rect1 = Rectangle((-width/2, 0), width, height, 
                        linewidth=2, edgecolor=top_rect_color, facecolor='none')
        rect2 = Rectangle((-width/2, -height_bot), width, height_bot, 
                        linewidth=2, edgecolor=bottom_rect_color, facecolor='none')
        
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        
        # Draw the horizontal dashed line at height/2
        ax.axhline(height/2, color='black', linestyle='--', linewidth=1.5)
        
        # Update the title to include waveguide size in nm
        ax.set_title(rf"""$\beta$-factor vs y and z
    Waveguide size: {width:.0f} nm x {height:.0f} nm""")
        
        ax.set_xlabel('y (nm)')
        ax.set_ylabel('z (nm)')

        ax.set_xlim([y_min, y_max])
        ax.set_ylim([z_min, z_max])
        
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax.legend()

    @staticmethod
    def plot_cutoff_line(data_points, ax):
        # Extract width and height values from data_points
        width = np.array([dp['width'] for dp in data_points]) * 1e6  # Convert to micrometers (µm)
        height = np.array([dp['height'] for dp in data_points]) * 1e6  # Convert to micrometers (µm)

        # Finding and plotting the cutoff line
        unique_heights = np.unique(height)
        cutoff_widths = []

        for h in unique_heights:
            # Filter data points for the current height
            filtered_points = [dp for dp in data_points if dp['height'] * 1e6 == h]

            # Sort by width to find the first width with more than two modes
            filtered_points.sort(key=lambda dp: dp['width'])

            # Find the first width where found_modes is greater than two
            for dp in filtered_points:
                try:
                    if dp['found_modes'] > 2:
                        cutoff_widths.append((dp['width'] * 1e6, h))  # Store the width and height
                        break
                except KeyError:
                    continue

        # Plot the cut-off line
        if cutoff_widths:
            cutoff_widths = np.array(cutoff_widths)
            ax.plot(cutoff_widths[:, 0], cutoff_widths[:, 1], 'k--', label='Single-mode Cutoff')


        ax.legend()

    
    def plot_electric_field(
        mode, ax, y_span=2, z_span=2, height_bot=350e-9, colormap='jet',
        top_rect_color='black', bottom_rect_color='black'):
        
        y = mode["y"] * 1e9  # Convert y to nanometers (nm)
        z = mode["z"] * 1e9  # Convert z to nanometers (nm)
        E2 = mode["E2"]  # Electric field intensity
        width = mode["width"] * 1e9  # Convert width to nanometers (nm)
        height = mode["height"] * 1e9  # Convert height to nanometers (nm)
        height_bot = height_bot * 1e9  # Convert height_bot to nanometers (nm)
        
        # Define the plotting range for y and z based on the spans
        y_min = -y_span * width / 2
        y_max = y_span * width / 2
        z_min = -z_span * (height + height_bot) / 2
        z_max = z_span * (height + height_bot) / 2
        
        # Determine the indices corresponding to the plotting range
        y_indices = np.where((y >= y_min) & (y <= y_max))[0]
        z_indices = np.where((z >= z_min) & (z <= z_max))[0]
        
        # Create a meshgrid for the selected y and z ranges
        y_grid, z_grid = np.meshgrid(y[y_indices], z[z_indices], indexing='ij')
        E2_grid = E2[np.ix_(y_indices, z_indices)]
        
        # Plot the electric field intensity with the specified colormap
        contour = ax.contourf(y_grid, z_grid, E2_grid, cmap=colormap, levels=100)
        plt.colorbar(contour, ax=ax, label=r"$E^2$ (a.u.)")
        
        # Draw the rectangles representing the waveguide structure
        rect1 = Rectangle((-width/2, 0), width, height, 
                        linewidth=2, edgecolor=top_rect_color, facecolor='none')
        rect2 = Rectangle((-width/2, -height_bot), width, height_bot, 
                        linewidth=2, edgecolor=bottom_rect_color, facecolor='none')
        
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        
        # Draw the horizontal dashed line at height/2
        ax.axhline(height/2, color='black', linestyle='--', linewidth=1.5)
        
        # Update the title to include waveguide size in nm
        ax.set_title(rf"""Electric Field Intensity
    Waveguide size: {width:.0f} nm x {height:.0f} nm""")
        
        ax.set_xlabel('y (nm)')
        ax.set_ylabel('z (nm)')

        ax.set_xlim([y_min, y_max])
        ax.set_ylim([z_min, z_max])
        
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)