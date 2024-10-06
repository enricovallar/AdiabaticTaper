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
    def find_te_modes_with_highest_neff(data_points):
        """
        Find the TE modes with the highest effective index (neff) for each data point.

        Parameters:
            data_points (list): A list of dictionaries, each containing waveguide data and modes.

        Returns:
            tuple: A tuple containing lists of widths, heights, and modes with the highest TE fraction.
        """
        modes = []
        widths = []
        heights = []

        for data_point in data_points:
            width = data_point["width"]
            height = data_point["height"]
            for mode in data_point["modes"]:
                # Check if the mode has a TE fraction greater than 0.5
                if mode["te_fraction"] > 0.5:
                    # Add width and height to the mode dictionary
                    mode["width"] = width
                    mode["height"] = height
                    # Append the width, height, and mode to their respective lists
                    widths.append(width)
                    heights.append(height)
                    modes.append(mode)
                    break  # Stop after finding the first TE mode with the highest neff
        return (widths, heights, modes)
    
    @staticmethod
    def get_beta_at_position(mode, y0=0, z0=313e-9/2, y_span=None, z_span=None):
        """
        Get the beta factor at a specific position (y0, z0) or within a specified span around that position.

        Parameters:
            mode (dict): A dictionary containing the mode data.
            y0 (float): The y-coordinate of the target position (default is 0).
            z0 (float): The z-coordinate of the target position (default is 313e-9 / 2).
            y_span (float, optional): The span around y0 to consider (default is None).
            z_span (float, optional): The span around z0 to consider (default is None).

        Returns:
            dict: A dictionary containing the target (maximum) beta factor, the position of the maximum and mode properties.
        """
        
        # Extract y and z arrays from mode data
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
            z_ = z[z_indices][max_index[1]]

        # Prepare the result dictionary with the target position, beta factor, and mode properties
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
    def map_beta(plottable_results, modes, ax, title, colormap="inferno"):
        """
        Map the beta factor for different waveguide dimensions and highlight the maximum beta factor.

        Parameters:
            plottable_results (list): List of dictionaries containing the results to plot. Each dictionary should contain:
            - 'width': The width of the waveguide.
            - 'height': The height of the waveguide.
            - 'beta_target': The beta factor at the target position.
            - 'y_target': The y-coordinate of the target position.
            - 'z_target': The z-coordinate of the target position.
            - 'te_fraction': The TE polarization fraction.
            modes (list): List of mode data dictionaries.
            ax (matplotlib.axes.Axes): The axes to plot on.
            title (str): The title of the plot.
            colormap (str): The colormap to use for the plot (default is "inferno").

        Returns:
            tuple: A tuple containing the width, height, y, and z of the maximum beta factor, and the corresponding mode.
        """
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
        max_y = int(y_values[np.argmax(beta)]*1e9)  # Convert to integer for whole nm
        max_z = int(z_values[np.argmax(beta)]*1e9)  # Convert to integer for whole nm

        # Convert max_width and max_height to nanometers for the legend
        max_width_nm = int(max_width * 1e3)  # Convert µm to nm
        max_height_nm = int(max_height * 1e3)  # Convert µm to nm

        # Plotting
        contour = ax.contourf(width_grid, height_grid, beta_grid, cmap=colormap, levels=np.linspace(0, 1, 100))
        cbar = plt.colorbar(contour, ax=ax, label=rf"$\beta$-factor")
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        # Highlight the maximum beta value
        ax.scatter(max_width, max_height, color='red',
                   label=rf"""Max $\beta$-factor: {max_beta:.2f}
    QD Position (y, z): ({max_y} nm, {max_z} nm)
    Waveguide (W, H): ({max_width_nm} nm, {max_height_nm} nm)""",
                   edgecolors='black')
        ax.legend()

        ax.set_xlabel('Width (µm)')
        ax.set_ylabel('Height (µm)')
        ax.set_title(rf"""
    {title}
    $\beta$-factor mapping
    """)

        # Find the corresponding mode in the modes array
        corresponding_mode = None
        for mode in modes:
            if mode['width'] * 1e6 == max_width and mode['height'] * 1e6 == max_height:
                corresponding_mode = mode
                break

        return max_width, max_height, max_y, max_z, corresponding_mode
    
    
    
    @staticmethod
    def map_TE_fraction(plottable_results, ax, title, colormap="inferno"):
        """
        Map the TE fraction for different waveguide dimensions.

        Parameters:
            plottable_results (list): List of dictionaries containing the results to plot.
            ax (matplotlib.axes.Axes): The axes to plot on.
            title (str): The title of the plot.
            colormap (str): The colormap to use for the plot (default is "inferno").

        Returns:
            None
        """
        # Convert width and height to micrometers (µm) for plotting
        width = np.array([result['width'] for result in plottable_results]) * 1e6
        height = np.array([result['height'] for result in plottable_results]) * 1e6
        te_fraction = np.array([result['te_fraction'] for result in plottable_results])

        # Create grid data for surface plot
        width_unique = np.unique(width)
        height_unique = np.unique(height)
        width_grid, height_grid = np.meshgrid(width_unique, height_unique)
        te_fraction_grid = np.zeros_like(width_grid, dtype=float)

        # Fill te_fraction_grid with corresponding TE fraction values
        for i in range(len(width)):
            width_index = np.where(width_unique == width[i])[0][0]
            height_index = np.where(height_unique == height[i])[0][0]
            te_fraction_grid[height_index, width_index] = te_fraction[i]

        # Plotting the TE fraction
        contour = ax.contourf(width_grid, height_grid, te_fraction_grid, cmap=colormap, levels=np.linspace(0.5, 1, 100))
        cbar = plt.colorbar(contour, ax=ax, label=rf"TE-fraction (%)")
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        # Set plot labels and title
        ax.set_xlabel('Width (µm)')
        ax.set_ylabel('Height (µm)')
        ax.set_title(rf"""
    {title}
    TE-fraction mapping
    """)

   

    @staticmethod
    def plot_beta_vs_yz(
                        mode, 
                        ax, 
                        title, 
                        y_span_plot: float = 2.0, 
                        z_span_plot: float = 2.0,
                        y0 : float = 0.0, 
                        z0 : float = 313e-9 / 2,
                        y_span_find_beta: float = 0.9,
                        z_span_find_beta: float = 0.9,
                        height_bot=350e-9, 
                        colormap='inferno',
                        top_rect_color='purple', 
                        bottom_rect_color='blue',
                ):
        """
        Plots the beta-factor as a function of y and z coordinates for a given mode.

        Parameters:
        -----------
        mode : dict
            A dictionary containing the mode data with keys "y", "z", "beta", "width", and "height".
            - "y" : numpy array
              The y-coordinates of the mode.
            - "z" : numpy array
              The z-coordinates of the mode.
            - "beta" : numpy array
              The beta-factor values corresponding to the y and z coordinates.
            - "width" : float
              The width of the waveguide.
            - "height" : float
              The height of the waveguide.
        ax : matplotlib.axes.Axes
            The axes object where the plot will be drawn.
        title : str
            The title of the plot.
        y_span_plot : float, optional
            The span of the y-axis in units of waveguide width (default is 2).
        z_span_plot : float, optional
            The span of the z-axis in units of waveguide height (default is 2).
        y0 : float, optional
            The y-coordinate of the target position (default is 0).
        z0 : float, optional
            The z-coordinate of the target position (default is 313e-9 / 2).
        y_span_find_beta : float, optional
            The span around y0 to consider for finding the beta factor (default is 0.9).
        z_span_find_beta : float, optional
            The span around z0 to consider for finding the beta factor (default is 0.9).
        height_bot : float, optional
            The height of the bottom rectangle in meters (default is 350e-9).
        colormap : str, optional
            The colormap to be used for the plot (default is 'inferno').
        top_rect_color : str, optional
            The color of the top rectangle (default is 'purple').
        bottom_rect_color : str, optional
            The color of the bottom rectangle (default is 'blue').

        Returns:
        --------
        None

        Notes:
        ------
        - The function converts the y, z, width, height, and height_bot values to nanometers for plotting.
        - It restricts the beta values to be within the waveguide area and finds the maximum beta value and its location.
        - The position of the maximum beta value is printed for verification.
        - The plot includes rectangles representing the waveguide and a scatter point marking the maximum beta value.
        - The title of the plot is updated to include the waveguide size in nanometers.
        """

        y = mode["y"] 
        z = mode["z"] 
        beta = mode["beta"]
        width = mode["width"] 
        height = mode["height"] 
        height_bot = height_bot 
        
        
        # Define the plotting range for y and z based on the spans
        y_min = -y_span_plot * width / 2
        y_max = y_span_plot* width / 2
        z_min = -z_span_plot* (height + height_bot) / 2
        z_max = z_span_plot * (height + height_bot) / 2
        
        # Determine the indices corresponding to the plotting range
        y_indices = np.where((y >= y_min) & (y <= y_max))[0]
        z_indices = np.where((z >= z_min) & (z <= z_max))[0]
        
        # Create a meshgrid for the selected y and z ranges
        y_grid, z_grid = np.meshgrid(y[y_indices], z[z_indices], indexing='ij')
        beta_grid = beta[np.ix_(y_indices, z_indices)]
        
        # Plot the beta values with the specified colormap and normalized from 0 to 1
        contour = ax.contourf(y_grid, z_grid, beta_grid, cmap=colormap, levels=np.linspace(0, 1, 100))
        cbar = plt.colorbar(contour, ax=ax, label=rf"$\beta$-factor")
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
        # Find the maximum beta value and its location
        result = Analysis_wg.get_beta_at_position(mode, 
                                                  y0=y0, 
                                                  z0=z0, 
                                                  y_span=y_span_find_beta*width, 
                                                  z_span=z_span_find_beta*height)
        
        max_beta = result["beta_target"]
        max_y = result["y_target"]
        max_z = result["z_target"]

        
        # Convert the QD position to nanometers and remove decimal digits
        max_y_nm = int(max_y*1e9)
        max_z_nm = int(max_z*1e9)
        
        # Print the position of the maximum beta for verification
        print(fr"Max $\beta$-factor position: y = {max_y_nm} nm, z = {max_z_nm} nm")
        
        # Mark the maximum beta value
        ax.scatter(max_y, max_z, color='red', s=50, 
                label=rf"""Max $\beta$-factor: {max_beta:.2f}
     QD Position: (y={max_y_nm:.2f} nm, z={max_z_nm:.2f} nm)""")
        
        # Draw the rectangles with specified colors and thicker black lines
        
        if int(height_bot*1e9) !=0: 
            rect1 = Rectangle((-width/2, 0), width, height, 
                            linewidth=2, edgecolor=top_rect_color, facecolor='none')
            rect2 = Rectangle((-width/2, -height_bot), width, height_bot, 
                            linewidth=2, edgecolor=bottom_rect_color, facecolor='none')
            
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            # Draw the horizontal dashed line at height/2
            ax.axhline(height/2, color='black', linestyle='--', linewidth=0.5)
        else:
            rect1 = Rectangle((-width/2, -height/2), width, height, 
                            linewidth=2, edgecolor=top_rect_color, facecolor='none')
            ax.add_patch(rect1)
            # Draw the horizontal dashed line at height/2
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)


        ax.set_aspect('equal', adjustable='box')
        
        
        
        # Update the title to include waveguide size in nm
        ax.set_title(rf"""
    {title}
    $\beta$-factor vs y and z
    Waveguide size: {width:.0f} nm x {height:.0f} nm""")
        
        ax.set_xlabel('y (nm)')
        ax.set_ylabel('z (nm)')

        ax.set_xlim([y_min, y_max])
        ax.set_ylim([z_min, z_max])
        
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax.legend()

    @staticmethod
    def plot_cutoff_line(data_points, ax):
        """
        Plot the single-mode cutoff line based on the data points.

        Parameters:
            data_points (list): List of dictionaries containing waveguide data points.
            ax (matplotlib.axes.Axes): The axes to plot on.

        Returns:
            None
        """
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
        mode, ax, title, y_span=2, z_span=2, height_bot=350e-9, colormap='jet',
        top_rect_color='black', bottom_rect_color='black'):
        """
        Plot the electric field intensity for a given mode.

        Parameters:
            mode (dict): A dictionary containing the mode data.
            ax (matplotlib.axes.Axes): The axes to plot on.
            title (str): The title of the plot.
            y_span (float, optional): The span of the y-axis in units of waveguide width (default is 2).
            z_span (float, optional): The span of the z-axis in units of waveguide height (default is 2).
            height_bot (float, optional): The height of the bottom rectangle in meters (default is 350e-9).
            colormap (str, optional): The colormap to be used for the plot (default is 'jet').
            top_rect_color (str, optional): The color of the top rectangle (default is 'black').
            bottom_rect_color (str, optional): The color of the bottom rectangle (default is 'black').

        Returns:
            None
        """
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
        
        # Draw the rectangles representing the waveguide
        if int(height_bot) != 0: 
            rect1 = Rectangle((-width/2, 0), width, height, 
                            linewidth=2, edgecolor=top_rect_color, facecolor='none')
            rect2 = Rectangle((-width/2, -height_bot), width, height_bot, 
                            linewidth=2, edgecolor=bottom_rect_color, facecolor='none')
            
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            # Draw the horizontal dashed line at height/2
            ax.axhline(height/2, color='black', linestyle='--', linewidth=0.5)
        else:
            rect1 = Rectangle((-width/2, -height/2), width, height, 
                            linewidth=2, edgecolor=top_rect_color, facecolor='none')
            ax.add_patch(rect1)
            # Draw the horizontal dashed line at height/2
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)

        ax.set_aspect('equal', adjustable='box')
        
        # Update the title to include waveguide size in nm
        ax.set_title(rf"""
    {title}
    Electric Field Intensity
    Waveguide size: {width:.0f} nm x {height:.0f} nm
    $n_{{eff}} = {mode['neff'].real.squeeze():0.2f}$, $f_{{TE}}={mode['te_fraction']*100 : 0.0f} \%$
    """)
        
        ax.set_xlabel('y (nm)')
        ax.set_ylabel('z (nm)')

        ax.set_xlim([y_min, y_max])
        ax.set_ylim([z_min, z_max])
        
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)