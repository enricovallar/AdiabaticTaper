import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np

class LumObj:
    """
    Base class for handling interactions with the simulation environment and managing object configurations.
    """
    
    def __init__(self, env):
        """
        Initialize the LumObj class with the given simulation environment.

        Parameters
        ----------
        env : lumapi.MODE or similar
            The simulation environment object.
        """
        self._env = env
        self._configuration = ()

    def update_configuration(self):
        """
        Update the configuration for each object with its corresponding parameters.
        This method applies stored configurations (name and parameter values) to the environment objects.
        """
        for obj, parameters in self.configuration:
            for k, v in parameters:
                self.env.setnamed(obj, k, v)

    def add_to_configuration(self, new_element_configuration):
        """
        Parameters
        ----------
        new_element_configuration : tuple
            A tuple representing the configuration to add.
        Returns
        -------
        None
        """
        
        self.configuration = self.configuration + new_element_configuration

    @property
    def env(self):
        """
        Return the simulation environment object.
        
        Returns
        -------
        lumapi.MODE or similar
            The simulation environment.

        """
        return self._env

    @property 
    def configuration(self):
        """
        Return the current configuration.

        Returns
        -------
        tuple
            The current configuration.
        """
        return self._configuration

    @configuration.setter
    def configuration(self, new_configuration):
        """
        Set a new configuration.

        Parameters
        ----------
        new_configuration : tuple
            The new configuration tuple.

        Returns
        -------
        None
        """
        self._configuration = new_configuration


import importlib.util
import numpy as np
class LumObj:
    """
    Base class for handling interactions with the simulation environment and managing object configurations.

    Parameters
    ----------
    env : lumapi.MODE or similar
        The simulation environment object (e.g., Lumerical MODE).
    """
    
    def __init__(self, env):
        """
        Initialize the LumObj class with the given simulation environment.

        Parameters
        ----------
        env : lumapi.MODE or similar
            The simulation environment object.
        """
        self._env = env
        self._configuration = ()

    def update_configuration(self):
        """
        Update the configuration for each object with its corresponding parameters.
        This method applies stored configurations (name and parameter values) to the environment objects.
        """
        for obj, parameters in self.configuration:
            for k, v in parameters:
                self.env.setnamed(obj, k, v)

    def add_to_configuration(self, new_element_configuration):
        """
        Add new elements to the existing configuration.

        Parameters
        ----------
        new_element_configuration : tuple
            A tuple representing the configuration to add.
        """
        self.configuration = self.configuration + new_element_configuration

    @property
    def env(self):
        """
        Return the simulation environment object.
        
        Returns
        -------
        lumapi.MODE or similar
            The simulation environment.
        """
        return self._env

    @property 
    def configuration(self):
        """
        Return the current configuration.

        Returns
        -------
        tuple
            The current configuration.
        """
        return self._configuration

    @configuration.setter
    def configuration(self, new_configuration):
        """
        Set a new configuration.

        Parameters
        ----------
        new_configuration : tuple
            The new configuration tuple.
        """
        self._configuration = new_configuration


class GeomBuilder(LumObj):
    """
    GeomBuilder class for creating and configuring geometric structures in the simulation environment.
    Inherits from LumObj.

    Parameters
    ----------
    env : lumapi.MODE or similar
        The simulation environment object.
    """

    def input_wg(
        self,
        name_top="InP_input_wg",
        name_bottom="SiN_input_wg", 
        material_top="InP - Palik",
        material_bottom="Si3N4 (Silicon Nitride) - Phillip",
        material_background="SiO2 (Glass) - Palik", 
        width_top: float = 550e-9,
        width_bottom: float = 550e-9,
        height_top: float = 313e-9, 
        height_bottom: float = 350e-9,
        length_top: float = 10e-6, 
        length_bottom: float = 10e-6,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        x_min: float = None,
        y_min: float = None,
        z_min: float = None,
        layout_group_name="Input Waveguide"
    ):
        """
        Create input waveguides and add them to the simulation environment.

        Parameters
        ----------
        name_top : str, optional
            Name of the top waveguide, defaults to "InP_input_wg".
        name_bottom : str, optional
            Name of the bottom waveguide, defaults to "SiN_input_wg".
        material_top : str, optional
            Material of the top waveguide, defaults to "InP - Palik".
        material_bottom : str, optional
            Material of the bottom waveguide, defaults to "Si3N4 (Silicon Nitride) - Phillip".
        material_background : str, optional
            Background material, defaults to "SiO2 (Glass) - Palik".
        width_top : float, optional
            Width of the top waveguide, defaults to 550e-9.
        width_bottom : float, optional
            Width of the bottom waveguide, defaults to 550e-9.
        height_top : float, optional
            Height of the top waveguide, defaults to 313e-9.
        height_bottom : float, optional
            Height of the bottom waveguide, defaults to 350e-9.
        length_top : float, optional
            Length of the top waveguide, defaults to 10e-6.
        length_bottom : float, optional
            Length of the bottom waveguide, defaults to 10e-6.
        x : float, optional
            X-coordinate of the center, defaults to 0.
        y : float, optional
            Y-coordinate of the center, defaults to 0.
        z : float, optional
            Z-coordinate of the center, defaults to 0.
        x_min : float, optional
            Minimum X-coordinate, defaults to None.
        y_min : float, optional
            Minimum Y-coordinate, defaults to None.
        z_min : float, optional
            Minimum Z-coordinate, defaults to None.
        layout_group_name : str, optional
            Layout group name, defaults to "Input Waveguide".
        """
        # Set center positions if the minimum values are provided
        if x_min is not None: x = x_min / 2
        if y_min is not None: y = y_min / 2
        if z_min is not None: z = z_min / 2
        
        env = self.env

        # Create top waveguide
        env.addrect()
        env.set("name", name_top)
        env.addtogroup(layout_group_name)
        
        # Create bottom waveguide
        env.addrect()
        env.set("name", name_bottom)
        env.addtogroup(layout_group_name)
        
        # Configuration for the geometry
        configuration_geometry = (
            (name_top, (
                ("material", material_top),
                ("x", x),
                ("y", y),
                ("z min", z),
                ("x span", length_top),
                ("y span", width_top),
                ("z max", z + height_top)
            )),
            (name_bottom, (
                ("material", material_bottom),
                ("x", x),
                ("y", y),
                ("z min", z - height_bottom),
                ("x span", length_bottom),
                ("y span", width_bottom),
                ("z max", z)
            )),
        )
        
        # Set the group scope and update the configuration
        env.groupscope(f"::model::{layout_group_name}")

        self.configuration = configuration_geometry
        self.update_configuration()

        env.groupscope('::model')

    def output_wg(
        self,
        name='Output_wg',
        material='Si3N4 (Silicon Nitride) - Phillip',
        width=1100e-9,
        height=350e-9,
        length=2e-6,
        x=0,
        y=0,
        z=0,
        x_min=None,
        y_min=None,
        z_min=None,
        layout_group_name='Output Waveguide'
    ):
        """
        Create an output waveguide with specified parameters and add it to the simulation environment.

        Parameters
        ----------
        name : str, optional
            Name of the output waveguide, defaults to 'Output_wg'.
        material : str, optional
            Material of the output waveguide, defaults to 'Si3N4 (Silicon Nitride) - Phillip'.
        width : float, optional
            Width of the output waveguide, defaults to 1100e-9.
        height : float, optional
            Height of the output waveguide, defaults to 350e-9.
        length : float, optional
            Length of the output waveguide, defaults to 2e-6.
        x : float, optional
            X-coordinate of the center, defaults to 0.
        y : float, optional
            Y-coordinate of the center, defaults to 0.
        z : float, optional
            Z-coordinate of the center, defaults to 0.
        x_min : float, optional
            Minimum X-coordinate, defaults to None.
        y_min : float, optional
            Minimum Y-coordinate, defaults to None.
        z_min : float, optional
            Minimum Z-coordinate, defaults to None.
        layout_group_name : str, optional
            Layout group name, defaults to "Output Waveguide".
        """
        env = self.env

        # Set center positions if the minimum values are provided
        if x_min is not None: x = x_min / 2
        if y_min is not None: y = y_min / 2
        if z_min is not None: z = z_min / 2

        # Create waveguide
        env.addrect()
        env.set('name', name)
        env.addtogroup(layout_group_name)

        # Configuration for the geometry
        configuration_geometry = (
            (name, (
                ('material', material),
                ('x', x),
                ('y', y),
                ('z', z),
                ('y span', width),
                ('z span', height),
                ('x span', length)
            )),
        )

        # Set the group scope and update the configuration
        env.groupscope(f'::model::{layout_group_name}')
        self.configuration = configuration_geometry
        self.update_configuration()
        env.groupscope('::model')

    def taper_top(
        self,
        name='Taper_in',
        height: float = 313e-9,
        length: float = 19e-6,
        width_in: float = 550e-9,
        width_out: float = 50e-9,
        material: str = 'InP - Palik',
        m: float = 0.8,
        layout_group_name="Taper",
        z=313e-9 / 2
    ):
        """
        Create a top taper structure with specified parameters and add it to the simulation environment.

        Parameters
        ----------
        name : str, optional
            Name of the taper structure, defaults to 'Taper_in'.
        height : float, optional
            Height of the taper structure, defaults to 313e-9.
        length : float, optional
            Length of the taper structure, defaults to 19e-6.
        width_in : float, optional
            Input width of the taper structure, defaults to 550e-9.
        width_out : float, optional
            Output width of the taper structure, defaults to 50e-9.
        material : str, optional
            Material of the taper structure, defaults to 'InP - Palik'.
        m : float, optional
            Exponent for the taper profile, defaults to 0.8.
        layout_group_name : str, optional
            Layout group name, defaults to "Taper".
        z : float, optional
            Z-coordinate of the center of the taper, defaults to 313e-9 / 2.
        """
        env = self.env

        # Script for creating the taper
        script_taper_in = ''' 
        res = 5000; # resolution of polygon
        xspan = linspace(-len/2, len/2, res);
        a = (w1/2 - w2/2) / len^m;
        yspan = a * (len * 0.5 - xspan)^m + w2/2;
        
        V = matrix(2 * res, 2);
        # [x, y] points
        V(1:2 * res, 1) = [xspan, flip(xspan, 1)]; 
        V(1:2 * res, 2) = [-yspan, flip(yspan, 1)];
        
        addpoly;
        set("name", "taper");
        set("x", 0);
        set("y", 0);
        set("z", 0);
        set("z span", h1);
        set("vertices", V);
        set("material", mat);
        '''

        # Create taper structure group
        env.addstructuregroup()
        env.set('name', name)
        env.set('construction group', 1)
        env.addtogroup(layout_group_name)

        # Add user properties for the taper
        env.adduserprop('len', 2, length)
        env.adduserprop('h1', 2, height)
        env.adduserprop('w1', 2, width_in)
        env.adduserprop('w2', 2, width_out)
        env.adduserprop('m', 0, m)
        env.adduserprop('mat', 1, material)

        # Set the script and position for the taper
        env.set('script', script_taper_in)
        env.set("z", z)

        # Go back to the model group
        env.groupscope('::model')

    def taper_bottom(
        self,
        name='Taper_out',
        height: float = 0.35e-6,
        length: float = 19e-6,
        width_in: float = 550e-9,
        width_out: float = 1.1e-6,
        m: float = 7,
        material: str = 'Si3N4 (Silicon Nitride) - Phillip',
        layout_group_name="Taper",
        z=-350e-9 / 2
    ):
        """
        Create a bottom taper structure with specified parameters and add it to the simulation environment.

        Parameters
        ----------
        name : str, optional
            Name of the taper structure, defaults to 'Taper_out'.
        height : float, optional
            Height of the taper structure, defaults to 0.35e-6.
        length : float, optional
            Length of the taper structure, defaults to 19e-6.
        width_in : float, optional
            Input width of the taper structure, defaults to 550e-9.
        width_out : float, optional
            Output width of the taper structure, defaults to 1.1e-6.
        m : float, optional
            Exponent for the taper profile, defaults to 7.
        material : str, optional
            Material of the taper structure, defaults to 'Si3N4 (Silicon Nitride) - Phillip'.
        layout_group_name : str, optional
            Layout group name, defaults to "Taper".
        z : float, optional
            Z-coordinate of the center of the taper, defaults to -350e-9 / 2.
        """
        env = self.env

        # Script for creating the taper
        script_taper_out = '''        
        res = 5000; # resolution of polygon
        xspan = linspace(-len/2, len/2, res);
        a = (w1/2 - w2/2) / len^m;
        yspan = a * (len * 0.5 - xspan)^m + w2/2;
        
        V = matrix(2 * res, 2);
        # [x, y] points
        V(1:2 * res, 1) = [xspan, flip(xspan, 1)]; 
        V(1:2 * res, 2) = [-yspan, flip(yspan, 1)];
        
        addpoly;
        set("name", "taper");
        set("x", 0);
        set("y", 0);
        set("z", 0);
        set("z span", h1);
        set("vertices", V);
        set("material", mat);
        '''

        # Create taper structure group
        env.addstructuregroup()
        env.set('name', name)
        env.set('construction group', 1)
        env.addtogroup(layout_group_name)

        # Add user properties for the taper
        env.adduserprop('len', 2, length)
        env.adduserprop('h1', 2, height)
        env.adduserprop('w1', 2, width_in)
        env.adduserprop('w2', 2, width_out)
        env.adduserprop('m', 0, m)
        env.adduserprop('mat', 1, material)

        # Set the script and position for the taper
        env.set('script', script_taper_out)
        env.set("z", z)

        # Go back to the model group
        env.groupscope('::model')
class LumSimulation(LumObj):
    """
    LumSimulation class for handling simulation-specific actions in the environment.
    Inherits from LumObj.

    This class provides methods to add simulation regions, meshes, and monitors, as well as running simulations.

    Parameters
    ----------
    env : lumapi.MODE or similar
        The simulation environment object.
    """

    def add_simulation_region(self):
        """
        Add a simulation region to the environment.

        This is a placeholder method that should be implemented in derived classes.
        """
        pass

    def add_mesh(self):
        """
        Add a mesh to the simulation environment.

        This is a placeholder method that should be implemented in derived classes.
        """
        pass

    def add_monitors(self):
        """
        Add monitors to the simulation environment.

        This is a placeholder method that should be implemented in derived classes.
        """
        pass

    def run_simulation(self):
        """
        Run the simulation in the environment.

        This is a placeholder method that should be implemented in derived classes.
        """
        pass


class WaveguideModeFinder(LumSimulation):
    """
    WaveguideModeFinder class for setting up and running waveguide mode simulations.
    Inherits from LumSimulation.

    This class provides methods for defining a simulation region, setting up a mesh, and extracting mode-finding results from the environment.

    Parameters
    ----------
    env : lumapi.MODE or similar
        The simulation environment object.

    
    """

    def add_simulation_region(
            self,
            width_simulation: float = 4e-6,
            height_simulation: float = 4e-6,
            x: float = 0,
            y: float = 0,
            z: float = 0,
            lambda_0: float = 1550e-9,
            material_background: str = "SiO2 (Glass) - Palik",
            boundary_condition: str = "PML", 
            number_of_trial_modes: int = 10
    ):
        """
        Add a simulation region to the environment for the mode finder.

        Parameters
        ----------
        width_simulation : float, optional
            Width of the simulation region, defaults to 4e-6.
        height_simulation : float, optional
            Height of the simulation region, defaults to 4e-6.
        x : float, optional
            X-coordinate of the simulation region, defaults to 0.
        y : float, optional
            Y-coordinate of the simulation region, defaults to 0.
        z : float, optional
            Z-coordinate of the simulation region, defaults to 0.
        lambda_0 : float, optional
            Operating wavelength, defaults to 1550e-9.
        material_background : str, optional
            Background material, defaults to "SiO2 (Glass) - Palik".
        boundary_condition : str, optional
            Boundary condition type (e.g., "PML"), defaults to "PML".
        number_of_trial_modes : int, optional
            Number of trial modes to compute, defaults to 10.
        """
        self.env.addfde()
        configuration_simulation = (
            ("FDE",    (("solver type", "2D X normal"),
                        ("x", x),
                        ("y", y),
                        ("z", z),
                        ("y span", width_simulation),
                        ("z span", height_simulation), 
                        ("wavelength", lambda_0),
                        ("y min bc", boundary_condition),
                        ("y max bc", boundary_condition),
                        ("z min bc", boundary_condition),
                        ("z max bc", boundary_condition), 
                        ("number of trial modes", number_of_trial_modes))),                   
        )

        self.add_to_configuration(configuration_simulation)
        self.update_configuration()
        if material_background is not None: 
            self.env.setnamed("FDE", "background material", material_background)

    def add_mesh(
            self, 
            width_mesh: float = 2e-6,
            height_mesh: float = 2e-6,
            length_mesh: float = 0.0,
            x : float = 0,
            y : float = 0,
            z : float = 0,
            dx : float = 10e-9,
            dy : float = 10e-9,
            dz : float = 10e-9,
            N: int = None
    ):
        """
        Add a mesh to the simulation region.

        Parameters
        ----------
        width_mesh : float, optional
            Width of the mesh, defaults to 2e-6.
        height_mesh : float, optional
            Height of the mesh, defaults to 2e-6.
        length_mesh : float, optional
            Length of the mesh, defaults to 0.0.
        x : float, optional
            X-coordinate of the mesh center, defaults to 0.
        y : float, optional
            Y-coordinate of the mesh center, defaults to 0.
        z : float, optional
            Z-coordinate of the mesh center, defaults to 0.
        dx : float, optional
            Mesh resolution in the x-direction, defaults to 10e-9.
        dy : float, optional
            Mesh resolution in the y-direction, defaults to 10e-9.
        dz : float, optional
            Mesh resolution in the z-direction, defaults to 10e-9.
        N : int, optional
            Number of divisions in the mesh. If provided, dx, dy, and dz will be automatically computed.
        """
        self.env.addmesh()

        if N is not None:
            dx = length_mesh / N
            dy = width_mesh / N
            dz = height_mesh / N
        
        configuration_mesh = (
            ("mesh", (("x", x),
                        ("y", y),
                        ("z", z),
                        ("x span", length_mesh),
                        ("y span", width_mesh),
                        ("z span", height_mesh), 
                        ("dx", dx),
                        ("dy", dy), 
                        ("dz", dz), 
                        )),
        )
        self.add_to_configuration(configuration_mesh)
        self.update_configuration()

    def run_simulation(self):
        """
        Run the waveguide mode simulation in the environment.
        """
        pass
            


    def get_sim_result(self):
        """
        Retrieve the simulation results including effective refractive index and TE polarization fraction.

        This method extracts the effective refractive index (neff) and TE polarization fraction for each mode found
        in the simulation.

        Returns
        -------
        tuple of list of float
            Two lists - one containing the effective refractive indices and one containing the TE polarization fractions.
        """
        number_of_found_modes = int(self.env.findmodes())
        n_eff_result = []
        te_fraction_result = []
        for i in range(number_of_found_modes):
            mode = f"FDE::data::mode{i+1}"
            n_eff_result.append(self.env.getdata(mode, "neff"))
            te_fraction_result.append(self.env.getdata(mode, "TE polarization fraction"))
        return n_eff_result, te_fraction_result
