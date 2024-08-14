import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np

class LumObj:
    def __init__(self, env):
        self._env = env
        self._configuration = ()

    def update_configuration(self):

        for obj, parameters in self.configuration:
            for k, v in parameters:
                self.env.setnamed(obj, k, v)


    def add_to_configuration(self, new_element_configuration):
        self.configuration = self.configuration + new_element_configuration
        
        

    @property
    def env(self):
        return self._env
    
    @property 
    def configuration(self):
        return self._configuration
    
    @configuration.setter
    def configuration(self, new_configuration):
        self._configuration = new_configuration

     
class GeomBuilder(LumObj):
         
    def input_wg(
        self,
        name_top    = "InP_input_wg",
        name_bottom = "SiN_input_wg", 
        material_top        = "InP - Palik",
        material_bottom     = "Si3N4 (Silicon Nitride) - Phillip",
        material_background = "SiO2 (Glass) - Palik", 
        width_top     : float = 550e-9,
        width_bottom  : float = 550e-9,
        height_top    : float = 313e-9, 
        height_bottom : float = 350e-9,
        length_top    : float = 10e-6, 
        length_bottom : float = 10e-6,
        x_ : float = 0,
        y_ : float = 0,
        z_ : float = 0,
        x_min_ : float = None,
        y_min_ : float = None,
        z_min_ : float = None,
        layout_group_name = "Input Waveguide"
    ): 
        
        if x_min_ is not None: x_ = x_min_/2
        if y_min_ is not None: y_ = y_min_/2
        if z_min_ is not None: z_ = z_min_/2
        
        env = self.env

        
        #Top Waveguide
        env.addrect()
        env.set("name", name_top)
        env.addtogroup(layout_group_name)
        #Bottom Waveguide
        env.addrect()
        env.set("name", name_bottom)
        env.addtogroup(layout_group_name)
        
        configuration_geometry = (
        (name_top,    (("material", material_top),
                       ("x", x_),
                       ("y", y_),
                       ("z min", z_),                       
                       ("x span", length_top),
                       ("y span", width_top),
                       ("z max",  z_ + height_top))),
                       

        (name_bottom, (("material", material_bottom),
                       ("x", x_),
                       ("y", y_),
                       ("z min", z_ -height_bottom),
                       ("x span", length_bottom),
                       ("y span", width_bottom),
                       ("z max", z_))),
        )
        
        env.groupscope(f"::model::{layout_group_name}")
        
        self.configuration = configuration_geometry
        self.update_configuration()
        
        env.groupscope('::model')

    def output_wg(
        self,
        name = 'Output_wg',
        material = 'Si3N4 (Silicon Nitride) - Phillip',
        width = 1100e-9,
        height = 350e-9,
        length = 2e-6,
        x_ = 0,
        y_ = 0,
        z_ = 0,
        x_min_ = None, #44e-6
        y_min_ = None,
        z_min_ = None,  #-300e-9
        layout_group_name = 'Output Waveguide'
    ):
        env = self.env

        if x_min_ is not None: x_ = x_min_/2
        if y_min_ is not None: y_ = y_min_/2
        if z_min_ is not None: z_ = z_min_/2


        #Waveguide
        env.addrect()
        env.set('name',name)
        env.addtogroup(layout_group_name)

        configuration_geometry =   (
        (name,   (('material',material),
                ('x',x_),
                ('y',y_),
                ('z',z_),
                ('y span',width),
                ('z span',height),
                ('x span',length))),
        )

        env.groupscope(f'::model::{layout_group_name}')

        self.configuration = configuration_geometry
        self.update_configuration()
    
        env.groupscope('::model')

    def taper_top(
        self,
        name = 'Taper_in',
        height: float = 313e-9,
        length: float = 19e-6,
        width_in: float = 550e-9,
        width_out: float = 50e-9,
        material: str = 'InP - Palik',
        m: float = 0.8,
        layout_group_name = "Taper",
        z_ = 313e-9/2
    ):
        env = self.env


        #script
        script_taper_in = ''' 

    res = 5000; #resolution of polygon
    xspan = linspace(-len/2,len/2,res);
    a = (w1/2 - w2/2)/len^m;
    yspan = a * (len*0.5 - xspan)^m + w2/2;
    
    V = matrix(2*res,2);
    #[x,y] points
    V(1:2*res,1) = [xspan, flip(xspan,1)]; 
    V(1:2*res,2) = [-yspan , flip(yspan,1)];
    
    addpoly;
    set("name", "taper");
    set("x",0);
    set("y",0);
    set("z",0);
    set("z span",h1);
    set("vertices",V);
    set("material", mat);

'''
        print(f"script variable is of type {type(script_taper_in)}\n Script is: {script_taper_in}")
        

        #Taper
        env.addstructuregroup()
        env.set('name',name)
        env.set('construction group', 1)
        env.addtogroup(layout_group_name)

        env.adduserprop('len',2,length)
        env.adduserprop('h1',2,height)
        env.adduserprop('w1',2,width_in)
        env.adduserprop('w2',2,width_out)
        env.adduserprop('m',0,m)
        env.adduserprop('mat',1,material)


        env.set('script',script_taper_in)
        #env.groupscope('::model::Taper')
        env.set("z", z_)
    
    def taper_bottom(
        self,
        name = 'Taper_out',
        height: float = 0.35e-6,
        length: float = 19e-6,
        width_in: float = 550e-9,
        width_out: float = 1.1e-6,
        m: float = 7,
        material: str = 'Si3N4 (Silicon Nitride) - Phillip',
        layout_group_name = "Taper",
        z_ = -350e-9/2
    ):
        env = self.env

        #Script
        script_taper_out = '''        
    res = 5000; #resolution of polygon
    xspan = linspace(-len/2,len/2,res);
    a = (w1/2 - w2/2)/len^m;
    yspan = a * (len*0.5 - xspan)^m + w2/2;
    
    V = matrix(2*res,2);
    #[x,y] points
    V(1:2*res,1) = [xspan, flip(xspan,1)]; 
    V(1:2*res,2) = [-yspan , flip(yspan,1)];
    
    addpoly;
    set("name", "taper");
    set("x",0);
    set("y",0);
    set("z",0);
    set("z span",h1);
    set("vertices",V);
    set("material", mat);
   

    '''
        
        #Taper group
        env.addstructuregroup()
        env.set('name',name)
        env.set('construction group', 1)
        env.addtogroup(layout_group_name)

        #Taper 
        env.adduserprop('len',2,length)
        env.adduserprop('h1',2,height)
        env.adduserprop('w1',2,width_in)
        env.adduserprop('w2',2,width_out)
        env.adduserprop('m',0,m)
        env.adduserprop('mat',1,material)


        env.set('script',script_taper_out)
        env.set("z", z_)
    
    
    @property
    def env(self):
        return self._env  

    

class LumSimulation(LumObj):
    def add_simulation_region():
        pass

    def add_mesh():
        pass
    
    def add_monitors():
        pass

    def run_simulation():
        pass
    
class WaveguideModeFinder(LumSimulation):

    def add_simulation_region(
            self,
            width_simulation: float = 4e-6,
            height_simulation: float = 4e-6,
            x_: float = 0,
            y_: float = 0,
            z_: float = 0,
            lambda_0: float = 1550e-9,
            material_background: str = "SiO2 (Glass) - Palik",
            boundary_condition: str = "PML", 
            number_of_trial_modes: int = 10
    ):
        self.env.addfde()
        configuration_simulation = (
            ("FDE",    (("solver type", "2D X normal"),
                        ("x", x_),
                        ("y", y_),
                        ("z", z_),
                        ("y span", width_simulation),
                        ("z span", height_simulation), 
                        ("wavelength", lambda_0),
                        ("background material", material_background),
                        ("y min bc", boundary_condition),
                        ("y max bc", boundary_condition),
                        ("z min bc", boundary_condition),
                        ("z max bc", boundary_condition), 
                        ("number of trial modes", number_of_trial_modes))),                   
        )

        self.add_to_configuration(configuration_simulation)
        self.update_configuration()


    def add_mesh(
            self, 
            width_mesh: float = 2e-6,
            height_mesh: float = 2e-6,
            length_mesh: float = 0.0,
            x_ : float = 0,
            y_ : float = 0,
            z_ : float = 0,
            dx_ : float = 10e-9,
            dy_ : float = 10e-9,
            dz_ : float = 10e-9,
            N: int = None
    ):
        
        self.env.addmesh()

        if N is not None:
            dx_ = length_mesh/N
            dy_ = width_mesh/N
            dz_ = height_mesh/N
        
        configuration_mesh = (
            ("mesh", (("x", x_),
                      ("y", y_),
                      ("z", z_),
                      ("x span", length_mesh),
                      ("y span", width_mesh),
                      ("z span", height_mesh), 
                      ("dx", dx_),
                      ("dy", dy_), 
                      ("dz", dz_), 
                      )),
        )
        self.add_to_configuration(configuration_mesh)
        self.update_configuration()

    def run_simulation(self):
        pass
            

    def get_sim_result(self):
        number_of_found_modes = int(self.env.findmodes())
        n_eff_result = []
        te_fraction_result = []
        for i in range(number_of_found_modes):
            mode = f"FDE::data::mode{i+1}"
            n_eff_result.append(self.env.getdata(mode, "neff"))
            te_fraction_result.append(self.env.getdata(mode, "TE polarization fraction"))
        return n_eff_result, te_fraction_result
        

    

def taper_geometry(
        env=lumapi.MODE(),
        width_in = 550e-9,
        width_out = 1100e-9,
        width_tip = 50e-9, 
        m_in = 0.8,
        m_out = 7,
        length_taper = 19e-6
    ):
    #print(f"Starting with width: {width}")
    geom = GeomBuilder(env)

    #realize groups
    groups = [
        "Input Waveguide",
        "Taper",
        "Output Waveguide",
    ]
    for group_name in groups:
        env.addgroup()
        env.set("name", group_name)

    #choose lenght
    length_input  =  10e-6

    length_output =  10e-6

    height_SiN = 350e-9
    height_InP = 313e-9



    geom.input_wg(layout_group_name="Input Waveguide", 
                  length_top=length_input, length_bottom=length_input, 
                  width_bottom=width_in, width_top=width_in
                  )
    geom.taper_in(layout_group_name="Taper", 
                  length=length_taper, 
                  width_in=width_in, width_out=width_tip,
                  m= m_in
                  )
    geom.taper_out(layout_group_name="Taper", 
                   length=length_taper, 
                   width_in=width_in, width_out=width_out,
                   m = m_out
                   )
    geom.output_wg(layout_group_name="Output Waveguide", 
                   length=length_output,
                   width=width_out
                   )



    #move the components
    centers_x = [
        length_input/2,
        length_input+length_taper/2,
        length_input+length_taper+length_output/2
    ]

    for group_name, position_x in zip(groups, centers_x):  
        env.setnamed(group_name, "x",  position_x)
    env.setnamed("Output Waveguide", "z", -height_SiN/2 )


# %%
