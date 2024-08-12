import lumapi
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
        name_top    = "InP_wg",
        name_bottom = "SiN_wg", 
        material_top        = "InP - Palik",
        material_bottom     = "Si3N4 (Silicon Nitride) - Phillip",
        material_background = "SiO2 (Glass) - Palik", 
        width_top     = 500e-9,
        width_bottom  = 500e-9,
        height_top    = 313e-9, 
        height_bottom = 350e-9,
        length_top    = 10e-6, 
        length_bottom = 10e-6,
    ): 
          
        env = self.env
  
        #Input wg group
        env.addgroup()
        env.set("name", "Input Waveguides")
        
        #Top Waveguide
        env.addrect()
        env.set("name", name_top)
        env.addtogroup("Input Waveguides")
        #Bottom Waveguide
        env.addrect()
        env.set("name", name_bottom)
        env.addtogroup("Input Waveguides")
        
        configuration_geometry = (
        (name_top,    (("material", material_top),
                       ("x", 0),
                       ("y", 0),
                       ("z min", 0),
                       ("z max", height_top),                       
                       ("x span", length_top),
                       ("y span", width_top),
                       ("z span", height_top))),
                       

        (name_bottom, (("material", material_bottom),
                       ("x", 0),
                       ("y", 0),
                       ("z min", -height_bottom),
                       ("z max", 0), 
                       ("x span", length_bottom),
                       ("y span", width_bottom),
                       ("z span", height_bottom))),
        )
        
        env.groupscope("::model::Input Waveguides");
        
        self.configuration = configuration_geometry
        self.update_configuration()
        
    
            
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
        

    

        
#----------------------TEST----------------------------
if __name__ == "__main__": 
    #%%
    from waveguides_simulations import GeomBuilder
    from waveguides_simulations import WaveguideModeFinder
    import lumapi
    import numpy as np
    import time
    import pickle
    import os
    import matplotlib.pyplot as plt


    width = 500e-6
    SIM_NAME = "input_waveguide"
    try:
        os.mkdir("dataset_"+SIM_NAME)
    except:
        pass

    # %%
    width_start = 100
    width_stop  = 1000
    width_step = 50
    width_array = np.arange(width_start, width_stop+width_step, width_step)*1e-9
    print(width_array)
    
    
    #%%
    from analysis_wg import Analysis_wg
    PATH = "sim_input_wg"
    found_modes = []
    i=0
    #%%

    for i,width in enumerate(width_array):
        
        env = lumapi.MODE()  

        layoutmode_ = env.layoutmode()
        if layoutmode_ == 0:
            env.eval("switchtolayout;")
        env.deleteall()
        print(f"Starting with width: {width}")
        geom = GeomBuilder(env)
        geom.input_wg(width_bottom = width, width_top = width)
    
        
        env.groupscope("::model")

        sim = WaveguideModeFinder(env)
        sim.add_simulation_region(width_simulation = 4*width, number_of_trial_modes=20)
        sim.add_mesh(width_mesh= 2*width, N=300)

        print("Saving model before simulation...")
        env.save(PATH + "/input_modes_"+str(i))
        print("Simulating...")

        env.run()
        found_modes_ = int(env.findmodes())
        print("Simulation finished!")
        print(f"{found_modes_} modes found")
        found_modes.append(found_modes_)
        print(f"I now found {found_modes_} modes")
        print(f"In total I have {found_modes} modes")

        print("Saving model after simulation...")
        env.save("input_modes_"+str(i))
        env.close()
    
    #%%
    import pickle
    import os
    import lumapi
    import numpy as np

    PATH = "sim_input_wg"

    width_start = 100
    width_stop  = 1000
    width_step = 50
    width_array = np.arange(width_start, width_stop+width_step, width_step)*1e-9
    print(width_array)

    data_array = []

    env = lumapi.MODE()
    env.load(PATH+"/input_modes_0")

    for i,width in enumerate(width_array):

        data = []
        for j in range(1, found_modes[i]+1):
            
            env.load("input_modes_"+str(i))
            mode = "FDE::data::mode" + str(j)
            try:       
                extracted_data = (Analysis_wg.extract_data(env, mode))
            except:
                extracted_data = ({})
            finally:
                extracted_data["width"] = width
                data.append(extracted_data)
        data_array.append(data)
        print(f"width {width} data collected")

    DATAFILE_PATH = 'input_wg_data.pickle'
    if os.path.exists(DATAFILE_PATH):
        os.remove(DATAFILE_PATH)
    with open(DATAFILE_PATH, 'wb') as file:
        pickle.dump({"width_array": width_array, 
                     "found_modes":found_modes, 
                     "data_array": data_array}, file)



    #%%
   
            
        

    



        
    
    

   
# %%
