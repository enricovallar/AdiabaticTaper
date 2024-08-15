import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np
from waveguides_simulations import GeomBuilder



class TaperDesigner:
    def __init__(
            self, 
            env, 
            m_top: float = 0.8, 
            m_bottom: float = 7, 
            length_taper: float = 19e-6, 
            width_in: float = 550e-9,
            width_out: float = 1100e-9,
            width_tip: float = 50e-9, 
            length_input: float  = 10e-6,
            length_output: float = 10e-6,
            height_top: float = 313e-9,
            height_bottom: float = 350e-9, 
            mul_w: float = 3,
            mul_h: float = 3
    ):
        self._m_top = m_top 
        self._m_bottom = m_bottom
        self._length_taper = length_taper
        self._width_in = width_in
        self._width_out = width_out
        self._width_tip = width_tip
        self._length_input = length_input
        self._length_output = length_output
        self._height_top = height_top
        self._height_bottom = height_bottom
        self._env = env
        self._geom = GeomBuilder(self._env)
        

        groups = [
            "Input Waveguide",
            "Taper",
            "Output Waveguide",
        ]

        for group_name in groups:
            self._env.addgroup()
            self._env.set("name", group_name)
        
        self._geom.input_wg(layout_group_name="Input Waveguide", 
                            length_top=length_input, length_bottom=length_input, 
                            width_bottom=width_in, width_top=width_in,
                            height_top=height_top,
                            height_bottom=height_bottom
                            )
        self._geom.taper_top(layout_group_name="Taper", 
                            length=length_taper, 
                            width_in=width_in, width_out=width_tip,
                            m= m_top,
                            height = height_top
                            )
        self._geom.taper_bottom(layout_group_name="Taper", 
                            length=length_taper, 
                            width_in=width_in, width_out=width_out,
                            m = m_bottom, 
                            height = height_bottom
                            )
        self._geom.output_wg(layout_group_name="Output Waveguide", 
                            length=length_output,
                            width=width_out, 
                            height=height_bottom
                            )
        
        #move the components
        centers_x = [
            length_input/2,
            length_input+length_taper/2,
            length_input+length_taper+length_output/2
        ]
        
        for group_name, position_x in zip(groups, centers_x):  
            self._env.setnamed(group_name, "x",  position_x)
        self._env.setnamed("Output Waveguide", "z", -height_bottom/2 )

        self.build_simulation_region(mul_w=mul_w, mul_h=mul_h)
        
        self.build_mesh()
    
    def build_simulation_region(self,
                                mul_w = 3,
                                mul_h = 3
                                ):
        #Simulation Parameters
        lam0 = 1550e-9
        mul_w = mul_w 
        mul_h = mul_h
        w_EME = mul_w * self._width_out
        h_EME = mul_h * (self._height_top + self._height_bottom)
        mat_back = "SiO2 (Glass) - Palik"
        N_modes = 50
        x_pen = 4/10*self._length_input #penetration of sim region inside single mode waveguides

        ##Simulation region
        self._env.addeme()
        self._env.set('wavelength', lam0)
        #Position
        self._env.set('x min', self._length_input - x_pen)
        self._env.set('y', 0)
        self._env.set('y span', w_EME)
        self._env.set('z', (self._height_top-self._height_bottom)/2)
        self._env.set('z span', h_EME)

        #Boundary conditions
        self._env.set('y min bc', 'PML')
        self._env.set('y max bc', 'PML')
        self._env.set('z min bc', 'PML')
        self._env.set('z max bc', 'PML')

        #Material
        self._env.set('background material', mat_back)

        #Cells
        self._env.set('number of cell groups', 3)
        self._env.set('group spans',np.array([x_pen, self._length_taper, x_pen]))

        self._env.set('cells',np.array([1,30,1]))
        self._env.set('subcell method',np.array([0,1,0]))   # 0 = None, 1 = CVCS

        self._env.set('display cells', 1)
        self._env.set('number of modes for all cell groups', N_modes)

        #Set up ports: port 1
        self._env.select('EME::Ports::port_1')
        self._env.set('use full simulation span', 1)
        self._env.set('y', 0)
        self._env.set('y span', w_EME)
        self._env.set('z', 0)
        self._env.set('z span', h_EME)
        self._env.set('mode selection', 'fundamental TE mode')

        #Set up ports: port 2
        self._env.select('EME::Ports::port_2')
        self._env.set('use full simulation span', 1)
        self._env.set('y', 0)
        self._env.set('y span', w_EME)
        self._env.set('z', 0)
        self._env.set('z span', h_EME)
        self._env.set('mode selection', 'fundamental TE mode')

    def build_mesh(self):
        #Mesh parameters
        #N_points = 500;
        w_mesh = 1.5*self._width_out
        h_mesh = 1.5*(self._height_top + self._height_bottom)
        len_mesh = (self._length_input + self._length_taper + self._length_output)
        #dy = w_mesh/N_points;
        #dz = h_mesh/N_points;
        #dx = len_mesh/N_points;
        dx = 5e-9
        dy = 5e-9
        dz = 0.01e-6

        ##Override mesh
        self._env.addmesh()

        self._env.set('y', 0)
        self._env.set('y span', w_mesh)
        self._env.set('z', (self._height_top-self._height_bottom)/2)
        self._env.set('z span', h_mesh)

        self._env.set('x', (self._length_input + self._length_taper + self._length_output)/2)
        self._env.set('x span', len_mesh)

        self._env.set('dx', dx)
        self._env.set('dy', dy)
        self._env.set('dz', dz)

if __name__ == "__main__":
    #%%
    import importlib.util
    #The default paths for windows
    spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
    #Functions that perform the actual loading
    lumapi = importlib.util.module_from_spec(spec_win) # 
    spec_win.loader.exec_module(lumapi)
    from taperDesigner import TaperDesigner
    

    env = lumapi.MODE()
    taper = TaperDesigner(env)

    #%%
    while True:
        pass