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
    ):
        self._env = env
        self._geom = GeomBuilder(env)


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