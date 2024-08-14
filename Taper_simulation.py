from waveguides_simulations import GeomBuilder
from waveguides_simulations import WaveguideModeFinder
import importlib.util
#The default paths for windows
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py')
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) # 
spec_win.loader.exec_module(lumapi)
import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as plt

#Initialisation 
env = lumapi.MODE()

#Switch to layout mode 
layoutmode_ = env.layoutmode()
if layoutmode_ == 0:
    env.eval('switchtolayout;')
env.deleteall()

#Realize device geometry
#env.eval('device_geometry;')
geom = GeomBuilder(env)
geom.input_wg()
geom.taper_in()
geom.taper_out()
geom.output_wg(x_ = 21e-6, z_ = -1.75e-7)

##Parameters

#Device Geometry parameters
#InP  parameters
h_InP = 250e-9
w1_InP = 550e-9
w2_InP = 50e-9 
len_InP = 2e-6
mat1 = "InP - Palik"

#SiN parameters
h_SiN = 350e-9 
w1_SiN = 550e-9
w2_SiN = 1.1e-6
len_SiN = 2e-6
mat2 = "Si3N4 (Silicon Nitride) - Phillip"

#Taper parameters
len_taper = 19e-6

#Simulation parameters
lam0 = 1550e-9
mul_w = 3 
mul_h = 3
w_EME = mul_w * w2_SiN
h_EME = mul_h * (h_SiN + h_InP)
mat_back = "SiO2 (Glass) - Palik"
N_modes = 50
x_pen = 4/10*len_InP #penetration of sim region inside single mode waveguides

#Mesh
#N_points = 500;
w_mesh = w2_SiN
h_mesh = (h_SiN + h_InP)
len_mesh = 2*(len_InP + len_taper + len_SiN)
#dy = w_mesh/N_points;
#dz = h_mesh/N_points;
#dx = len_mesh/N_points;
dx = 5e-9
dy = 5e-9
dz = 0.01e-6

##Simulation region
env.addeme()
env.set('wavelength', lam0)
#Position
env.set('x min', len_InP - x_pen)
env.set('y', 0)
env.set('y span', w_EME)
env.set('z', (h_InP-h_SiN)/2)
env.set('z span', h_EME)

#Boundary conditions
env.set('y min bc', 'PML')
env.set('y max bc', 'PML')
env.set('z min bc', 'PML')
env.set('z max bc', 'PML')

#Material
env.set('background material', mat_back)

#Cells
env.set('number of cell groups', 3)
env.set('group spans',np.array([x_pen, len_taper, x_pen]))

env.set('cells',np.array([1,30,1]))
env.set('subcell method',np.array([0,1,0]))   # 0 = None, 1 = CVCS

env.set('display cells', 1)
env.set('number of modes for all cell groups', N_modes)

#Set up ports: port 1
env.select('EME::Ports::port_1')
env.set('use full simulation span', 1)
env.set('y', 0)
env.set('y span', w_EME)
env.set('z', 0)
env.set('z span', h_EME)
env.set('mode selection', 'fundamental TE mode')

#Set up ports: port 2
env.select('EME::Ports::port_2')
env.set('use full simulation span', 1)
env.set('y', 0)
env.set('y span', w_EME)
env.set('z', 0)
env.set('z span', h_EME)
env.set('mode selection', 'fundamental TE mode')


##Override mesh
env.addmesh()

env.set('y', 0)
env.set('y span', w_mesh)
env.set('z', (h_InP-h_SiN)/2)
env.set('z span', h_mesh)

env.set('x', (len_InP + len_SiN + len_taper)/2)
env.set('x span', len_mesh)

env.set('dx', dx)
env.set('dy', dy)
env.set('dz', dz)


##Monitors

#--InP
#Field profile
env.addemeprofile()
env.set('name', 'monitor_field_InP')

env.set('y', 0)
env.set('y span', w_EME*1.3)

env.set('z', h_InP/2)

env.set('x', (len_InP + len_SiN + len_taper)/2)
env.set('x span', len_mesh*1.3)

#Index profile
env.addemeindex()
env.set('name', 'monitor_index_InP')

env.set('y', 0)
env.set('y span', w_EME*1.3)

env.set('z', h_InP/2)

env.set('x', (len_InP + len_SiN + len_taper)/2)
env.set('x span', len_mesh*1.3)


#--SiN
#Field profile
env.addemeprofile()
env.set('name', 'monitor_field_SiN')

env.set('y', 0)
env.set('y span', w_EME*1.3)

env.set('z', -h_SiN/2)

env.set('x', (len_InP + len_SiN + len_taper)/2)
env.set('x span', len_mesh*1.3)

#Index profile
env.addemeindex()
env.set('name', 'monitor_index_SiN')

env.set('y', 0)
env.set('y span', w_EME*1.3)

env.set('z', -h_SiN/2)

env.set('x', (len_InP + len_SiN + len_taper)/2)
env.set('x span', len_mesh*1.3)


#--y=0
#Field profile
env.addemeprofile()
env.set('name', 'monitor_field_y0')
env.set('monitor type', '2D Y-normal')

env.set('y', 0)

env.set('z', 0)
env.set('z span', h_mesh*1.3)

env.set('x', (len_InP + len_SiN + len_taper)/2)
env.set('x span', len_mesh*1.3)

#Index profile
env.addemeindex()
env.set('name', 'monitor_index_y0')
env.set('monitor type', '2D Y-normal')

env.set('y', 0)

env.set('z', 0)
env.set('z span', h_mesh*1.3)

env.set('x', (len_InP + len_SiN + len_taper)/2)
env.set('x span', len_mesh*1.3)


#Keep simulation open
while True:
    pass

