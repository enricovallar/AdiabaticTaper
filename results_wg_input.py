#%%
import pickle 
import os
import numpy as np 
from analysis_wg import Analysis_wg
import matplotlib.pyplot as plt

PATH = "../working_data/"
DATAFILE = 'wg_input_data.pickle'
PICS = 'wg_input_plots/'

try:
    os.mkdir(f"{PATH}{PICS}/E2/")
    print(f"Directory '{PATH}{PICS}/E2/' created successfully")
except FileExistsError:
    print(f"Directory '{PATH}{PICS}/E2/' already exists")
except OSError as error:
    print(f"Error creating directory '{PATH}{PICS}/E2/': {error}")

try:
    os.mkdir(f"{PATH}{PICS}/purcell/")
    print(f"Directory '{PATH}{PICS}/purcell/' created successfully")
except FileExistsError:
    print(f"Directory '{PATH}{PICS}/purcell/' already exists")
except OSError as error:
    print(f"Error creating directory '{PATH}{PICS}/purcell/': {error}")






with open(f"{PATH}{DATAFILE}", 'rb') as file:
    loaded_data = pickle.load(file)

width_array = loaded_data["width_array"]
found_modes = loaded_data["found_modes"]
data_array  = loaded_data["data_array"]

#%%
#--------------EFFECTIVE INDEX---------------------------------------------
title = "Effective index"
subtitle = r"""
$313\; nm$ thick $InP$ over $350\; nm$ thick $Si_3N_4$ with the same width. 
Dot color is related to TE polarization fraction, red is TE and blue is TM."
"""
Analysis_wg.plot_neff(data_array, width_array, title, subtitle)
plt.savefig(f"{PATH}{PICS}neff.png")

#%%
import pickle 
import os
import numpy as np 
from analysis_wg import Analysis_wg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PATH = "../working_data/"
DATAFILE = 'wg_input_data.pickle'
with open(f"{PATH}{DATAFILE}", 'rb') as file:
    loaded_data = pickle.load(file)

width_array = loaded_data["width_array"]
found_modes = loaded_data["found_modes"]
data_array  = loaded_data["data_array"]

height_top = 313e-9
height_bottom = 350e-9


#Function to draw contour
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
      
      
for data, width in zip(data_array, width_array):
    figure, axs = plt.subplots(2,2, constrained_layout = True)
    #plt.tight_layout(rect=[0, 0, 1, 0.85])
    
    fig_title = rf"""
        Electric field intensity in $InP$ over $Si_3N_4$
        $Width={width/1e-9:.0f}nm$  
"""
    
    figure.suptitle(fig_title, fontsize=14, fontweight="bold")
    
    i =0
    for mode, ax in zip(data[0:4], axs.flatten()):
        i+=1
        title =  f"$mode\; {i}$: $n_{{eff}} = {np.squeeze(mode['neff'].real):.2f}$, $f_{{TE}}={mode['te_fraction']*100:.0f}\%$"
        Analysis_wg.plot_field(ax,mode, title, y_span=3*width/1e-6, z_span=2*(height_top+height_bottom)/1e-6)
        draw_contour(ax,height=[height_top,height_bottom],width=width)

    plt.savefig(f"{PATH}{PICS}/E2/E2_{width/1e-9:.0f}.png")
    print(f"figure '{PATH}{PICS}/E2/E2_{width/1e-9:.0f}.png' saved")
    #plt.show()
    plt.close(figure)
#_____________________________________________________________________________________________-
#%%
    Analysis_wg.collect_purcell(data_array)
    height_top = 313e-9
    height_bottom = 350e-9


    for data, width in zip(data_array, width_array):
        figure, axs = plt.subplots(2,2, constrained_layout = True)
        #plt.tight_layout(rect=[0, 0, 1, 0.85])
        
        fig_title = f"""
            Purcell factor in $InP$ over $Si_3N_4$
            $Width={width/1e-9:.0f}nm$  
    """
        
        figure.suptitle(fig_title, fontsize=14, fontweight="bold")
        
        i =0
        for mode, ax in zip(data[0:4], axs.flatten()):
            i+=1
            title =  f"$mode\; {i}$: $n_{{eff}} = {np.squeeze(mode['neff'].real):.2f}$, $f_{{TE}}={mode['te_fraction']*100:.0f}\%$"
            Analysis_wg.plot_purcell(ax,mode, title, y_span=3*width/1e-6, z_span=2*(height_top+height_bottom)/1e-6, k="y")
        plt.savefig(f"{PATH}{PICS}/purcell/purcell_{width/1e-9:.0f}.png")
        print(f"figure '{PATH}{PICS}/purcell/purcell_{width/1e-9:.0f}.png' saved")
        #plt.show()
        plt.close(figure)
    
