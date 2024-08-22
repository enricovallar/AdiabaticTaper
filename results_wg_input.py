#%%
import pickle 
import os
import numpy as np 
from analysis_wg import Analysis_wg
import matplotlib.pyplot as plt

PATH = "../working_data/"
DATAFILE = 'wg_input_data.pickle'
PICS = 'wg_input_plots'

DPI = 300

try:
    os.mkdir('../working_data/wg_input_plots')
except:
    pass

try:
    os.mkdir(f"{PATH}{PICS}/E2")
    print(f"Directory '{PATH}{PICS}/E2/' created successfully")
except FileExistsError:
    print(f"Directory '{PATH}{PICS}/E2/' already exists")
except OSError as error:
    print(f"Error creating directory '{PATH}{PICS}/E2/': {error}")

try:
    os.mkdir(f"{PATH}{PICS}/purcell")
    print(f"Directory '{PATH}{PICS}/purcell/' created successfully")
except FileExistsError:
    print(f"Directory '{PATH}{PICS}/purcell/' already exists")
except OSError as error:
    print(f"Error creating directory '{PATH}{PICS}/purcell/': {error}")

try:
    os.mkdir(f"{PATH}{PICS}/beta")
    print(f"Directory '{PATH}{PICS}/beta/' created successfully")
except FileExistsError:
    print(f"Directory '{PATH}{PICS}/beta/' already exists")
except OSError as error:
    print(f"Error creating directory '{PATH}{PICS}/beta/': {error}")

try:
    os.mkdir(f"{PATH}{PICS}/beta_gradient")
    print(f"Directory '{PATH}{PICS}/brata_gradient/' created successfully")
except FileExistsError:
    print(f"Directory '{PATH}{PICS}/beta_gradient/' already exists")
except OSError as error:
    print(f"Error creating directory '{PATH}{PICS}/beta_gradient/': {error}")


#%%
# _____LOAD DATA_______________________________________________
with open(f"{PATH}{DATAFILE}", 'rb') as file:
    loaded_data = pickle.load(file)
    file.close()

width_array = loaded_data["width_array"]
found_modes = loaded_data["found_modes"]
data_array  = loaded_data["data_array"]
height_top = 313e-9
height_bottom = 350e-9
normalize_ = True



#%%
# ______CALCULATE BETA FACTOR_____________________________________


Analysis_wg.collect_purcell(data_array, lam0= 1550e-9)
Analysis_wg.calculate_beta_factor(data_array)
Analysis_wg.integrate_beta_in_region(data_array,
                                    y_0=0,
                                    z_0=height_top/2,
                                    y_span = 60e-9,
                                    z_span = 60e-9) 
Analysis_wg.normalize_beta_integrals(data_array)
Analysis_wg.calculate_beta_gradient(data_array, height_top/2)
Analysis_wg.normalize_beta_gradients(data_array)

#save data after calculations
with open(f"{PATH}{DATAFILE}", 'wb') as file:
    pickle.dump({"data_array": data_array,
                    "found_modes": found_modes,
                    "width_array": width_array}, file)
    file.close()



#%%
# ________PLOT EFFECTIVE INDICES___________________________________________
title = "Effective index"
subtitle = r"""
$313\; nm$ thick $InP$ over $350\; nm$ thick $Si_3N_4$ with the same width. 
Dot color is related to TE polarization fraction, red is TE and blue is TM.
Dot size is related to the $\beta$-factor.
"""
Analysis_wg.plot_neff(data_array, width_array, title, subtitle)
plt.savefig(f"{PATH}{PICS}/neff.png")


#%%___________________PLOT FIELD__________________________________________
for data, width in zip(data_array, width_array):
    figure, axs = plt.subplots(2,2, constrained_layout = True, dpi=300)
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
        Analysis_wg.draw_contour(ax,height=[height_top,height_bottom],width=width)

    plt.savefig(f"{PATH}{PICS}/E2/E2_{width/1e-9:.0f}.png", dpi=300)
    print(f"figure '{PATH}{PICS}/E2/E2_{width/1e-9:.0f}.png' saved")
    #plt.show()
    plt.close(figure)



#%%___________PLOT BETA_____________________________________________________
#______________QD EMSISSION POLARIZED IN Y DIRECTION
for data,  width in zip(data_array,  width_array):
    figure, axs = plt.subplots(2,2, constrained_layout = True, dpi = 300)
    #plt.tight_layout(rect=[0, 0, 1, 0.85])
    
    fig_title = f"""
        $\\beta$-factor in $InP$ over $Si_3N_4$. 
        QD emission polarized along y.
        $Width={width/1e-9:.0f}nm$  
"""
    
    figure.suptitle(fig_title, fontsize=14, fontweight="bold")
    
    i =0
    for mode, ax in zip(data[0:4], axs.flatten()):
        i+=1
        title =  f"$mode\; {i}$: $n_{{eff}} = {np.squeeze(mode['neff'].real):.2f}$, $f_{{TE}}={mode['te_fraction']*100:.0f}\%$"
        Analysis_wg.plot_beta(ax,mode, title, y_span=3*width/1e-6, z_span=2*(height_top+height_bottom)/1e-6, k="y", normalize= normalize_)
        Analysis_wg.draw_contour(ax,height=[height_top,height_bottom],width=width)
    plt.savefig(f"{PATH}{PICS}/beta/beta_{width/1e-9:.0f}_y.png", dpi = 300)
    print(f"figure '{PATH}{PICS}/beta/beta_{width/1e-9:.0f}_y.png' saved")
    #plt.show()
    plt.close(figure)

#______________QD EMSISSION POLARIZED IN Z DIRECTION
for data, width in zip(data_array, width_array):
    figure, axs = plt.subplots(2,2, constrained_layout = True, dpi = 300)
    #plt.tight_layout(rect=[0, 0, 1, 0.85])
    
    fig_title = f"""
        $\\beta$-factor in $InP$ over $Si_3N_4$. 
        QD emission polarized along z.
        $Width={width/1e-9:.0f}nm$   
"""
    
    figure.suptitle(fig_title, fontsize=14, fontweight="bold")
    
    i =0
    for mode, ax in zip(data[0:4], axs.flatten()):
        i+=1
        title =  f"$mode\; {i}$: $n_{{eff}} = {np.squeeze(mode['neff'].real):.2f}$, $f_{{TE}}={mode['te_fraction']*100:.0f}\%$"
        Analysis_wg.plot_beta(ax,mode, title, y_span=3*width/1e-6, z_span=2*(height_top+height_bottom)/1e-6, k="z", normalize = normalize_)
        Analysis_wg.draw_contour(ax,height=[height_top,height_bottom],width=width)
    plt.savefig(f"{PATH}{PICS}/beta/beta_{width/1e-9:.0f}_z.png", dpi = 300)
    print(f"figure '{PATH}{PICS}/beta/beta_{width/1e-9:.0f}_z.png' saved")
    #plt.show()
    plt.close(figure)
    


# %%
#________________BETA_FACTOR_GRADIENT______________________________________
for data,  width in zip(data_array,  width_array):
    figure, axs = plt.subplots(2,2, constrained_layout = True, dpi = 300, figsize=(12,10))
    
    
    fig_title = f"""
        $\\beta$-factor and its gradient in $InP$ over $Si_3N_4$. 
        $Width={width/1e-9:.0f}nm$  
"""
    
    figure.suptitle(fig_title, fontsize=12, fontweight="bold")
    
    i =0
    for mode, ax in zip(data[0:4], axs.flatten()):
        i+=1
        title =  f"$mode\; {i}$: $n_{{eff}} = {np.squeeze(mode['neff'].real):.2f}$, $f_{{TE}}={mode['te_fraction']*100:.0f}\%$"
        line1 = Analysis_wg.plot_beta_gradient(ax, mode, 
                                       z_position=height_top/2,
                                       title = title,
                                       normalize=True, 
                                       xlim = (-0.1,0.1))

        line2 = Analysis_wg.plot_beta_at_z(height_top/2, ax, mode)
        
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc=0)


    plt.savefig(f"{PATH}{PICS}/beta_gradient/gradient_{width/1e-9:.0f}.png", dpi = 300)
    print(f"figure '{PATH}{PICS}/beta_gradient/beta_gradient_{width/1e-9:.0f}.png' saved")
    #plt.show()
    plt.close(figure)
# %%
    modes, widths = Analysis_wg.find_te_modes_with_highest_neff(data_array)
    results = Analysis_wg.get_beta_at_position(modes, y0=None,z0 = 313e-9/2)
    Analysis_wg.plot_beta3D(results)

#%%  
    modes, widths = Analysis_wg.find_te_modes_with_highest_neff(data_array)
    results = Analysis_wg.get_beta_at_position(modes, y0=0 ,z0 = 313e-9/2)
    Analysis_wg.plot_beta2D(results)
# %%
