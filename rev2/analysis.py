import os
ROOT_DIR = os.path.abspath(os.sep)


import os
import numpy as np

import importlib.util

ROOT_DIR = os.path.abspath(os.sep)
lumerical_api_path = os.path.join(ROOT_DIR, 'appl', 'lumerical', '2024-R1.03', 'api', 'python', 'lumapi.py')

spec_win = importlib.util.spec_from_file_location('lumapi', lumerical_api_path)
lumapi = importlib.util.module_from_spec(spec_win)
spec_win.loader.exec_module(lumapi)

TAPER_FOLDER = os.path.join(ROOT_DIR, 'work3', 's232699', 'taper_new')
for subfolder in os.listdir(TAPER_FOLDER):
    print(subfolder)

FILE_FOLDER = os.path.join(TAPER_FOLDER, 'sweep_m_top_2')

files = []
for file in os.listdir(FILE_FOLDER):
    if file.endswith(".lms"):
        files.append(os.path.join(FILE_FOLDER, file))

files.sort()
files

        
import plotly.graph_objects as go
def extract_data(env, start, stop, steps,  fig=None):
    env.groupscope('::model')
    env.select("Top Waveguide::taper_top")
    m_top = env.get('m')
    print(m_top)

    env.setemeanalysis("propagation sweep", 1)
    env.setemeanalysis("parameter", "group span 2")
    env.setemeanalysis("start", start)
    env.setemeanalysis("stop", stop)
    env.setemeanalysis("number of points", steps)

    env.emesweep()

    S = env.getemesweep("S")
    print(S)
    S21 = S['s21'].squeeze()
    group_span = S['group_span_2'].squeeze()
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=group_span, y=abs(S21)**2, mode='lines' , name=f'm={m_top}'))
    fig.update_layout(
        title='Transmission vs taper length',
        xaxis_title='Taper length',
        yaxis_title='Transmission',
        width=1200,  # Set the width of the figure
        height=800,  # Set the height of the figure
        showlegend=True  # Ensure the legend is shown
    )

    data_point = {
        "m_top": m_top,
        "S21": S21,
        "group_span": group_span
    }
     
    return fig, data_point

import pickle

data = []
fig = go.Figure()
for i,file in enumerate(files):
    env = lumapi.MODE()
    env.load(file)
    fig, data_point = extract_data(env, start=10e-6,stop=40e-6, steps=40, fig=fig)
    
    fig.write_html(f"partial_{i}.html")
    
    data.append(data_point)
    # Pickle the data
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    env.close()


fig.write_html("figure.html")
