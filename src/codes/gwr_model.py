import numpy as np
import libpysal as ps
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.funcs import *

# Constants
filepath = '../data.json'

with open(filepath) as f:
    data = json.load(f)


outputs_path = '../outputs'


combined_output=os.path.join(outputs_path,  "combined.csv")
results_output=os.path.join(outputs_path,  "results.csv")

if check_size(combined_output):
    truncate_file(results_output)
    exit_program()



def process_data(output_path):
    data = pd.read_csv(output_path)
    data['ndvi'] = data['ndvi'] * 0.0001
    rows=data.shape[0]
    data.replace(np.nan, 0, inplace=True)
    geodf = gp.GeoDataFrame(data, geometry=gp.points_from_xy(data.lon, data.lat), crs="EPSG:4326")
    return geodf,rows,data;
gdf,rows,data=process_data(combined_output);

def run_gwr_model(gd):
    y = gd['ndvi'].values.reshape((-1,1))
    X = gd[['sm','preci']].values
    u = gd['lon']
    v = gd['lat']
    coords = list(zip(u,v))
    se=rows+100000;
    np.random.seed(se)
    sample = np.random.choice(range(rows), 0)
    mask = np.ones_like(y,dtype=bool).flatten()
    mask[sample] = False
    cal_coords = np.array(coords)[mask]
    cal_y = y[mask]
    cal_X = X[mask]
    pred_coords = np.array(coords)[~mask]
    pred_y = y[~mask]
    pred_X = X[~mask]
    gwr_selector = Sel_BW(cal_coords, cal_y, cal_X,fixed=False, kernel='gaussian')
    gwr_bw = gwr_selector.search()
    index = np.arange(len(cal_y))
    test = index[-rows:]
    X_test = X[test]
    coords_test = np.array(coords)[test]
    model = GWR(cal_coords, cal_y, cal_X, bw=gwr_bw, fixed=False, kernel='gaussian')
    res = model.predict(coords_test, X_test)
    return res

results=run_gwr_model(gdf)
data['pred'] = results.predictions

data['biom'] = ((6480.2 * data['pred']) - 958.6)/1000

data.to_csv(results_output)