import numpy as np
import libpysal as ps
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json
import os
import funcs

# Constants
filepath = os.path.join(os.getcwd(), "data.json")

with open(filepath) as f:
    data = json.load(f)

all_path = data["base_directory"]

outputs_path = os.path.join(all_path, "outputs")

combined_output=os.path.join(outputs_path,  "combined.csv")
results_output=os.path.join(outputs_path,  "results.csv")

if funcs.check_size(combined_output):
    funcs.truncate_file(results_output)
    funcs.exit_program()

data = pd.read_csv(combined_output)
data['ndvi'] = data['ndvi'] * 0.0001

rows=data.shape[0]


data.replace(np.nan, 0, inplace=True)

gdf = gp.GeoDataFrame(
    data, geometry=gp.points_from_xy(data.lon, data.lat), crs="EPSG:4326"
)

y = gdf['ndvi'].values.reshape((-1,1))
X = gdf[['sm','preci']].values
u = gdf['lon']
v = gdf['lat']
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
results = model.predict(coords_test, X_test)

data['pred'] = results.predictions

data['biom'] = ((6480.2 * data['pred']) - 958.6)/1000

data.to_csv(results_output)