#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import h5py as h5
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.coords import BoundingBox
from rasterio.windows import Window
from shapely.geometry import box
import requests
import geopandas as gpd
from rasterstats import zonal_stats
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# ========= PATH & CONFIG SETUP =========
# Add parent dir to sys.path for relative imports if needed
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filepath = os.path.join(project_root, 'data.json')

with open(filepath) as f:
    data = json.load(f)

inputs_path = os.path.join(project_root, 'inputs')
outputs_path = os.path.join(project_root, 'outputs')
os.makedirs(inputs_path, exist_ok=True)
os.makedirs(outputs_path, exist_ok=True)

trends_url = data["trends_url"]
api_key = data["api_key"]

print("[CONFIG] Project root:", project_root)
print("[CONFIG] Inputs path:", inputs_path)
print("[CONFIG] Outputs path:", outputs_path)
print("[CONFIG] Trends API URL:", trends_url)
print("[CONFIG] API key length:", len(api_key))

# ======================================


class aggregate_time_series:
    """
    Zonal statistics over time with robust handling:
      - CRS fix + index-safe selection
      - Explicit NoData handling: convert sentinels to NaN and pass nodata=np.nan
      - Pass 1 strict (all_touched=False), pass 2 retry with all_touched=True if empty
      - Optional tiny-polygon buffering (half pixel)
      - Fallback: centroid sample, then 3×3 mean
      - CSV + diagnostics outputs
    """

    def __init__(
        self,
        biomass_files,
        shapefile_path,
        new,
        name_of_shapefile_column,
        *,
        base_all_touched=False,     # main pass uses this; retry flips to True if empty
        buffer_tiny_polygons=True,  # gently buffer tiny geometries
        debug=True
    ):
        self.biomass_files = biomass_files
        self.shapefile_path = shapefile_path
        self.new = new
        self.trends_url = trends_url
        self.api_key = api_key
        self.base_all_touched = base_all_touched
        self.buffer_tiny_polygons = buffer_tiny_polygons
        self.debug = debug

        # read a sample raster for CRS/bounds
        with rasterio.open(self.biomass_files[0]) as sample_raster:
            self.sample_crs = sample_raster.crs
            self.sample_bounds = sample_raster.bounds
            self.sample_transform = sample_raster.transform

        # read polygons, reproject to raster CRS, fix topology, reset index
        self.shapefile_data = gpd.read_file(shapefile_path)
        self.shapefile_data = self.shapefile_data.to_crs(self.sample_crs)
        self.shapefile_data["geometry"] = self.shapefile_data.buffer(0)
        self.shapefile_data = self.shapefile_data.reset_index(drop=True)

        # identifiers
        self.datasets = [str(v).replace('/', '-') for v in self.shapefile_data[str(name_of_shapefile_column)].tolist()]
        self.name_of_shapefile_column = name_of_shapefile_column

        # final array: [zone, date_index, (Date, Biomass)]
        self.final_array = np.empty((len(self.datasets), len(self.biomass_files), 2), dtype='float')
        self.final_array[:] = np.nan

        # holders
        self.raw_biomass = None            # ndarray with NaNs (after masking)
        self.raw_biomass_affine = None
        self._nodata_val = None
        self.date = None
        self.date_counter = None
        self.shape_counter = None

        # diagnostics
        self._diag_rows = []

        self._log(f"[INIT] polygons={len(self.shapefile_data)} rasters={len(self.biomass_files)}")
        self._log(f"[INIT] CRS={self.sample_crs}")
        self._log(f"[INIT] Bounds (raster)={self.sample_bounds}")
        self._log(f"[INIT] base_all_touched={self.base_all_touched}, buffer_tiny_polygons={self.buffer_tiny_polygons}")

    def _log(self, msg: str):
        if self.debug:
            print(msg)

    @staticmethod
    def _features_from_row(row: gpd.GeoDataFrame):
        return [json.loads(row.to_json())['features'][0]['geometry']]

    def open_files(self):
        for self.date_counter, biomass_file in enumerate(self.biomass_files):
            self.date = os.path.splitext(os.path.basename(biomass_file))[0].split('_')[-1]

            # --- Read raster as ndarray and build a robust NoData mask -> NaNs
            with rasterio.open(biomass_file) as src:
                band = src.read(1)  # ndarray
                scale  = (src.scales[0]  if src.scales  else 1.0) or 1.0
                offset = (src.offsets[0] if src.offsets else 0.0) or 0.0

                # Collect NoData sentinels
                nodata_sentinels = set()
                if src.nodata is not None:
                    nodata_sentinels.add(src.nodata)
                # Add common fill codes seen in your logs
                nodata_sentinels.update([-99999, -9999, -8888, -32768])

                mask = np.zeros(band.shape, dtype=bool)
                if nodata_sentinels:
                    mask |= np.isin(band, list(nodata_sentinels))
                # Optional guard: anything absurdly negative (domain sanity)
                mask |= (band < -1e3)

                # Apply scale/offset and convert masked to NaN
                band = band.astype("float64") * scale + offset
                band[mask] = np.nan

                self.raw_biomass = band                     # ndarray with NaNs
                self.raw_biomass_affine = src.transform
                self._nodata_val = src.nodata               # informational
                rbounds = src.bounds
                rpath = src.name

            self._log(f"[RASTER] {os.path.basename(biomass_file)} date={self.date} "
                      f"scale={scale} offset={offset} nodata={self._nodata_val} bounds={rbounds}")

            self._crop_to_shapefile(rbounds, rpath)

            if self.date_counter % 20 == 0:
                oc = os.path.join(outputs_path, "Output_check")
                os.makedirs(oc, exist_ok=True)
                np.save(os.path.join(oc, f"{self.date} is done.npy"), np.arange(0, 10))

        self._save_to_hdf()
        self.update_trends()  # enable when ready

    def _clip_valid_pixel_count(self, raster_path, crop_boundaries):
        """Count unmasked pixels in a clip; useful for diagnosing count=0."""
        try:
            with rasterio.open(raster_path) as ds:
                clipped, _ = rio_mask(
                    dataset=ds,
                    shapes=crop_boundaries,
                    crop=False,
                    filled=False  # MaskedArray
                )
            return int(np.count_nonzero(~clipped.mask[0]))
        except Exception as e:
            self._log(f"[CLIP-ERR] {e}")
            return None

    def _half_pixel_diag(self):
        px_w = abs(self.sample_transform.a)
        px_h = abs(self.sample_transform.e)
        return 0.5 * (px_w**2 + px_h**2) ** 0.5

    def _crop_to_shapefile(self, raster_bounds: BoundingBox, raster_path: str):
        rbox = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
        half_diag = self._half_pixel_diag()

        for shape_counter in range(len(self.datasets)):
            try:
                row = self.shapefile_data.iloc[shape_counter:shape_counter+1]
                geom = row.geometry.iloc[0]
                ds_name = self.datasets[shape_counter]

                # sanity checks
                geom_is_valid = geom.is_valid
                geom_is_empty = geom.is_empty
                intersects = geom_is_valid and (geom.intersects(rbox))

                if not intersects:
                    self._log(f"[WARN] No intersection for {ds_name} with raster extent on {self.date}")

                # Optional: gently buffer tiny polygons (in degree units for geographic CRS)
                geom_to_use = geom
                used_buffer = False
                if self.buffer_tiny_polygons:
                    px_area = abs(self.sample_transform.a * self.sample_transform.e) or 1e-12
                    if geom.area < (0.25 * abs(px_area)):
                        geom_to_use = geom.buffer(half_diag)
                        used_buffer = True

                # build features
                row2 = row.copy()
                row2.geometry = [geom_to_use]
                crop_boundaries = self._features_from_row(row2)

                # ===== PASS 1: strict (center inclusion), explicit nodata=np.nan
                zs = zonal_stats(
                    crop_boundaries,
                    self.raw_biomass,                  # ndarray with NaNs
                    affine=self.raw_biomass_affine,
                    stats=["mean", "count"],
                    nodata=np.nan,                     # EXPLICIT
                    all_touched=self.base_all_touched  # typically False
                )
                mean_biomass = zs[0]["mean"]
                pix_count    = zs[0]["count"]
                used_retry_all_touched = False
                used_fallback = False

                # ===== PASS 2: retry with all_touched=True if empty
                if (pix_count == 0 or mean_biomass is None):
                    zs2 = zonal_stats(
                        crop_boundaries,
                        self.raw_biomass,
                        affine=self.raw_biomass_affine,
                        stats=["mean", "count"],
                        nodata=np.nan,
                        all_touched=True
                    )
                    if (zs2[0]["count"] or 0) > 0 and zs2[0]["mean"] is not None:
                        mean_biomass = float(zs2[0]["mean"])
                        pix_count    = int(zs2[0]["count"])
                        used_retry_all_touched = True

                # ===== PASS 3: fallback — centroid sample, then 3×3 mean
                if (pix_count == 0 or mean_biomass is None):
                    centroid = geom_to_use.centroid
                    with rasterio.open(raster_path) as ds:
                        s = list(ds.sample([(centroid.x, centroid.y)]))[0][0]
                        val = np.nan
                        if ds.nodata is None or (s != ds.nodata):
                            scale  = (ds.scales[0]  if ds.scales  else 1.0) or 1.0
                            offset = (ds.offsets[0] if ds.offsets else 0.0) or 0.0
                            val = s * scale + offset
                        if not np.isfinite(val):
                            r, c = ds.index(centroid.x, centroid.y)
                            window = Window(c-1, r-1, 3, 3)
                            arr = ds.read(1, window=window, masked=True).astype("float64")
                            arr = arr * ((ds.scales[0] if ds.scales else 1.0) or 1.0) + \
                                  ((ds.offsets[0] if ds.offsets else 0.0) or 0.0)
                            if arr.count() > 0:
                                val = float(arr.mean())
                        if np.isfinite(val):
                            mean_biomass = float(val)
                            pix_count    = 1
                            used_fallback = True

                # Optional: directly measure valid pixels in a clip (proof)
                valid_in_clip = self._clip_valid_pixel_count(raster_path, crop_boundaries)

                # write to final array
                if not self.final_array.flags.writeable:
                    self.final_array = self.final_array.copy()
                self.final_array[shape_counter, self.date_counter, 0] = float(self.date)
                self.final_array[shape_counter, self.date_counter, 1] = (
                    np.nan if mean_biomass is None else float(mean_biomass)
                )

                # diagnostics row
                self._diag_rows.append({
                    "Dataset": ds_name,
                    "Date": self.date,
                    "Mean": (np.nan if mean_biomass is None else float(mean_biomass)),
                    "PixelCount(zonal_stats)": int(pix_count) if pix_count is not None else None,
                    "ValidInClip(mask)": valid_in_clip,
                    "GeomValid": bool(geom_is_valid),
                    "GeomEmpty": bool(geom_is_empty),
                    "IntersectsRaster": bool(intersects),
                    "BaseAllTouched": bool(self.base_all_touched),
                    "RetriedAllTouched": bool(used_retry_all_touched),
                    "UsedFallback": bool(used_fallback),
                    "BufferedTiny": bool(used_buffer)
                })

                # verbose debug if suspicious
                if (mean_biomass is None) or (pix_count == 0) or (not intersects) or (not geom_is_valid):
                    self._log(f"[DEBUG] {ds_name} {self.date} mean={mean_biomass} "
                              f"count={pix_count} valid={geom_is_valid} empty={geom_is_empty} "
                              f"intersects={intersects} valid_in_clip={valid_in_clip} "
                              f"retry_all_touched={used_retry_all_touched} fallback={used_fallback} buffered={used_buffer}")

                self.shape_counter = shape_counter

            except Exception as e:
                print(f"[ERROR] shape {shape_counter} ({self.datasets[shape_counter]}) "
                      f"date {self.date}: {e}")

    def _save_to_hdf(self):
        import pandas as pd

        file_name = os.path.splitext(os.path.basename(self.shapefile_path))[0]
        db_dir = os.path.join(outputs_path, "Databases")
        os.makedirs(db_dir, exist_ok=True)

        h5_path = os.path.join(db_dir, f"{file_name}.h5")
        csv_path = os.path.join(db_dir, f"{file_name}.csv")
        diag_csv_path = os.path.join(db_dir, f"{file_name}_diagnostics.csv")

        # === Replace NaN or None with 0 before saving ===
        self.final_array = np.nan_to_num(self.final_array, nan=0.0)

        # flat CSV
        rows = []
        for dsi, dataset in enumerate(self.datasets):
            for i in range(len(self.biomass_files)):
                date_val = self.final_array[dsi, i, 0]
                biomass_val = self.final_array[dsi, i, 1]
                rows.append({
                    "Dataset": dataset,
                    "Date": str(int(date_val)) if date_val != 0 else "",
                    "Biomass": biomass_val  # already guaranteed to be 0.0 if was NaN
                })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[SAVE] CSV file saved: {csv_path}")

        # diagnostics CSV
        if self._diag_rows:
            pd.DataFrame(self._diag_rows).to_csv(diag_csv_path, index=False)
            print(f"[SAVE] Diagnostics CSV saved: {diag_csv_path}")

        # HDF5
        if self.new:
            with h5.File(h5_path, 'w') as storage_file:
                for dataset_counter, dataset in enumerate(self.datasets):
                    storage_file.create_dataset(
                        dataset,
                        data=self.final_array[dataset_counter, :, :],
                        compression='lzf',
                        maxshape=(None, None)
                    )
                    storage_file[dataset].attrs['Column_Names'] = np.array(['Date', 'Biomass'], dtype=np.bytes_)
            print('[SAVE] New HDF5 database created:', h5_path)
        else:
            with h5.File(h5_path, 'a') as storage_file:
                for dataset_counter, dataset in enumerate(self.datasets):
                    data = storage_file[dataset]
                    dataset_length = len(data)
                    data.resize(dataset_length + len(self.final_array[0, :, 0]), axis=0)
                    data[dataset_length - 10:-10, 0] = self.final_array[dataset_counter, :, 0]
            print('[SAVE] HDF5 database updated:', h5_path)

    def update_trends(self):
        """Send data to API. Keep NaN as None in payload."""
        all_data_to_send = []
        for shape_counter, dataset in enumerate(self.datasets):
            for date_index in range(len(self.biomass_files)):
                val = self.final_array[shape_counter, date_index, 1]
                data_to_send = {
                    "extId": dataset,
                    "mean": None if (isinstance(val, float) and np.isnan(val)) else round(float(val), 6),
                    "date": datetime.strptime(
                        str(int(self.final_array[shape_counter, date_index, 0])),
                        "%Y%m%d"
                    ).strftime("%Y-%m-%d"),
                }
                all_data_to_send.append(data_to_send)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = requests.post(self.trends_url, json=all_data_to_send, headers=headers)
            if response.status_code == 201:
                print("[API] All data saved successfully.")
            elif response.status_code == 400:
                print(f"[API] Validation error: {response.json()}")
            elif response.status_code == 401:
                print("[API] Authentication failed: Invalid or missing API key.")
            else:
                print(f"[API] Unexpected response ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"[API] Error sending data to Flask API: {e}")


# ========= Example usage (adjust paths) =========
# if __name__ == "__main__":
#     biomass_files = sorted([
#         os.path.join(inputs_path, "biomass_20251016.tif"),
#         # ... add more rasters
#     ])
#     shapefile_path = os.path.join(inputs_path, "woredas.shp")
#     at = aggregate_time_series(
#         biomass_files=biomass_files,
#         shapefile_path=shapefile_path,
#         new=True,
#         name_of_shapefile_column="WOREDA_CODE",  # change to your field
#         base_all_touched=False,      # strict first; polygon-level retry uses True only when needed
#         buffer_tiny_polygons=True,   # soften tiny polygons
#         debug=True
#     )
#     at.open_files()
#     # at.update_trends()  # enable when ready
