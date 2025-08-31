#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:45:06 2025

@author: leahclayton
"""

import rasterio
from rasterio import mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy.stats import kurtosis
import pandas as pd

"""
Input needs:
- shp_base: base path for fire polygon shapefiles separated into burn classifications
- base_path: base path for PDO product tifs and in situ data xlsx
- PDO product tifs
    - Source data: https://doi.org/10.3334/ORNLDAAC/2004
    - 'alt' layer saved from nc4 --> geoTIFF and named 'ykd_alt.tif'
    - 'mv_alt' layer saved from nc4 --> geoTIFF and named 'ykd_vwc_alt.tif'
- fp: in situ data xlsx
    - Source data: https://doi.org/10.3334/ORNLDAAC/1903
    - Filter to relevant geographic area
    - Separated into burned and unburned based on YKD field report notes
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,2.5), layout='constrained')
axes = [ax1, ax2]

#%% Bulk VWC
shp_base = ''
shp_burn = shp_base + '/ykd_py_burned/ykd_py_burned.shp'
shp_1k = shp_base + '/ykd_py_1k/ykd_py_1k.shp'
shp_5kdif1k = shp_base + '/ykd_py_5kdif1k/ykd_py_5kdif1k.shp'
shp_re = shp_base + '/ykd_py_reburns/ykd_py_reburns.shp'

gdf_burn = gpd.read_file(shp_burn)
gdf_1k = gpd.read_file(shp_1k)
gdf_5kdif1k = gpd.read_file(shp_5kdif1k)
gdf_re = gpd.read_file(shp_re)

# target_crs = 'EPSG:9822'
target_crs = pyproj.CRS.from_string(
    '+proj=aea +lat_1=50.0 +lat_2=70.0 +lat_0=40.0 +lon_0=-96.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m'
)

gdf_burn_reproj = gdf_burn.to_crs(target_crs)
gdf_1k_reproj = gdf_1k.to_crs(target_crs)
gdf_5kdif1k_reproj = gdf_5kdif1k.to_crs(target_crs)
gdf_re_reproj = gdf_re.to_crs(target_crs)

base_path = ''
input_file = base_path + 'ykd_vwc_alt.tif'
# Plot the raster output
with rasterio.open(input_file) as src:
    mask_burn = rasterio.mask.mask(src, gdf_burn_reproj.geometry, crop=True)
    mask_1k = rasterio.mask.mask(src, gdf_1k_reproj.geometry, crop=True)
    mask_5kdif1k = rasterio.mask.mask(src, gdf_5kdif1k_reproj.geometry, crop=True)
    mask_re = rasterio.mask.mask(src, gdf_re_reproj.geometry, crop=True) 
    
    f_burn = mask_burn[0].flatten() 
    f_burn[f_burn == -9999] = np.nan
    f_burn = f_burn[~np.isnan(f_burn)]
 
    f_1k = mask_1k[0].flatten()
    f_1k[f_1k == -9999] = np.nan
    f_1k = f_1k[~np.isnan(f_1k)]
 
    f_5kdif1k = mask_5kdif1k[0].flatten()
    f_5kdif1k[f_5kdif1k == -9999] = np.nan
    f_5kdif1k = f_5kdif1k[~np.isnan(f_5kdif1k)]
 
    f_re = mask_re[0].flatten()
    f_re[f_re == -9999] = np.nan
    f_re = f_re[~np.isnan(f_re)]
 
    f_raster = src.read(1)
    f_raster[f_raster == -9999] = np.nan
    f_raster = f_raster[~np.isnan(f_raster)]
 
    # Calculate lengths
    l_burn = len(f_burn)
    l_1k = len(f_1k)
    l_5kdif1k = len(f_5kdif1k)
    l_re = len(f_re)
    l_raster = len(f_raster)
 
    # Calculate means
    mean_burn = np.mean(f_burn)
    mean_1k = np.mean(f_1k)
    mean_5kdif1k = np.mean(f_5kdif1k)
    mean_re = np.mean(f_re)
    mean_raster = np.mean(f_raster)
    
    # Calculate medians
    med_burn = np.median(f_burn)
    med_1k = np.median(f_1k)
    med_5kdif1k = np.median(f_5kdif1k)
    med_re = np.median(f_re)
    med_raster = np.median(f_raster)
    
    # Calculate min
    min_burn = np.min(f_burn)
    min_1k = np.min(f_1k)
    min_5kdif1k = np.min(f_5kdif1k)
    min_re = np.min(f_re)
    min_raster = np.min(f_raster)
    
    # Calculate max
    max_burn = np.max(f_burn)
    max_1k = np.max(f_1k)
    max_5kdif1k = np.max(f_5kdif1k)
    max_re = np.max(f_re)
    max_raster = np.max(f_raster)
    
    perc_dif_burn = ((med_burn - mean_burn)/(mean_burn)) * 100
    perc_dif_1k = ((med_1k - mean_1k)/(mean_1k)) * 100
    
    kurt_burn = kurtosis(f_burn)
    kurt_1k = kurtosis(f_1k)
    
    print('burned mean:', mean_burn, 'median:', med_burn, 'min:', min_burn, 'max:', max_burn)
    print('burned mean/med perc dif:', perc_dif_burn)
    print('burned kurtosis:', kurt_burn)
    print('1k mean:', mean_1k, 'median:', med_1k, 'min:', min_1k, 'max:', max_1k)
    print('1k mean/med perc dif:', perc_dif_1k)
    print('1k kurtosis:', kurt_1k)
    print('5kdif1k mean', mean_5kdif1k, 'median:', med_5kdif1k, 'min:', min_5kdif1k, 'max:', max_5kdif1k)
    print('re mean:', mean_re, 'median:', med_re, 'min:', min_re, 'max:', max_re)
    print('raster mean:', mean_raster, 'median:', med_raster, 'min:', min_raster, 'max:', max_raster)

    axes[0].hist(f_raster, bins=50, color='lightgrey', alpha=0.7, density=False)
    axes[0].axvline(mean_raster, color='grey', linestyle='dotted', linewidth=2)
    axes[0].hist(f_1k, bins=50, histtype='step', color='navy', alpha=1, linewidth=3, density=False)

    axes[0].hist(f_burn, bins=50, histtype='step', color='red', linewidth=3, alpha=1, density=False)
    axes[0].axvline(mean_burn, color='red', linestyle='dotted', linewidth=2)
    axes[0].hist(f_re, bins=50, color='orange', histtype='step', linewidth =3, alpha=1, density=False)
    axes[0].axvline(mean_re, color='orange', linestyle='dotted', linewidth=2)
    axes[0].axvline(mean_1k, color='navy', linestyle='dotted', linewidth=2)
    
    def thousands(x, pos):
        return '{:.0f}'.format(x/1000)
    
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(thousands))
 
    axes[0].set_title('(a)', loc='left')
    axes[0].set_xlim(0.2, 1)
    axes[0].set_ylim(0, 5)
    axes[0].set_ylabel('Pixel Count (1000 pixels)')
    axes[0].set_xlabel('')
    axes[0].legend(fontsize='8', loc='upper right')
    axes[0].grid(True)

#%% ALT

input_file = base_path + 'ykd_alt.tif'

with rasterio.open(input_file) as src:
    mask_burn = rasterio.mask.mask(src, gdf_burn_reproj.geometry, crop=True)
    mask_1k = rasterio.mask.mask(src, gdf_1k_reproj.geometry, crop=True)
    mask_5kdif1k = rasterio.mask.mask(src, gdf_5kdif1k_reproj.geometry, crop=True)
    mask_re = rasterio.mask.mask(src, gdf_re_reproj.geometry, crop=True) 
    
    f_burn = mask_burn[0].flatten() 
    f_burn[f_burn == -9999] = np.nan
    f_burn = f_burn[~np.isnan(f_burn)]
 
    f_1k = mask_1k[0].flatten()
    f_1k[f_1k == -9999] = np.nan
    f_1k = f_1k[~np.isnan(f_1k)]
 
    f_5kdif1k = mask_5kdif1k[0].flatten()
    f_5kdif1k[f_5kdif1k == -9999] = np.nan
    f_5kdif1k = f_5kdif1k[~np.isnan(f_5kdif1k)]
 
    f_re = mask_re[0].flatten()
    f_re[f_re == -9999] = np.nan
    f_re = f_re[~np.isnan(f_re)]
 
    f_raster = src.read(1)
    f_raster[f_raster == -9999] = np.nan
    f_raster = f_raster[~np.isnan(f_raster)]
 
    # Calculate lengths
    l_burn = len(f_burn)
    l_1k = len(f_1k)
    l_5kdif1k = len(f_5kdif1k)
    l_re = len(f_re)
    l_raster = len(f_raster)
 
    # Calculate means
    mean_burn = np.mean(f_burn)
    mean_1k = np.mean(f_1k)
    mean_5kdif1k = np.mean(f_5kdif1k)
    mean_re = np.mean(f_re)
    mean_raster = np.mean(f_raster)
    
    # Calculate medians
    med_burn = np.median(f_burn)
    med_1k = np.median(f_1k)
    med_5kdif1k = np.median(f_5kdif1k)
    med_re = np.median(f_re)
    med_raster = np.median(f_raster)
    
    # Calculate min
    min_burn = np.min(f_burn)
    min_1k = np.min(f_1k)
    min_5kdif1k = np.min(f_5kdif1k)
    min_re = np.min(f_re)
    min_raster = np.min(f_raster)
    
    # Calculate max
    max_burn = np.max(f_burn)
    max_1k = np.max(f_1k)
    max_5kdif1k = np.max(f_5kdif1k)
    max_re = np.max(f_re)
    max_raster = np.max(f_raster)
    
    perc_dif_burn = ((med_burn - mean_burn)/(mean_burn)) * 100
    perc_dif_1k = ((med_1k - mean_1k)/(mean_1k)) * 100
    
    kurt_burn = kurtosis(f_burn)
    kurt_1k = kurtosis(f_1k)
    
    print('burned mean:', mean_burn, 'median:', med_burn, 'min:', min_burn, 'max:', max_burn)
    print('burned mean/med perc dif:', perc_dif_burn)
    print('burned kurtosis:', kurt_burn)
    print('1k mean:', mean_1k, 'median:', med_1k, 'min:', min_1k, 'max:', max_1k)
    print('1k mean/med perc dif:', perc_dif_1k)
    print('1k kurtosis:', kurt_1k)
    print('5kdif1k mean', mean_5kdif1k, 'median:', med_5kdif1k, 'min:', min_5kdif1k, 'max:', max_5kdif1k)
    print('re mean:', mean_re, 'median:', med_re, 'min:', min_re, 'max:', max_re)
    print('raster mean:', mean_raster, 'median:', med_raster, 'min:', min_raster, 'max:', max_raster)
 
    axes[1].hist(f_raster, bins=50, color='lightgrey', alpha=0.7, density=False)
    axes[1].axvline(mean_raster, color='grey', linestyle='dotted', linewidth=2)
    axes[1].hist(f_1k, bins=50, histtype='step', color='navy', alpha=1, linewidth=3, density=False)
    axes[1].hist(f_burn, bins=50, histtype='step', color='red', linewidth=3, alpha=1, density=False)
    axes[1].axvline(mean_burn, color='red', linestyle='dotted', linewidth=2)
    axes[1].hist(f_re, bins=50, color='orange', histtype='step', linewidth =3, alpha=1, density=False)
    axes[1].axvline(mean_re, color='orange', linestyle='dotted', linewidth=2)
    axes[1].axvline(mean_1k, color='navy', linestyle='dotted', linewidth=2)

    def thousands(x, pos):
        return '{:.0f}'.format(x/1000)
    
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(thousands))

    axes[1].set_title('(c)', loc='left')
    axes[1].set_xlim(0, 1.25)
    axes[1].set_ylim(0, 5)
    axes[1].set_ylabel('Pixel Count (1000 pixels)')
    axes[1].set_xlabel('')
    axes[1].legend(fontsize='8', loc='upper right')
    axes[1].grid(True)


#%%
save_path = base_path + 'reburn_alt_vwc_subset.png'
plt.savefig(save_path, dpi=300)
plt.show()