#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:44:58 2021

@author: yungyun
"""


#%% plot
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio


#%% rain setting
nws_precip_colors = [
    "#fdfdfd",  # 0.01 - 0.10 inches
    "#c9c9c9",  # 0.10 - 0.25 inches
    "#9dfeff",
    "#01d2fd",  # 0.25 - 0.50 inches
    "#00a5fe",  # 0.50 - 0.75 inches
    "#0177fd",  # 0.75 - 1.00 inches
    "#27a31b",  # 1.00 - 1.50 inches
    "#00fa2f",  # 1.50 - 2.00 inches
    "#fffe33",  # 2.00 - 2.50 inches
    "#ffd328",  # 2.50 - 3.00 inches
    "#ffa71f",  # 3.00 - 4.00 inches
    "#ff2b06",
    "#da2304",  # 4.00 - 5.00 inches
    "#aa1801",  # 5.00 - 6.00 inches
    "#ab1fa2",  # 6.00 - 8.00 inches
    "#db2dd2",  # 8.00 - 10.00 inches
    "#ff38fb",  # 10.00+
    "#ffd5fd"]


precip_colormap = mpl.colors.ListedColormap(nws_precip_colors)
item = 18
clevel = [0, 0.5, 1, 2, 6, 10, 15, 20,  30, 40, 50,
          70, 90, 110, 130,150,200,300,400]
norm = mpl.colors.BoundaryNorm(clevel, item)

def plot_windy(data_train,preds_test,data_label,time):

        
    font = {'family'     : 'DejaVu Sans Mono',
            'weight'     : 'bold',
            'size'       : 18
            }
    axes = {'titlesize'  : 18,
            'titleweight': 'heavy',
            'labelsize'  : 18,
            'labelweight': 'bold'
            }
    mpl.rc('font', **font)  # pass in the font dict as kwargs
    mpl.rc('axes', **axes)
    #ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
    #                  edgecolor='black', facecolor='gray', alpha=0.5)
    # Pracific
    latStart = 23; latEnd =24.2375;#25.35
    lonStart = 121.875; lonEnd = 123.1125
    lat = np.linspace(latStart,latEnd,100)
    lon = np.linspace(lonStart,lonEnd,100)
    lons, lats = np.meshgrid(lon, lat)
    plt.figure(figsize=(18,6))
    plt.suptitle(time)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    ax1 = plt.subplot(1, 3, 1)
    # ax1 = plt.axes(projection=ccrs.PlateCarree())
    # ss = np.sum(data_train,axis=2)
    filled_c = ax1.contourf(lons, lats, data_train[15,:,:], levels=np.linspace(200,300,21),cmap='gray',linesytles=None)
 
    ax1.set_xticks(np.arange(120, 123))
    ax1.set_yticks(np.arange(22, 26))
    ax1.set_title('Himawari_images')
    ax1.tick_params('both', labelsize=16)

    
    ax2 = plt.subplot(1, 3, 2)
    filled_c = ax2.contourf(lons, lats, preds_test.detach().numpy()[0,:,:],levels=clevel, norm=norm,cmap=precip_colormap,linesytles=None)    #
    ax2.set_xticks(np.arange(120, 123))
    ax2.set_yticks(np.arange(22, 26))
    ax2.set_title('pred_rainfall')
    ax2.tick_params('both', labelsize=16)
    
    
    
    ax3 = plt.subplot(1, 3, 3)
    filled_c = ax3.contourf(lons, lats, data_label[0,:,:],levels=clevel, norm=norm,cmap=precip_colormap,linesytles=None)
    ax3.set_xticks(np.arange(120, 123))
    ax3.set_yticks(np.arange(22, 26))
    ax3.tick_params('both', labelsize=16)
    ax3.set_title('QPESUMS')
    plt.savefig('tmp.png')
    plt.close('all')
    RGB_matrix = imageio.imread('tmp.png')
    return RGB_matrix
