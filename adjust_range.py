import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RangeSlider

P_wedge_ratio=np.load("P_wedge_ratio.npy")
sh0,sh1=P_wedge_ratio.shape
cyl_extent=np.load("cyl_extent.npy")

# plt.imshow(P_wedge_ratio.T,cmap="RdBu",norm="log",origin="lower",aspect=sh0/sh1)

fig, ax = plt.subplots(figsize=(6, 5))
plt.subplots_adjust(bottom=0.25)  # space at the bottom for the slider

im = ax.imshow(P_wedge_ratio.T, cmap="RdBu",origin="lower",norm="log",extent=cyl_extent)
ax.set_ylim(cyl_extent[0],cyl_extent[1])
cbar = fig.colorbar(im, ax=ax)

# Create the RangeSlider axes and widget
# Syntax: [left, bottom, width, height] as fractions of the figure size
slider_ax = fig.add_axes([0.20, 0.08, 0.55, 0.03]) 
slider = RangeSlider(slider_ax, "Color Range", P_wedge_ratio.min(), P_wedge_ratio.max(), valinit=(P_wedge_ratio.min(), P_wedge_ratio.max()))

# Define the callback function to handle drag events
def update(val):
    # 'val' is a tuple containing the current (low_handle, high_handle) positions
    im.norm.vmin = val[0]
    im.norm.vmax = val[1]

    # ax.set_ylim(cyl_extent[2], cyl_extent[3]) # to show the whole range (default)
    # ax.set_ylim(cyl_extent[0],cyl_extent[1]) # to make the plot square

    
    # Redraw the canvas smoothly without recreating the colorbar
    fig.canvas.draw_idle()

# 5. Bind the listener function to the slider
slider.on_changed(update)

plt.show()