---
title: Colorbars
description: Colorbars in matplotlib
authors:
    - J. Emmanuel Johnson
path: docs/snippets/visualization/matplotlib
source: colorbars.md
---
# Colorbars

---

## Basics

* [AstroML](https://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut3.html)

---

## Plot Colorbar Only

* [StackOverFlow](https://stackoverflow.com/questions/16595138/standalone-colorbar-matplotlib)


---

## Plot n Colorbars, 1 Plot

* [How to Plot Only One Colorbar for Multiple Plot Using Matplotlib](https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/) - jdhao's Blog
* [Matplotlib Docs](https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/colorbar_placement.html)
* [StackOverflow](https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar)

---

## Colorbar Position

* [Matplotlib Docs](https://matplotlib.org/3.1.0/gallery/axes_grid1/demo_colorbar_with_axes_divider.html)

---

## Scaling


```python
cbar = plt.colorbar(pts,  fraction=0.046,)

```

---

## Normalizing

```python
import matplotlib.colors as colors

boundaries = (0.0, 1.0)

# plot data
pts = ax.pcolor(
    X,Y,Z,
    norm=colors.Normalize(vmin=boundaries[0], vmax=boundaries[1])
    cmap='grays'
)

# plot colorbar
cbar = plt.colorbar(pts, ax=ax, extend='both')
```

**Normalizations**

* Log Scale
* Symmetric Log Scale


**Resources**

* Matplotlib Tutorial - [ColorMap Norms](https://matplotlib.org/tutorials/colors/colormapnorms.html)
* Jake Vanderplas - [Customizing Colorbars](https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html)