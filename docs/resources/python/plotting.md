# Visualization Tricks


---

## Tutorials

*

---

## Colorbars

#### Scaling


```python
cbar = plt.colorbar(pts,  fraction=0.046,)

```

#### Normalizing

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