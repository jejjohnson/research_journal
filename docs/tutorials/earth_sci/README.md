# Earth Science Tools

These are a few simple tools that can be helpful with dealing spatial-temporal aware datasets in particular from the [xarray](http://xarray.pydata.org/en/stable/) package. These `xr.Datasets` are in the format (lat x lon x time x variable) and many times we just need X and y. There are a few useful functions in here that will help getting coverting that data into useful arrays for processing.


---

### Key Helpful Functions

[**Density Cubes**](esdc/transform.py)

We can extract 'density cubes' which transform point estimates to estimates with spatial and temporal features. It is like a sliding window except in lat-lon-time space. Very useful when attempted to get do density estimation. A demo notebook can be found [here](notebooks/minicubes.ipynb).

[**Regrid Function**](esdc/transform.py)

Utilizing the [xESMF](https://xesmf.readthedocs.io/en/latest/) package, we can regrid datasets based on a reference dataset. Very useful when we don't have datasets with the same spatial dimensions.

[**ShapeFile Masks**](esdc/shape.py)

This has various functions to allows one to use shapefiles to mask datasets. I have only a few sample functions that parse shape files of interest like countries or US states. But it is flexible enough to use other interesting attributes like population. We only need the raster functions. A demo notebook can be found [here](notebooks/shapefile_masks.ipynb).

## Useful Resources

### Manipulating ShapeFiles

* [RegionMask](https://regionmask.readthedocs.io/en/stable/index.html)
    > Some additional functionality for having specialized regions.
* [Shapely](https://shapely.readthedocs.io/en/stable/manual.html)
    > The original library which allows one parse shapefiles in an efficient way.
* [Affine](https://github.com/sgillies/affine)
    > The package used to do the tranformation of the polygons to the lat,lon coordinates.
* [Rasterio](https://rasterio.readthedocs.io/en/stable/)
    > Very powerful python package that is really good at transforming the coordinates of your datasets. See [Gonzalo's tutorial](https://www.uv.es/gonmagar/blog/2018/11/11/RasterioExample) for a more specific usecase.

### Visualization 

* [xarray](http://xarray.pydata.org/en/stable/plotting.html)
    > The xarray docs have a few examples for how one can plot.
* [geopandas]()
    > This package handles polygons and I have found that it's really good for plotting polygons out-of-the-box.
* [cartopy](https://scitools.org.uk/cartopy/docs/latest/gallery/index.html)
    > A package that handles all of the projections needed for better visualizations of the globe. Works well with geopandas and xarray.
* [folium](https://python-visualization.github.io/folium/)
    > This is the first of the packages on this list that starts to utilize javascript under the hood. This one is particularly nice for things to do with maps and polygons.
* [hvplot](https://hvplot.pyviz.org/)
    > A nice package that offers a higher level API to the Bokeh library. It allows you do do quite a lot of interactive plots. They have some tutorials for [geographic data](https://hvplot.pyviz.org/user_guide/Geographic_Data.html) whether it be polygons or gridded data.
* [holoviews](holoviews.org/index.html)
    > This has been recommended for large scale datasets with [millions of points](http://holoviews.org/user_guide/Large_Data.html)! They have some tutorials for [polygons](http://holoviews.org/user_guide/Geometry_Data.html) and xarray [grids](http://holoviews.org/user_guide/Gridded_Datasets.html).
* [Maps in Scientific Python](https://rabernat.github.io/research_computing_2018/maps-with-cartopy.html)
    > A great tutorial by Rabernat