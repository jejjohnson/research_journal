# Earth Science Tools

- [DataStructures](#datastructures)
  - [Xarray](#xarray)
  - [GeoPandas](#geopandas)
- [Manipulating ShapeFiles](#manipulating-shapefiles)
- [Visualization](#visualization)
  - [Built-In](#built-in)
  - [Dedicated Plotting Libraries](#dedicated-plotting-libraries)
- [Other Useful Utilities](#other-useful-utilities)
  - [Regridding: **xESMF**](#regridding-xesmf)
  - [Geometries: **rioxarray**](#geometries-rioxarray)

---

These are a few simple tools that can be helpful with dealing spatial-temporal aware datasets in particular from the [xarray](http://xarray.pydata.org/en/stable/) package. These `xr.Datasets` are in the format (lat x lon x time x variable) and many times we just need X and y. There are a few useful functions in here that will help getting coverting that data into useful arrays for processing.

---

## DataStructures

---

### Xarray

<center>
<img src="http://xarray.pydata.org/en/stable/_images/dataset-diagram.png" width="600">

**Source**: [Xarray Data Structure documentation](http://xarray.pydata.org/en/stable/data-structures.html)

</center>



This image says a lot: is the default package for handling spatial-temporal-variable datasets. This alone has helped me contain data where I care about the meta-data as well. Numpy Arrays are great but they are limited in their retention of meta-data. In addition, it has many features that allow you to work with it from a numpy array perspective and even better from a pandas perspective. It makes the whole ordeal a lot easier.

---

### GeoPandas

If you don't work with arrays and you prefer to use shapefiles, then I suggest using GeoPandas to store this data. In the end, it is exactly like [Pandas](https://pandas.pydata.org/) but it has some special routines that cater to working with Earth Sci data. I am no expert and I have really only used the plotting routines and the shape extract routines. But in the end, as a data structure, this would be an easy go to with a relatively low learning curve if you're already familiar with Pandas.


---

## Manipulating ShapeFiles

* [RegionMask](https://regionmask.readthedocs.io/en/stable/index.html)
    > Some additional functionality for having specialized regions.
* [Shapely](https://shapely.readthedocs.io/en/stable/manual.html)
    > The original library which allows one parse shapefiles in an efficient way.
* [Affine](https://github.com/sgillies/affine)
    > The package used to do the tranformation of the polygons to the lat,lon coordinates.
* [Rasterio](https://rasterio.readthedocs.io/en/stable/)
    > Very powerful python package that is really good at transforming the coordinates of your datasets. See [Gonzalo's tutorial](https://www.uv.es/gonmagar/blog/2018/11/11/RasterioExample) for a more specific usecase.

---

## Visualization 


### Built-In

These are packages where the routine is built-in as a side option and not a fully-fledged packaged dedicated to this. Not saying that the built-in functionality isn't extensive, but typically it might use another framework to do some simple routines.


[**xarray**](http://xarray.pydata.org/en/stable/plotting.html)

> The xarray docs have a few examples for how one can plot.


[**geopandas**](http://geopandas.org/)

> This package handles polygons and I have found that it's really good for plotting polygons out-of-the-box.

[**cartopy**](https://scitools.org.uk/cartopy/docs/latest/gallery/index.html)

> A package that handles all of the projections needed for better visualizations of the globe. Works well with matplotlib, geopandas, and xarray.


<details>
<summary>Tutorials</summary>

* [Maps in Scientific Python](https://rabernat.github.io/research_computing_2018/maps-with-cartopy.html)
    > A great tutorial by Rabernat

</details>

---

### Dedicated Plotting Libraries

[**folium**](https://python-visualization.github.io/folium/)

> This is the first of the packages on this list that starts to utilize javascript under the hood. This one is particularly nice for things to do with maps and polygons.

[**hvplot**](https://hvplot.pyviz.org/)

> A nice package that offers a higher level API to the Bokeh library. It allows you do do quite a lot of interactive plots. They have some tutorials for [geographic data](https://hvplot.pyviz.org/user_guide/Geographic_Data.html) whether it be polygons or gridded data.

[**holoviews**](holoviews.org/index.html)

This has been recommended for large scale datasets with [millions of points](http://holoviews.org/user_guide/Large_Data.html)! They have some tutorials for [polygons](http://holoviews.org/user_guide/Geometry_Data.html) and xarray [grids](http://holoviews.org/user_guide/Gridded_Datasets.html).

---

## Other Useful Utilities

### Regridding: [**xESMF**](https://xesmf.readthedocs.io/en/latest/why.html)
  > Ever had one dataset in xarray that had one lat,lon resolution and then you have another dataset with a different lat,lon resolution? Well, you can use this package to easily move from one coordinate grid reference to another. It removes a lot of the difficulty and it is relatively fast for small-medium sized datasets.


### Geometries: [**rioxarray**](https://corteva.github.io/rioxarray/html/index.html)
  > A useful package that allows you to couple geometries with xarray. You can 
    mask or reproject data. I like this package because it's simple and it 
    focuses on what it is good at and nothing else.