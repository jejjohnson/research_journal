# Ideas

---

## Dask Tutorial - gufunc

In this tutorial, I would like to see how we can use dask to do fast calculations on the Earth Science data cubes. This will be based on the following tutorials:

* Calculating Pearson and Spearman - [XArray Docs](http://xarray.pydata.org/en/stable/dask.html) | [Dask Docs](https://examples.dask.org/xarray.html)
* Generalized ufuncs - [Dask Docs](https://docs.dask.org/en/latest/array-gufunc.html)
* Apply ufunc with Xarray - [Docs](http://xarray.pydata.org/en/stable/examples/apply_ufunc_vectorize_1d.html)


**Algorithms**

* Pearson Correlation Coefficient
* Spearman Correlation Coefficient
* $\rho$-V Coefficient
* Centered Kernel Alignment

**Smoothing**

* [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) | [Scipy - From scratch](https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html) | [StackOverFlow](https://stackoverflow.com/questions/57092509/how-to-apply-a-1d-median-filter-to-a-3d-dataarray-using-xarray-apply-ufunc)


## Shape Files and Rasters

* [Corteva GeoCube](https://corteva.github.io/geocube/stable/examples/examples.html)
* [Corteva RioXarray](https://corteva.github.io/rioxarray/html/examples/clip_geom.html) - [Clip](https://corteva.github.io/rioxarray/html/examples/clip_geom.html) | [Clip Box](https://corteva.github.io/rioxarray/html/examples/clip_box.html) | [Demo](https://gis.stackexchange.com/questions/328128/extracting-data-within-geometry-shape/328320#328320)
* [Xarray to CSV](https://gis.stackexchange.com/questions/358051/convert-raster-to-csv-with-lat-lon-and-value-columns/358057#358057)

---

## Loading Data

#### CMIP6 

I've used this data for my experiments, so I would like to make a quick tutorial about how we can load this, query the parts you need, and do operations.

* [CMIPData Docs](https://cmipdata.readthedocs.io/en/latest/)
  * [Demo with Anomalies](https://github.com/swartn/cmip6-gmst-anoms) - [Normal](https://github.com/swartn/cmip6-gmst-anoms/blob/master/gmst_cmip6.ipynb) | [Parallel](https://github.com/swartn/cmip6-gmst-anoms/blob/master/gmst_cmip6_parallel.ipynb)
* [CMIP6 Preprocessing](https://github.com/jbusecke/cmip6_preprocessing)
  * [Pangeo Examples](https://github.com/pangeo-data/pangeo-cmip6-examples)
  * [Pangeo Datastore](https://github.com/pangeo-data/pangeo-datastore)


#### ERA5

This is some reanalysis data that I've used in my experiments. I would like to do the same as the above part.

* [Pangeo Datastore](https://github.com/pangeo-data/pangeo-datastore)
* [Pangeo Examples](https://github.com/pangeo-data/pangeo-era5)
  * [ML Flow](https://github.com/pangeo-data/ml-workflow-examples)
  * [Tutorials](https://github.com/pangeo-data/pangeo-tutorial/tree/agu2019/notebooks)

#### Earth Science Data Cubes

* [X] - Done!