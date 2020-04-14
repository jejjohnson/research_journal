---
title: 2.0 Processed Data
description: Overview of my ML4OCN Research
authors:
    - J. Emmanuel Johnson
path: docs/projects/ML4OCN/ARGO_project/global_data
source: 1_load_processed_data.md
---
# Loading the Data


### **Important** - Paths and Directories

This is annoying but it needs to be defined otherwise things get confusing. We need a few important paths to be pre-defined:

| Name           | Variable       | Purpose                                                                      |
| -------------- | -------------- | ---------------------------------------------------------------------------- |
| Project        | `PROJECT_PATH` | top level directory for the project (assuming megatron)                      |
| Code           | `CODE_PATH`    | folder of any dedicated functions that we use                                |
| Raw Data       | `RAW_PATH`     | where the raw data is. Ideally, we **never** touch this ever except to read. |
| Processed Data | `DATA_PATH`    | where the processed data is stored                                           |
| Interim Data   | `INTERIM_PATH` | where we save the training, validation and testing data                      |
| Saved Models   | `MODEL_PATH`   | where we save any trained models                                             |
| Results Data   | `RESULTS_PATH` | where we save any data results or outputs from ML models                     |
| Figures        | `FIG_PATH`     | where we store any plotted figures during any part of our ML pipeline        |

This cell checks to see if all of the paths exist. If there is a path missing, it probably means you're not in megatron. If that's the case...well, we'll cross that bridge when we get there.

<details>

```python
import pathlib
import sys

# define the top level directory
PROJECT_PATH = pathlib.Path("/media/disk/erc/papers/2019_ML_OCN/")
CODE_PATH = PROJECT_PATH.joinpath("ml4ocean", "src")

# check if path exists and is a directory
assert PROJECT_PATH.exists() & PROJECT_PATH.is_dir()
assert CODE_PATH.exists() & CODE_PATH.is_dir()

# add code and project paths to PYTHONPATH (to call functions)
sys.path.append(str(PROJECT_PATH))
sys.path.append(str(CODE_PATH))

# specific paths
FIG_PATH = PROJECT_PATH.joinpath("ml4ocean/reports/figures/global/")
RAW_PATH = PROJECT_PATH.joinpath("data/global/raw/")
DATA_PATH = PROJECT_PATH.joinpath("data/global/processed/")
INTERIM_PATH = PROJECT_PATH.joinpath("data/global/interim/")
MODEL_PATH = PROJECT_PATH.joinpath("models/global/")
RESULTS_PATH = PROJECT_PATH.joinpath("data/global/results/")

# check if path exists and is a directory
assert FIG_PATH.exists() & FIG_PATH.is_dir()
assert RAW_PATH.exists() & RAW_PATH.is_dir()
assert DATA_PATH.exists() & DATA_PATH.is_dir()
assert INTERIM_PATH.exists() & INTERIM_PATH.is_dir()
assert MODEL_PATH.exists() & MODEL_PATH.is_dir()
assert RESULTS_PATH.exists() & RESULTS_PATH.is_dir()
```

</details>

---

## Python Packages


```python
# Standard packages
import numpy as np
import pandas as pd
```

## 1. Load Processed Global Data

In this section, I will load the metadata and the actual data. The steps involved are:

1. Define the filepath (check for existence)
2. Open meta data and real data
3. Check that the samples correspond to each other.
4. Check if # of features are the same

### 1.1 - Meta Data


```python
# name of file
meta_name = "METADATA_20200310.csv"

# get full path
meta_file = DATA_PATH.joinpath(meta_name)

# assert meta file exists
error_msg = f"File '{meta_file.name}' doesn't exist. Check name or directory."
assert meta_file.exists(), error_msg

# assert meta file is a file
error_msg = f"File '{meta_file.name}' isn't a file. Check name or directory."
assert meta_file.is_file(), error_msg

# open meta data
meta_df = pd.read_csv(f"{meta_file}")

meta_df.head().to_markdown()
```




|      |     wmo | n_cycle |    N |     lon |     lat |    juld | date                |
| ---: | ------: | ------: | ---: | ------: | ------: | ------: | :------------------ |
|    0 | 2902086 |       1 |    1 | 88.6957 | 12.1639 | 23009.2 | 2012-12-30 03:58:59 |
|    1 | 2902086 |      10 |   10 | 88.6033 | 12.4128 | 23018.1 | 2013-01-08 03:24:59 |
|    2 | 2902086 |     100 |   64 | 86.2039 | 13.7915 | 23432.1 | 2014-02-26 03:34:59 |
|    3 | 2902086 |     101 |   65 | 86.3116 |   13.75 | 23437.1 | 2014-03-03 03:26:59 |
|    4 | 2902086 |     102 |   66 | 86.3971 | 13.7588 | 23442.1 | 2014-03-08 03:31:59 |



### 1.2 - Input Data


```python
# name of file
data_name = "SOCA_GLOBAL2_20200310.csv"

# get full path
data_file = DATA_PATH.joinpath(data_name)

# assert exists
error_msg = f"File '{data_file.name}' doesn't exist. Check name or directory."
assert data_file.exists(), error_msg

# assert meta file is a file
error_msg = f"File '{data_file.name}' isn't a file. Check name or directory."
assert data_file.is_file(), error_msg

# load data
data_df = pd.read_csv(f"{data_file}")

data_df.head().iloc[:, :6].to_markdown()
```




|      |    N |         wmo | n_cycle |     sla |     PAR | RHO_WN_412 |
| ---: | ---: | ----------: | ------: | ------: | ------: | ---------: |
|    0 |    1 | 2.90209e+06 |       1 | -4.7044 | 42.6541 |  0.0254617 |
|    1 |    2 | 2.90209e+06 |       2 | -4.0382 | 42.6541 |  0.0254617 |
|    2 |    3 | 2.90209e+06 |       3 | -3.4604 | 44.2927 |  0.0240942 |
|    3 |    4 | 2.90209e+06 |       4 | -2.8404 | 42.7664 |   0.024917 |
|    4 |    5 | 2.90209e+06 |       5 |  -2.394 | 42.7664 |   0.024917 |



### 1.3 - Checks

I do a number of checks to make sure that our data follows a standard and that I am reproducing the same results.

* Number of samples are equal for both
* 7 meta features 
* 48 data features (26 data + 19 levels + 3 meta)
* check features in columns


```python
# same number of samples
error_msg = f"Mismatch between meta and data: {data_df.shape[0]} =/= {meta_df.shape[0]}"
assert data_df.shape[0] == meta_df.shape[0], error_msg

# check number of samples
n_samples = 25413
error_msg = f"Incorrect number of samples: {data_df.shape[0]} =/= {n_samples}"
assert data_df.shape[0] == n_samples, error_msg

# check meta feature names
meta_features = ['wmo', 'n_cycle', 'N', 'lon', 'lat', 'juld', 'date']
error_msg = f"Missing features in meta data."
assert meta_df.columns.tolist() == meta_features, error_msg

# check data feature names
input_meta_features = ['N', 'wmo', 'n_cycle']
input_features = ['sla', 'PAR', 'RHO_WN_412', 'RHO_WN_443',
       'RHO_WN_490', 'RHO_WN_555', 'RHO_WN_670', 'doy_sin', 'doy_cos',
       'x_cart', 'y_cart', 'z_cart', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
       'PC7', 'PC1.1', 'PC2.1', 'PC3.1', 'PC1.2', 'PC2.2', 'PC3.2', 'PC4.1']
output_features = ['bbp', 'bbp.1', 'bbp.2', 'bbp.3', 'bbp.4', 'bbp.5', 'bbp.6', 'bbp.7',
       'bbp.8', 'bbp.9', 'bbp.10', 'bbp.11', 'bbp.12', 'bbp.13', 'bbp.14',
       'bbp.15', 'bbp.16', 'bbp.17', 'bbp.18']
features = input_meta_features + input_features + output_features
error_msg = f"Missing features in input data."
assert data_df.columns.tolist() == features, error_msg
```

### 1.4 - Convert metadata to indices (**Important**)

To make our life easier, we're going to eliminate the need to keep track of meta data all of the time. So I'm going to merge the datasets together to form one dataframe. Then I will set the index to be the metadata values. The remaining parts will be columns which will be features. 

So in the end, we will have a dataframe where:

* the **indices** is the metadata (e.g. wmo, n_cycle) 
* the **columns** are the actual features (e.g. sla, pca components, bbp, etc).


```python
# merge meta and data
full_df = pd.merge(meta_df, data_df)

# convert meta information to indices
full_df = full_df.set_index(meta_features)

# checks - check indices match metadata
meta_features = ['wmo', 'n_cycle', 'N', 'lon', 'lat', 'juld', 'date']
error_msg = f"Missing features in input data."
assert full_df.index.names == meta_features, error_msg

# checks - check column names match feature names
input_features = ['sla', 'PAR', 'RHO_WN_412', 'RHO_WN_443',
       'RHO_WN_490', 'RHO_WN_555', 'RHO_WN_670', 'doy_sin', 'doy_cos',
       'x_cart', 'y_cart', 'z_cart', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
       'PC7', 'PC1.1', 'PC2.1', 'PC3.1', 'PC1.2', 'PC2.2', 'PC3.2', 'PC4.1']
output_features = ['bbp', 'bbp.1', 'bbp.2', 'bbp.3', 'bbp.4', 'bbp.5', 'bbp.6', 'bbp.7',
       'bbp.8', 'bbp.9', 'bbp.10', 'bbp.11', 'bbp.12', 'bbp.13', 'bbp.14',
       'bbp.15', 'bbp.16', 'bbp.17', 'bbp.18']
features = input_features + output_features
error_msg = f"Missing features in input data."
assert full_df.columns.tolist() == features, error_msg
```


```python
print('Dataframe Features:', full_df.shape)
full_df.columns.tolist()[:10]
```

    Dataframe Features: (25413, 45)

    ['sla',
     'PAR',
     'RHO_WN_412',
     'RHO_WN_443',
     'RHO_WN_490',
     'RHO_WN_555',
     'RHO_WN_670',
     'doy_sin',
     'doy_cos',
     'x_cart']


```python
print('Dataframe Indices (meta vars):', len(full_df.index.names))
full_df.index.names
```

    Dataframe Indices (meta vars): 7

    FrozenList(['wmo', 'n_cycle', 'N', 'lon', 'lat', 'juld', 'date'])


---

## 2 - Training and Test Split

### 2.1 - Independent Set I (SOCA2016)

This independent set has a set number of independent floats which are not counted in the training or validation phase. These floats were in a paper (Sauzede et. al., 2016) and used during the testing phase to showcase how well the models did.

* 6901472
* 6901493
* 6901523
* 6901496

So we need to take these away from the data.


```python
# soca2016 independent floats
soca2016_floats = ["6901472", "6901493", "6901523", "6901496"]

# subset soca2016 floats
soca2016_df = full_df[full_df.index.isin(soca2016_floats, level='wmo')]
```

#### Checks


```python
# check number of samples (meta, inputs)
n_samples = 378
error_msg = f"Incorrect number of samples for soca2016 floats: {soca2016_df.shape[0]} =/= {n_samples}"
assert soca2016_df.shape[0] == n_samples, error_msg
```

### 2.2 - Indpendent Set II (ISPRS2020)

This independent set was a set of floats taken from the ISPRS paper (Sauzede et. al., 2020 (pending...)). These floats were used as the independent testing set to showcase the performance of the ML methods.

* 6901486 (North Atlantic?)
* 3902121 (Subtropical Gyre?)

So we need to take these away from the data.


```python
# isprs2020 independent floats
isprs2020_floats = ["6901486", "3902121"]

# isprs2020 independent floats
isprs2020_df = full_df[full_df.index.isin(isprs2020_floats, level='wmo')]
```

#### Checks


```python
# check number of samples (meta, inputs)
n_samples = 331
error_msg = f"Incorrect number of samples for isprs2016 floats: {isprs2020_df.shape[0]} =/= {n_samples}"
assert isprs2020_df.shape[0] == n_samples, error_msg
```

### 2.3 - ML Data

Now we want to subset the input data to be used for the ML models. Basically, we can subset all datasets that **are not** in the independent floats. In addition, we want all of the variables in the input features that we provided earlier.


```python
# subset non-independent flows
ml_df = full_df[~full_df.index.isin(isprs2020_floats + soca2016_floats, level='wmo')]
```

#### Checks


```python
# check number of samples (meta, inputs)
n_samples = 24704
error_msg = f"Incorrect number of samples for non-independent floats: {ml_df.shape[0]} =/= {n_samples}"
assert ml_df.shape[0] == n_samples, error_msg
```

### 2.4 - Inputs, Outputs

Lastly, we need to split the data into training, validation (and possibly testing). Recall that all the inputs are already written above and the outputs as well.


```python
input_df = ml_df[input_features]
output_df = ml_df[output_features]

# checks - Input Features
n_input_features = 26
error_msg = f"Incorrect number of features for input df: {input_df.shape[1]} =/= {n_input_features}"
assert input_df.shape[1] == n_input_features, error_msg

# checks - Output Features
n_output_features = 19
error_msg = f"Incorrect number of features for output df: {output_df.shape[1]} =/= {n_output_features}"
assert output_df.shape[1] == n_output_features, error_msg
```

---

## 3. Final Dataset (saving)

### 3.1 - Print out data dimensions (w. metadata)


```python
print("Input Data:", input_df.shape)
print("Output Data:", output_df.shape)
print("SOCA2016 Independent Data:", soca2016_df[input_features].shape, soca2016_df[output_features].shape)
print("ISPRS2016 Independent Data:", isprs2020_df[input_features].shape, isprs2020_df[output_features].shape)
```

    Input Data: (24704, 26)
    Output Data: (24704, 19)
    SOCA2016 Independent Data: (378, 26) (378, 19)
    ISPRS2016 Independent Data: (331, 26) (331, 19)


### 3.2 - Saving

* We're going to save the data in the `global/interim/` path. This is to prevent any overwrites. 
* We also need to `index=True` for the savefile in order to preserve the metadata indices.


```python
input_df.to_csv(f"{INTERIM_PATH.joinpath('inputs.csv')}", index=True)
```

### 3.3 - Loading

This is a tiny bit tricky if we want to preserve the meta data as the indices. So we need to set the index to be the same meta columns that we used last time via the `.set_index(meta_vars)` command.


```python
test_inputs_df = pd.read_csv(f"{INTERIM_PATH.joinpath('inputs.csv')}")

# add index
test_inputs_df = test_inputs_df.set_index(meta_features)
```

### 3.4 - Checking

So curiously, we cannot compare the dataframes directly because there is some numerical error when saving them. But if we calculate the exact differences between them, we find that they are almost equal. See below what happens if we calculate the exact difference between the arrays.


```python
# example are they exactly the same?
# np.testing.assert_array_equal(test_inputs_df.describe(), input_df.describe())
np.testing.assert_array_equal(test_inputs_df.values, input_df.values)
```

```python
AssertionError: 
Arrays are not equal

Mismatched elements: 96056 / 642304 (15%)
Max absolute difference: 1.42108547e-14
Max relative difference: 3.92139227e-16
    x: array([[-4.704400e+00,  4.265410e+01,  2.546170e-02, ..., -3.458944e+00,
        -1.017509e-02, -1.025450e+00],
        [-9.015000e-01,  4.455050e+01,  2.060340e-02, ..., -3.691716e+00,...
    y: array([[-4.704400e+00,  4.265410e+01,  2.546170e-02, ..., -3.458944e+00,
        -1.017509e-02, -1.025450e+00],
        [-9.015000e-01,  4.455050e+01,  2.060340e-02, ..., -3.691716e+00,...
```

We get an assertion error that they're not equal. There is a mismatch difference of order 1e-15 for the absolute and relative differences. That's numerical error probably due to compression that comes when saving and loading data. Let's check again but with a little less expected precision.


```python
np.testing.assert_array_almost_equal(test_inputs_df.values, input_df.values, decimal=1e-14)
```

so just by reducing the precision by a smidge (1e-14 instead of 1e-15), we find that the arrays are the same. So we can trust it.

### 3.5 - Save the rest of the data


```python
input_df.to_csv(f"{INTERIM_PATH.joinpath('inputs.csv')}", index=True)
output_df.to_csv(f"{INTERIM_PATH.joinpath('outputs.csv')}", index=True)
soca2016_df.to_csv(f"{INTERIM_PATH.joinpath('soca2016.csv')}", index=True)
isprs2020_df.to_csv(f"{INTERIM_PATH.joinpath('isprs2020.csv')}", index=True)
```
