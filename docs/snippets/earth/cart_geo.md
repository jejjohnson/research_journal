# Cartesian Coordinates 2 Geocoordinates




### Geocoordinates 2 Cartesian Coordinates


```python
import pandas as pd
import numpy as np

def geo_2_cartesian(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms geo coordinates (lat, lon) to cartesian coordinates
    (x, y, z).
    
    Parameters 
    ----------
    df : pd.DataFrame
        A dataframe with the geo coordinates values. The columns need to 
        have the following ['lat', 'lon]
    
    Returns
    -------
    df : pd.DataFrame
        A dataframe with the converted values.

    Example
    -------
    >> df = geo_2_cartesian(df)
    """
    cols = df.columns.tolist()

    if "lat" not in cols or "lon" not in cols:
        print("lat,lon columns not present in df.")
        return df

    # approximate earth radius
    earth_radius = 6371

    # transform from degrees to radians
    # df = df.apply(lambda x: np.deg2rad(x) if x.name in ['lat', 'lat'] else x)
    df["lat"] = np.deg2rad(df["lat"])
    df["lon"] = np.deg2rad(df["lon"])

    # From Geo coords to cartesian coords
    df["x"] = earth_radius * np.cos(df["lat"]) * np.cos(df["lon"])
    df["y"] = earth_radius * np.cos(df["lat"]) * np.sin(df["lon"])
    df["z"] = earth_radius * np.sin(df["lat"])

    # drop original lat,lon columns
    df = df.drop(["lat", "lon"], axis=1)

    return df
```

---

### Cartesian Coordinates 2 Geocoordinates

```python
def cartesian_2_geo(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    R = 6371 # radius of the earth
    lat = np.degrees(np.arcsin(z / R))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon
```

**Source**: [Blog](http://www.movable-type.co.uk/scripts/latlong.html) | [Stackoverflow](https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates)