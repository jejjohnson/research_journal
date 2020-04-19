# Colab

## Installing Packages from Conda-Forge

**Example**: Cartopy

```bash
# get package then extract
!wget https://anaconda.org/conda-forge/cartopy/0.16.0/download/linux-64/cartopy-0.16.0-py36h81b52dc_2.tar.bz2
!tar xvjf cartopy-0.16.0-py36h81b52dc_2.tar.bz2
!cp -r lib/python3.6/site-packages/* /usr/local/lib/python3.6/dist-packages/
# install dependencies
!pip install shapely pyshp
!apt install libproj-dev libgeos-dev
# finally
import cartopy
```