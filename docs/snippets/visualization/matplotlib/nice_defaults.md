# Defaults

## My Go to

I go with this setup and it typically gives me decent style of plots for my first pass:

```python
import matplotlib
import seaborn as sns

sns.reset_defaults()
sns.set_context(context='talk',font_scale=0.7, rc={'font.family': 'sans-serif'})
%matplotlib inline
```

**Note**: change the context accordingly (e.g. `poster`, `talk`, `paper`)
