# Running Subsequent Scripts


### Case I: Run Script 1, Wait, Run Script 2

For this, we want to wait for `[script1.py](http://script1.py)` to finish successfully, then we run `script2.py`.

```bash
#!/usr/bin/env bash
python script1.py & python script2.py
```

### Case II: Run Script 1, Wait, Run Script 2 IFF Script 1 has failed

This is the case where we want to run `[script2.py](http://script2.py)` but IFF `[script1.py](http://script1.py)` has finished

```bash
#!/usr/bin/env bash
python script1.py || python script2.py
```

### Case III: Run script1 AND script 2

We want to run both scripts concurrently as background processes

```bash
#!/usr/bin/env bash
python script1.py & python script2.py
```