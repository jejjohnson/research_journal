# Running Subsequent Scripts


## Case I: Run Script 1, Wait, Run Script 2

For this, we want to wait for `[script1.py](http://script1.py)` to finish successfully, then we run `script2.py`.

```bash
#!/usr/bin/env bash
python script1.py && 
python script2.py
```

## Case II: Run Script 1, Wait, Run Script 2 IFF Script 1 has failed

This is the case where we want to run `[script2.py](http://script2.py)` but IFF `[script1.py](http://script1.py)` has finished

```bash
#!/usr/bin/env bash
python script1.py || python script2.py
```

## Case III: Run script1 AND script 2

We want to run both scripts concurrently as background processes

```bash
#!/usr/bin/env bash
python script1.py & 
python script2.py
```

## Bonus Stuff

### Wait Command

You can manually force the code to wait for all background processes to finish in the script before proceeding forward. This will continue even if there is a failure. It's very similar to the `||` command. There isn't a real difference.

```bash
wait
```

### Permissions

So once you have a bash script `.sh`, make sure you have the permissions otherwise you won't be able to execute it. Just run this command

```bash
chmod +x <filename>
```

Now you can run the script.

```bash
./ <filename>
```