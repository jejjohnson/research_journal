# Using Jupyter Notebooks for VSCode Remote Computing

In this tutorial, I will quickly be going over how one can open up a Jupyter Notebook in VSCode from one that has been activated on a slurm server through an interactive node.

## 1. Connect to the server via VSCode

<p align="center">
  <img src="tutorials/jupyter_vscode/pics/1_connect_remote.png" width="600" align="center"/>
</p>


## 2. Connect to an interactive node

Try to use something explicit like the following command:

```bash
srun --nodes=1  --ntasks-per-node=1 --cpus-per-task=28 --time 100:00:00 --exclude=nodo17 --job-name bash-jupyter --pty bash -i
```

<p align="center">
  <img src="tutorials/jupyter_vscode/pics/2_connect_node.png" width="600" align="center"/>
</p>

## 3. Start a Jupyter Notebook


```bash
conda activate jupyterlab
jupyter notebook --ip localhost --port 3001 --no-browser
```

<p align="center">
  <img src="tutorials/jupyter_vscode/pics/3_run_jupyter.png" width="600" align="center"/>
</p>


## 4. Open Jupyter Notebook in VSCode


At this point, something should pop up asking you if you would like to enter a token or your password for your notebook.

<p align="center">
  <img src="tutorials/jupyter_vscode/pics/4_start_notebook.png" width="600" align="center"/>
</p>