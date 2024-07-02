## Docker image usage instructions

The docker image that can be built using `Dockerfile` allows automating setup and ensures reproducible work environment.

### Building

You need `docker` installed (Linux: `sudo apt install docker.io` or Windows/MacOS: download Docker Desktop).

Build image with:

```shell
cd rl-testbed-for-energyplus/
docker build . -f docker/Dockerfile -t rl-testbed-for-energyplus
```
Building the docker image takes some time, as it installs a patched version of EnergyPlus and other dependencies.

### Running

The image can be used to run a training as it contains all necessary dependencies. Run:

```shell
docker run -t -i rl-testbed-for-energyplus
```

Another option is to use your local files and run them in the docker container. This is useful for
development purposes. To do so, you can start docker image with your project sources mounted as a container volume:

```shell
docker run -t -i -v /path/to/project:/root/my-project rl-testbed-for-energyplus
```

Tip: simply use `$(pwd)` instead of `/path/to/project` if you want to mount the current directory.

Note: all modifications you will make to your sources located on your hard drive will be reflected in the running 
docker container.

When the container is started, you can run the training script inside you project directory (root/rl-testbed-for-energyplus or root/my-project):

```shell
# Run experiment
python3 RL_training/run_PPO.py 
```

## Considerations
Currently docker installs a CPU-only version of PyTorch as this reduces the size of the image significantly. If you want to use GPU, you need to adjust the `Dockerfile` to install the appropriate version of PyTorch.


## Running jupyter inside docker.

For analysis you may find running a jupyter notebook within the docker container useful. 

The dockerfile exposes port 8888, but you need to link your host port with the docker port when you run the container:

```shell
docker run -t -i -v /path/to/project:/root/my-project rl-testbed-for-energyplus
```

Once you are inside your docker, run the following: 

```shell
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

You will get some output which will be familiar to notebook users, but this time with a token. Copy the url with `127.0.0.1` into your brower (if that doesn't work, try replacing `127.0.0.1` with `localhost`). You should now have the usual notebook interface in your local browser.

```
To access the server, open this file in a browser:
    file:///root/.local/share/jupyter/runtime/jpserver-20-open.html
Or copy and paste one of these URLs:
    http://ac2f2b4f84f6:8888/tree?token=6755614d69db58f07773a2f13e2f5e66494e0884f3af7621
    http://127.0.0.1:8888/tree?token=6755614d69db58f07773a2f13e2f5e66494e0884f3af7621

```