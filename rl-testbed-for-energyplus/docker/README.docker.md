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