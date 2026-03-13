# DeepSpeed Test Cases <!-- omit in toc -->

[DeepSpeed](https://github.com/microsoft/DeepSpeed) enables world's most powerful language models like MT-530B and BLOOM. It is an easy-to-use deep learning optimization software suite that powers unprecedented scale and speed for both training and inference. `deepspeed` illustrates several example test cases for DeepSpeed training on AWS. 

## 1. Preparation

This guide assumes that you have the following:

* A functional Slurm cluster on AWS.
* Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
* An FSx for Lustre filesystem mounted on `/fsx`.

We recommend that you set up a Slurm cluster using the templates in the architectures [directory](../../1.architectures). You need to set the following environment variables to run these test cases:

```bash
export APPS_PATH=/fsx/apps
export ENROOT_IMAGE=$APPS_PATH/deepspeed.sqsh
export FSX_PATH=/fsx
export MODEL_PATH=$FSX_PATH/deepspeed
export TEST_CASE_PATH=${HOME}/18.deepspeed  # where you copy the test case or set to your test case path
cd $TEST_CASE_PATH                          # Note that we assume that you are here during the following command executions
```



## 2. Build the container

Before running training jobs, you need to use a build docker container image. [Enroot](https://github.com/NVIDIA/enroot) will be used to turn the image into unprivileged sandbox for Slurm but build step may exceed the storage available on the head node so we reccomend building it on a compute node following instructions below (option 2)

### Option 1: build image on a head node

Below are the steps you need to follow:


1. Build the Docker image with the command below in this directory.

   ```bash
    docker build -t deepspeed -f 0.deepspeed.dockerfile .
   ```


2. Once the Docker image is built, you can check if it is present with `docker images`. You should see an output similar to this one:

   ```bash
    REPOSITORY   TAG       IMAGE ID       CREATED          SIZE
    deepspeed     latest    b6c49033c424   9 minutes ago    23.3GB
   ...
   ```

3. Convert the Docker image to a squash file with the command below.

   ```bash
   enroot import -o ${ENROOT_IMAGE} dockerd://deepspeed:latest
   ```

   The file will be stored in the `/apps` directory (by default). The output should look as below.

    ```bash
    [INFO] Fetching image

    36a8c752c28a2db543d2a632a3fc1fcbd5789a6f3d45b9d3a24632420dedcfa8

    [INFO] Extracting image content...
    [INFO] Creating squashfs filesystem...

    Parallel mksquashfs: Using 32 processors
    Creating 4.0 filesystem on /apps/deepspeed.sqsh, block size 131072.
    [========================================================================================================================================================================================================================-] 291068/291068 100%

    Exportable Squashfs 4.0 filesystem, gzip compressed, data block size 131072
            uncompressed data, uncompressed metadata, uncompressed fragments, uncompressed xattrs
            duplicates are not removed
    ...
    ```

Once done proceed to the next stage.

### Option 2: Build image on a compute node

In this option, you will use a compute node to build the image. Submit the job as:

    ```bash
    sbatch 1.build-image.sbatch
    ```


Once the image is prepared, you can proceed to `examples_*` directory for various deepspeed test cases.