# Usage

After successfully building the Docker image you can use it as described below.

## Preparation

Create a directory for your images with two sub directories `in` and `out`.
Put all the images you want to get a depth estimation for into the `in` sub directory.

## Executing Depth Estimation

Decide on the device you want the model to work on, e.g., `cpu` or `cuda`. 
Then you can execute

    docker run --rm -v /path/to/your/images:/images -e DEVICE=cpu name_of_docker_image

You will now find the estimations (images and numpy arrays) in the out folder.