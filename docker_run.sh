docker run -dit \
    --name unet_well_03 \
    -p 6006:6006 \
    -p 8888:8888 \
    -v ./:/workspace/ \
    -w /workspace/ \
    --gpus all \
    --shm-size 8G \
    unet_well:1.0.3 \
    sleep infinity
