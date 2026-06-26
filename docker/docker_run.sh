username=yyang
project_name=gs_base
container_name=LangSplat
folder_name=LangSplat

docker run --gpus all -itd \
    -u $(id -u $username):$(id -g $username) \
    --name ${username}_${container_name} \
    -v /mnt/workspace2024/${username}/${folder_name}:/home/${username}/mnt/workspace \
    --mount type=bind,source="/mnt/poplin/share/2023/users/yang/gaussianplant_data",target=/mnt/data \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    repo-luna.ist.osaka-u.ac.jp:5000/yyang/gs_base:build \
