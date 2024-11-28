username=yyang
project_name=gaussianplant

docker run --gpus all -itd \
    -u $(id -u $username):$(id -g $username) \
    --name ${username}_${project_name} \
    -v /mnt/workspace2024/${username}/GaussianPlant:/home/${username}/mnt/workspace \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    repo-luna.ist.osaka-u.ac.jp:5000/${username}/${project_name}:build \
