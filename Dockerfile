# Chosen to match the CUDA 11.7 installed on this machine
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ARG USER_ID=1130
ARG GROUP_ID=300
ARG USER_NAME="yyang"
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN groupadd -g "${GROUP_ID}" "${USER_NAME}" && useradd -u "${USER_ID}" -m "${USER_NAME}" -g "${USER_NAME}"
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# Install dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y
RUN apt install -y build-essential wget

#  for viewers
RUN apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
RUN apt install -y cmake git vim

#  for training
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

#  for dataset conversion
RUN apt install -y colmap imagemagick

#  cleanup
RUN apt clean && rm -rf /var/lib/apt/lists/*


# Setup Python environment
COPY ./environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml
RUN /bin/bash -c "conda init bash"
RUN echo "conda activate gs_base" >> /root/.bashrc

# Now mount the actual directory, hopefully
WORKDIR /home/yyang/mnt/workspace/