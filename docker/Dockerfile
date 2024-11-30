FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
ARG USER_ID=1130
ARG GROUP_ID=300
ARG USER_NAME="yyang"
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN groupadd -g "${GROUP_ID}" "${USER_NAME}" && useradd -u "${USER_ID}" -m "${USER_NAME}" -g "${USER_NAME}"
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# Install base utilities
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y build-essential wget ninja-build unzip libgl-dev ffmpeg cmake g++  libglfw3-dev libgles2-mesa-dev  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

COPY ./ ./

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN conda update -n base conda
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba
RUN conda env create -f environment.yml
RUN conda init bash
RUN echo "conda activate gaussian_splatting" >> ~/.bashrc
SHELL ["conda", "run", "-n", "gaussian_splatting", "/bin/bash", "-c"]
RUN conda install jupyter colmap
RUN conda remove ffmpeg -y


ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gaussian_splatting", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]