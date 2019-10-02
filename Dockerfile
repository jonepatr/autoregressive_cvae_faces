FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
RUN apt-get update
RUN apt-get install -y libsndfile1 wget cmake build-essential tmux git
ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN mkdir /root/.conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && echo "conda activate base" >> ~/.bashrc

RUN pip install luigi librosa pysoundfile pytube ruffus beautifulsoup4 pandas dlib docker matplotlib pytorch_lightning pylint black torchcontrib pdbpp jupyter
RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN conda install -y opencv
WORKDIR /workspace