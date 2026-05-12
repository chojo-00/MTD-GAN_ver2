FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure pip to use Kakao mirror (faster in Korea)
RUN printf "%s\n"\
    "[global]"\
    "index-url=https://mirror.kakao.com/pypi/simple/"\
    "extra-index-url=https://pypi.org/simple/"\
    "trusted-host=mirror.kakao.com"\
    > /etc/pip.conf

# Install system packages
RUN apt update -qq && apt install -qqy\
        sudo\
        tzdata\
        vim\
        curl\
        jq\
        git\
        libgl1-mesa-glx\
        libglib2.0-0\
    && apt clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip + install all Python packages in one layer
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
    accelerate==0.27.2\
    click\
    ipywidgets\
    jupyter\
    jupyter_contrib_nbextensions\
    jupyterlab==4.2.3\
    matplotlib\
    natsort\
    numpy==1.24.4\
    nibabel\
    monai\
    opencv-contrib-python-headless\
    openpyxl\
    pandas\
    Pillow\
    psutil\
    pylibjpeg\
    pylibjpeg-libjpeg\
    pylibjpeg-openjpeg\
    pydicom\
    requests\
    scipy\
    scikit-image\
    scikit-learn\
    tqdm \
    gudhi \
    cvxpy
