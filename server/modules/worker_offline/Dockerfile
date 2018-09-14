FROM debian:8
MAINTAINER Ilyes REBAI <irebai@linagora.com>

# Install all our dependencies and set some required build changes
RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    bc \
    bzip2 \
    default-jre \
    g++ \
    git \
    gzip \
    libatlas3-base \
    libtool-bin \
    make \
    python2.7 \
    python3   \
    python-dev \
    python3-dev \
    python-pip \
    python3-pip \
    sox \
    libav-tools \
    subversion \
    vorbis-tools \
    wget \
    zlib1g-dev &&\
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    ln -s /usr/bin/python2.7 /usr/bin/python ; ln -s -f bash /bin/sh

ENV BASE_DIR /opt/speech-to-text

# Speaker diarization
RUN cd /opt && wget http://www-lium.univ-lemans.fr/diarization/lib/exe/fetch.php/lium_spkdiarization-8.4.1.jar.gz && \
    gzip -d lium_spkdiarization-8.4.1.jar.gz && \
    mkdir -p $BASE_DIR

# Build kaldi
COPY kaldi-soft /opt/kaldi-soft
RUN cd /opt && git clone https://github.com/kaldi-asr/kaldi.git && \
    cp /opt/kaldi-soft/Makefile            /opt/kaldi/src/Makefile && \
    cp /opt/kaldi-soft/bin/Makefile        /opt/kaldi/src/bin && \
    cp /opt/kaldi-soft/featbin/Makefile    /opt/kaldi/src/featbin && \
    cp /opt/kaldi-soft/gmmbin/Makefile     /opt/kaldi/src/gmmbin && \
    cp /opt/kaldi-soft/ivectorbin/Makefile /opt/kaldi/src/ivectorbin && \
    cp /opt/kaldi-soft/latbin/Makefile     /opt/kaldi/src/latbin && \
    cp /opt/kaldi-soft/online2bin/Makefile /opt/kaldi/src/online2bin && \
    cd /opt/kaldi/tools && \
    make -j$(nproc) && \
    cd /opt/kaldi/src && ./configure --shared && make depend -j$(nproc) && make -j$(nproc)

# Install RNNOISE package
RUN cd /opt && wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-64bit-static.tar.xz && tar xf ffmpeg-git-64bit-static.tar.xz && \
    git clone https://github.com/xiph/rnnoise.git && cd rnnoise && ./autogen.sh && ./configure && make

WORKDIR $BASE_DIR

# Install python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install numpy
RUN pip3 install -r requirements.txt

COPY . .

# Deploy our offline server
RUN mv scripts/util scripts/utils && \
    ./deploy-offline-decoding.sh /opt/lium_spkdiarization-8.4.1.jar /opt/rnnoise/examples/rnnoise_demo /opt/models /opt/worker.config /opt/ffmpeg-git-*-64bit-static/ffmpeg /opt/kaldi
# Set the default command
CMD ./worker_offline.py
