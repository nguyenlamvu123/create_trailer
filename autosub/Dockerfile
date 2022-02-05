FROM ubuntu:20.04

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

#RUN apt-get update \
#    && apt-get install -y python3-pip python3-dev ffmpeg\
#    && cd /usr/local/bin \
#    && ln -s /usr/bin/python3 python \
#    && pip3 install --upgrade pip
    
RUN apt-get update
RUN apt-get install -y python3-pip python3-dev
#RUN cd /usr/local/bin
#RUN ln -s /usr/bin/python3 python
RUN pip3 install --upgrade pip

RUN apt install -y ffmpeg
RUN apt-get -y install curl
RUN curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
RUN chmod a+rx /usr/local/bin/youtube-dl

# set work directory
WORKDIR /usr/src/app

# install dependencies
#RUN python3 -m pip install --ignore-installed --upgrade "Download URL"
#RUN python3 -m pip install --upgrade pip 
#COPY ./requirements.txt /usr/src/app
#RUN python3 -m pip install -r requirements.txt
RUN apt-get install -y git
RUN python3 -m pip install git+https://github.com/agermanidis/autosub.git

# copy project
COPY . /usr/src/app

CMD ["/bin/bash"]
