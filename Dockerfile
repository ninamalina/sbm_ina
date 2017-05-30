FROM tiagopeixoto/graph-tool

RUN pacman -S --noconfirm python-pip

RUN pacman -Sy
RUN pacman -S --noconfirm pacman-mirrorlist
RUN pacman -S --noconfirm jdk8-openjdk

ENV SBM /sbm
ADD requirements.txt ${SBM}/requirements.txt
WORKDIR ${SBM}

RUN pip install -r requirements.txt

