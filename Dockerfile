FROM tiagopeixoto/graph-tool

RUN pacman -S --noconfirm python-pip

ENV SBM /sbm
ADD requirements.txt ${SBM}/requirements.txt
WORKDIR ${SBM}

RUN pip install -r requirements.txt

