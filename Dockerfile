FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer="endrem@hi.no"

RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

RUN apt-get update && apt-get install -y git

RUN pip install pandas

RUN pip install opencv-python

RUN pip install keras

RUN pip install keras-applications

RUN pip install pandas

RUN pip install sklearn

RUN pip install scipy

RUN pip install pillow

RUN pip install scikit-image

RUN pip install -U git+https://github.com/qubvel/efficientnet@v1.1.0
RUN pip install -U git+https://github.com/emoen/deepaugment

#DeepAugment packages
RUN mkdir /usr/local/lib/python3.6/dist-packages/reports/
RUN mkdir /usr/local/lib/python3.6/dist-packages/reports/experiments/
RUN chmod 777 -R /usr/local/lib/python3.6/dist-packages/reports/experiments
RUN chmod 777 -R /usr/local/lib/python3.6/dist-packages
