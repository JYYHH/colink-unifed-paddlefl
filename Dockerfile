FROM paddlepaddle/paddlefl:1.2.0-gpu

SHELL ["/bin/bash", "-c"]

WORKDIR /
RUN source /root/.bashrc \
    && pip3.8 install pandas==1.1.0 \
    && pip3.8 install flbenchmark \
    && pip3.8 install scikit-learn==1.0.1

COPY ./src/unifed/frameworks/paddlefl/commit.patch /usr/local/lib/python3.8/site-packages/paddle_fl

WORKDIR /usr/local/lib/python3.8/site-packages/paddle_fl
RUN git apply --whitespace=warn commit.patch

RUN source /root/.bashrc \
    && pip3.8 install pytest

WORKDIR /test

RUN source /root/.bashrc \
    && pip3.8 install --upgrade pip \
    && pip3.8 install RUST

RUN source /root/.bashrc \
    && pip3.8 install protobuf==3.19.0 \
    && pip3.8 install colink

COPY src /test/src/
COPY setup.py /test/

RUN source /root/.bashrc \
    && pip3.8 install -e .
    

VOLUME ../data /data

WORKDIR /test
COPY test /test/test/
COPY src/unifed/frameworks/paddlefl/sub_pack /test/
# COPY demo/download.py /test


# COPY demo/fl_master.py demo/fl_scheduler.py demo/fl_server.py demo/fl_trainer.py /test/
# COPY demo/run.sh demo/stop.sh demo/utils.py demo/my_model.py demo/get_report.py /test/

# COPY demo/data_loader.py demo/leaf_utils.py /test/

# ENTRYPOINT ["/bin/bash", "run.sh"]
ENTRYPOINT [ "unifed-paddlefl" ]