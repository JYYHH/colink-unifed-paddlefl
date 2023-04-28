docker run --rm --net=host -v $PWD/config.json:/test/config.json -it -v $PWD/log:/test/log -v $PWD/../data:/data flbenchmark/paddlefl /bin/bash
# docker run --rm --gpus all -v $PWD/config.json:/test/config.json -v $PWD/../data:/data -v $PWD/log:/test/log flbenchmark/paddlefl
