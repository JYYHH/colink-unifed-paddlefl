cd ..
mkdir server0 && cd server0
cp -r ../test/* .
unifed-paddlefl-workload server 0 127.0.0.1
cd ..
mkdir client0 && cd client0
cp -r ../test/* .
unifed-paddlefl-workload client 0 127.0.0.1 &
cd ..
mkdir client1 && cd client1
cp -r ../test/* .
unifed-paddlefl-workload client 1 127.0.0.1 &
cd ..