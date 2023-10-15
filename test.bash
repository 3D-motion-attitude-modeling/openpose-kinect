cd build
make -j`nproc`
cd ..
./build/examples/openpose/openpose.bin --num_gpu 0 --kinect_camera --kinect_camera_index 0

