#ifndef OPENPOSE_PRODUCER_KINECT_READER_HPP
#define OPENPOSE_PRODUCER_KINECT_READER_HPP

#include <openpose/core/common.hpp>
#include <openpose/producer/producer.hpp>
#include <openpose/producer/kinectWrapper.hpp>

namespace op
{
    class OP_API KinectReader : public Producer
    {
    public:
        /**
         * Constructor of KinectReader. It opens all the available Kinect cameras
         */
        // cameraParametersPath is like 'models/cameraParameters/flir/'
        // 是否去畸变--undistortImage, 摄像头编号--cameraIndex（-1表示所有摄像头一同同步读取）
        // eplicit 指定构造函数或转换函数为显式, 即它不能用于隐式转换和复制初始化
        // const Point<int>& cameraResolution Kinect相机分辨率不能随便设置
        explicit KinectReader(const std::string& cameraParametersPath, const Point<int>& cameraResolution,
                            const bool undistortImage = true, const int cameraIndex = -1);

        virtual ~KinectReader();

        // 获取多摄像头的内外参数，在完成文件生成后可以借用Openpose内部CameraParameterReader来实现
        std::vector<Matrix> getCameraMatrices();

        std::vector<Matrix> getCameraExtrinsics();

        std::vector<Matrix> getCameraIntrinsics();

        std::string getNextFrameName();

        // 详情见producer.hpp
        bool isOpened() const;

        // End acquisition for each camera
        void release();

        double get(const int capProperty);

        void set(const int capProperty, const double value);

    private:
        //辅助实现的类
        KinectWrapper mKinectWrapper;

        // 保留
        Point<int> mResolution;
        unsigned long long mFrameNameCounter;

        // 获取多摄像头中第一个摄像头的帧图像
        Matrix getRawFrame();

        // 获取多摄像头同一时刻的帧图像
        std::vector<Matrix> getRawFrames();

        DELETE_COPY(KinectReader);
    };
}

#endif // OPENPOSE_PRODUCER_KINECT_READER_HPP
