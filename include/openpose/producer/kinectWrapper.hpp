#ifndef OPENPOSE_PRODUCER_KINECT_WRAPPER_HPP
#define OPENPOSE_PRODUCER_KINECT_WRAPPER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    /**
     * KinectWrapper imitates the class of SpinnakerWrapper. It decouples the final interface (meant to imitates
     * cv::VideoCapture) from the kinect SDK wrapper.
     */
    int32_t color_exposure_usec = 8000;  // somewhat reasonable default exposure time
    int32_t powerline_freq = 1;          // default to a 50 Hz powerline   

    class OP_API KinectWrapper
    {
    public:
        /**
         * Constructor of KinectWrapper. It opens all the available Kinect cameras
         * cameraIndex = -1 means that all cameras are taken
         */
        explicit KinectWrapper(const std::string& cameraParameterPath, const Point<int>& cameraResolution,
                                  const bool undistortImage, const int cameraIndex = -1);

        virtual ~KinectWrapper();

        std::vector<Matrix> getRawFrames();

        /**
         * Note: The camera parameters are only read if undistortImage is true. This should be changed to add a
         * new bool flag in the constructor, e.g., readCameraParameters
         */
        std::vector<Matrix> getCameraMatrices() const;

        std::vector<Matrix> getCameraExtrinsics() const;

        std::vector<Matrix> getCameraIntrinsics() const;

        Point<int> getResolution() const;

        bool isOpened() const;

        void release();

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplKinectWrapper;
        std::shared_ptr<ImplKinectWrapper> upImpl;

        DELETE_COPY(KinectWrapper);
    };
}

#endif // OPENPOSE_PRODUCER_SPINNAKER_WRAPPER_HPP
