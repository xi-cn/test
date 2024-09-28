#ifndef BUFF_INFERENCE
#define BUFF_INFERENCE

#include <openvino/openvino.hpp>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <chrono>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <list>
#include <ceres/ceres.h>


class buffFan
{
public:
    rclcpp::Time time_stamp;
    Eigen::Quaterniond camera_fan_q;
    Eigen::Vector3d camera_fan_v;
    Eigen::Vector3d gimbol_camera_v;
    Eigen::Quaterniond gimbol_camera_q;

    buffFan(){
        gimbol_camera_v = Eigen::Vector3d::Zero();
        gimbol_camera_q = Eigen::Quaterniond::Identity();
    }
};


struct SinModel {  
    SinModel(const double& t, const double& speed)  
        : x(t), y(speed){}  
  
    template <typename T>  
    bool operator()(const T* const param, T* residuals) const {  
        residuals[0] = T(y) - (param[0] * ceres::sin(param[1] * T(x) + T(param[2])) + 2.09 - param[0]);  
        return true;  
    }  
  
    const double x;  
    const double y;  
};



class buffInferenceNode : public rclcpp::Node
{
public:
    buffInferenceNode() : Node("buffInferenceNode")
    {
        // init ov model
        imaga_data.resize(640 * 640 * 3);
        conf_threshold = 0.8;
        nms_threshold = 0.5;
        initOVModel();
        infer_image = cv::Mat::zeros(cv::Size(640, 640), CV_8UC3);

        img_sub = this->create_subscription<sensor_msgs::msg::Image>(
            "/raw_image", 10, std::bind(&buffInferenceNode::imageCallback, this, std::placeholders::_1));
        tf_pub = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        objectPoints.push_back(cv::Point3f(-0.1125, -0.027, 0));
        objectPoints.push_back(cv::Point3f(-0.1125, 0.027, 0));
        objectPoints.push_back(cv::Point3f(0, 0.7, 0.05));
        objectPoints.push_back(cv::Point3f(0.1125, 0.027, 0));
        objectPoints.push_back(cv::Point3f(0.1125, -0.027, 0));

        // 相机内参  
        cameraMatrix = (cv::Mat_<double>(3, 3) << 647.2255218, 0, 321.0205024,
                                                      0, 646.8296461, 254.68725356,
                                                      0, 0, 1);  
        // 畸变系数  
        distCoeffs = (cv::Mat_<double>(1, 5) << -0.07703471, 0.06098453, 0.00027992, -0.00054497, 0.26117949);

        cameraMatrix = (cv::Mat_<double>(3, 3) << 1675.042871, 0, 623.481121,
                                                      0, 1674.696037, 556.002821,
                                                      0, 0, 1);  
        // 畸变系数  
        distCoeffs = (cv::Mat_<double>(1, 5) << -0.06719080320827164, 0.08771589756714641, 0.0, 0.0009056377800116876, -0.004604447207374161);

        rvec = (cv::Mat_<double>(1, 3) << 0.0, 0.0, 0.0);
        tvec = (cv::Mat_<double>(1, 3) << 0.0, 0.0, 0.0);

        kf = cv::KalmanFilter(stateSize, measureSize, controlSize, CV_64F);
        kf.transitionMatrix = A;
        kf.measurementMatrix = H;
        kf.processNoiseCov = Q;
        kf.measurementNoiseCov = R;
        // 初始状态估计
        kf.statePre.at<double>(0) = 0;
        kf.statePre.at<double>(1) = 0;
        // 初始状态协方差
        cv::Mat P = cv::Mat::eye(stateSize, stateSize, CV_64F) * 1;
        kf.errorCovPre = P;

        google::InitGoogleLogging("ceres");  

    }

private:
    // init openvino model
    void initOVModel();

    // raw image callback function
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    // preprocess the image
    void preprocessImage();

    // post process
    void postprocess(ov::Tensor &output);

    // infer the image
    ov::Tensor inference();

    // caculate position
    void pnp(std::vector<cv::Point2f> imagePoints);

    // send the transform
    void makeTransforms();

    // subscribe the raw image
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
    // publish the rotate speed
    // publish the transform from camera to fan1
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_pub;

    // 卡尔曼滤波器
    int stateSize = 2;
    int measureSize = 1; 
    int controlSize = 0; 

    cv::KalmanFilter kf;
    cv::Mat A = (cv::Mat_<double>(2, 2) << 1, 0, 
                                            0, 1);
    cv::Mat H = (cv::Mat_<double>(1, 2) << 1, 0);
    cv::Mat Q = (cv::Mat_<double>(2, 2) << 1, 0,
                                            0, 1) * 0.001;
    cv::Mat R = (cv::Mat_<double>(1, 1) << 1) * 0.001;


    ov::Core core;
    ov::CompiledModel model;
    ov::InferRequest request;
    cv::Mat infer_image;
    cv::Mat raw_image;
    std::vector<float> imaga_data;
    double conf_threshold;
    double nms_threshold;
    double scaleRate;

    std::vector<cv::Point3f> objectPoints;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat rvec;
    cv::Mat tvec;

    std::list<buffFan> fans;

    // 角度和时间
    std::vector<double> angles;
    std::vector<double> time_recorder;
    std::vector<double> filter_angles;
};

#endif