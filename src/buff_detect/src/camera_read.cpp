#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <thread>

class CameraRead : public rclcpp::Node
{
public:
    CameraRead() : Node("CameraReadNode")
    {
        img_pub = this->create_publisher<sensor_msgs::msg::Image>("/raw_image", 10);
        std::thread([this]() { readVideo(); }).detach();
    }
private:
    void readVideo();

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub;
};

void CameraRead::readVideo()
{
    cv::VideoCapture cap("/home/dhu/DIODE_Buff/src/buff_detect/buff_video/video_red.MP4");
    cv::Mat src;
    sensor_msgs::msg::Image msg;
    while (cap.read(src) && rclcpp::ok()) {
        msg.width = src.cols;
        msg.height = src.rows;
        msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(src.step);
        msg.data.assign(src.datastart, src.dataend);
        msg.header.stamp = this->get_clock()->now();
        msg.encoding = "bgr8";
        img_pub->publish(std::move(msg));
        cv::waitKey(10);
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraRead>());
    rclcpp::shutdown();
}