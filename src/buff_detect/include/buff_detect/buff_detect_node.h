# ifndef BUFF_DETECT_NODE
# define BUFF_DETECT_NODE

# include <rclcpp/rclcpp.hpp>
# include <sensor_msgs/msg/image.hpp>
# include <global_msg/msg/detect_msg.hpp>
# include <cv_bridge/cv_bridge.h>
# include <opencv2/opencv.hpp>
# include <Eigen/Core>
# include <Eigen/Geometry>

# include <buff_detect/pnp.h>
# include <buff_detect/detect.h>


namespace Buff
{
  class BuffDetectNode : public rclcpp::Node
  {
  public:
    BuffDetectNode() : Node("BuffDetectNode")
    {
      // 发布节点
      detect_pub = this->create_publisher<global_msg::msg::DetectMsg>("/buff_detect", 10);
      // 接收节点
      img_sub = this->create_subscription<sensor_msgs::msg::Image>(
            "/raw_image", 10, std::bind(&BuffDetectNode::imageCallback, this, std::placeholders::_1));

      // 初始化pnp解算模型和图像识别模型
      pnp = std::make_shared<PnP>("");
      detector = std::make_shared<BuffDetector>("");
    }
  
  private:
    void imageCallback(sensor_msgs::msg::Image::SharedPtr msg);

    // 展示检测图像
    void showDetectImage(cv::Mat image, KeyPoints& key_points);

    // geometry格式转Eigen
    Eigen::Quaterniond geometry2eigen(geometry_msgs::msg::Quaternion q);
    Eigen::Vector3d geometry2eigen(geometry_msgs::msg::Point v);
    // Point2f转Point
    cv::Point point2f_2point(cv::Point2f p);

    // 发布和接受节点
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
    rclcpp::Publisher<global_msg::msg::DetectMsg>::SharedPtr detect_pub;

    // pnp规划器
    std::shared_ptr<PnP> pnp;
    // 推理器
    std::shared_ptr<BuffDetector> detector;
  };
}

# endif