# include "buff_detect/buff_detect_node.h"


namespace Buff
{
  //geometry格式转Eigen
  Eigen::Quaterniond BuffDetectNode::geometry2eigen(geometry_msgs::msg::Quaternion q)
  {
    Eigen::Quaterniond eigen_q(q.w, q.x, q.y, q.z);
    return eigen_q;
  }
  //geometry格式转Eigen
  Eigen::Vector3d BuffDetectNode::geometry2eigen(geometry_msgs::msg::Point v)
  {
    Eigen::Vector3d eigen_v(v.x, v.y, v.z);
  }

  // Point2f转Point
  cv::Point BuffDetectNode::point2f_2point(cv::Point2f p)
  {
    return cv::Point(p.x, p.y);
  }

  // 展示检测图像
  void BuffDetectNode::showDetectImage(cv::Mat image, KeyPoints& key_points)
  {
    if (key_points.is_detected){
      cv::Point p1 = point2f_2point(key_points.point[0]);
      cv::Point p2 = point2f_2point(key_points.point[1]);
      cv::Point p3 = point2f_2point(key_points.point[2]);
      cv::Point p4 = point2f_2point(key_points.point[3]);
      cv::Point p5 = point2f_2point(key_points.point[4]);
      cv::line(image, p1, p2, cv::Scalar(255, 0, 0), 3);
      cv::line(image, p2, p4, cv::Scalar(255, 0, 0), 3);
      cv::line(image, p4, p5, cv::Scalar(255, 0, 0), 3);
      cv::line(image, p5, p1, cv::Scalar(255, 0, 0), 3);
      cv::circle(image, p1, 5, cv::Scalar(0, 255, 0), cv::FILLED);
      cv::circle(image, p2, 5, cv::Scalar(0, 255, 0), cv::FILLED);
      cv::circle(image, p4, 5, cv::Scalar(0, 255, 0), cv::FILLED);
      cv::circle(image, p5, 5, cv::Scalar(0, 255, 0), cv::FILLED);
      cv::circle(image, p3, 5, cv::Scalar(0, 255, 0), cv::FILLED);

      // cv::addText(image, "detected", cv::Point(0, 0), 5);
    }
    else {
      // cv::addText(image, "lost", cv::Point(0, 0), 5);
    }
    
    cv::imshow("detect image", image);
    cv::waitKey(10);
  }

  // 图像消息回调函数
  void BuffDetectNode::imageCallback(sensor_msgs::msg::Image::SharedPtr msg)
  {
    // 将图像转换为bgr8格式
    cv::Mat raw_image = cv_bridge::toCvShare(msg, "bgr8")->image;  
    // 存储检测结果
    KeyPoints key_points;
    // 检测图像
    if (detector->detect(raw_image, key_points)) {
      // pnp计算坐标转换
      global_msg::msg::DetectMsg detect_msg = pnp->run(raw_image, key_points);

      // 与图像使用相同的时间戳
      detect_msg.header.set__stamp(msg->header.stamp);

      
      // 更新相机到世界坐标系下的变换
      detect_msg.camera2world_q.set__x(0);
      detect_msg.camera2world_q.set__y(0);
      detect_msg.camera2world_q.set__z(0);
      detect_msg.camera2world_q.set__w(1);

      detect_msg.camera2world_v.set__x(0);
      detect_msg.camera2world_v.set__y(0);
      detect_msg.camera2world_v.set__z(0);

      // 更新扇叶中心到世界坐标系的转换
      // 转换到Eigen格式便于计算
      Eigen::Quaterniond buff2world_eigen_q = geometry2eigen(detect_msg.camera2world_q) * geometry2eigen(detect_msg.buff2camera_q);
      Eigen::Vector3d buff2world_eigen_v = buff2world_eigen_q * geometry2eigen(detect_msg.buff2camera_v) + geometry2eigen(detect_msg.camera2world_v);
      // 赋值到结果中
      detect_msg.buff2world_q.set__w(buff2world_eigen_q.w());
      detect_msg.buff2world_q.set__x(buff2world_eigen_q.x());
      detect_msg.buff2world_q.set__y(buff2world_eigen_q.y());
      detect_msg.buff2world_q.set__z(buff2world_eigen_q.z());

      detect_msg.buff2world_v.set__x(buff2world_eigen_v.x());
      detect_msg.buff2world_v.set__y(buff2world_eigen_v.y());
      detect_msg.buff2world_v.set__z(buff2world_eigen_v.z());

      // 发布消息
      detect_pub->publish(detect_msg);
    }

    // 可视化推理结果
    if (true) {
      showDetectImage(raw_image, key_points);
    }
  }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Buff::BuffDetectNode>());
    rclcpp::shutdown();
}