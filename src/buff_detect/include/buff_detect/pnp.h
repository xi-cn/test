# ifndef PNP
# define PNP

# include <iostream>
# include <opencv2/opencv.hpp>
# include <global_msg/msg/detect_msg.hpp>
# include <tf2/LinearMath/Quaternion.h>
# include "buff_detect/detect.h"

namespace Buff
{
  class PnP
  {
    public:
      // 构造函数 相机参数地址
      PnP(std::string path);

      // pnp计算 图像 关键点
      global_msg::msg::DetectMsg run(cv::Mat &img, KeyPoints& key_points);

    private:
      // 相机内参  
      cv::Mat cameraMatrix;
      // 畸变系数  
      cv::Mat distCoeffs;
      // 世界坐标
      std::vector<cv::Point3f> objectPoints;

  };
} // namespace BUFF

#endif