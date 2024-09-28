# ifndef KALMAN
# define KALMAN

# include <opencv2/opencv.hpp>

namespace Buff
{
  class Kalman
  {
  public:
    Kalman();
    // 预测
    cv::Mat predict() {return kf.predict();}
    // 估计
    cv::Mat correct(cv::Mat z) {return kf.correct(z);}
  private:
    // 卡尔曼滤波器
    cv::KalmanFilter kf;

    int stateSize;
    int measureSize;
    int controlSize;

    cv::Mat A;
    cv::Mat H;
    cv::Mat Q;
    cv::Mat R;
    // 初始状态协方差
    cv::Mat P;
  };
}

# endif