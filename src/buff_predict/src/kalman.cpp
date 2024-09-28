# include "buff_predict/kalman.h"

namespace Buff
{
  Kalman::Kalman()
  {
    stateSize = 2;
    measureSize = 1; 
    controlSize = 0; 

    A = (cv::Mat_<double>(2, 2) << 1, 0, 
                                   0, 1);
    H = (cv::Mat_<double>(1, 2) << 1, 0);
    Q = (cv::Mat_<double>(2, 2) << 1, 0,
                                   0, 1) * 0.001;
    R = (cv::Mat_<double>(1, 1) << 1) * 0.001;
    P = cv::Mat::eye(stateSize, stateSize, CV_64F) * 1;

    kf = cv::KalmanFilter(stateSize, measureSize, controlSize, CV_64F);
    kf.transitionMatrix = A;
    kf.measurementMatrix = H;
    kf.processNoiseCov = Q;
    kf.measurementNoiseCov = R;
    // 初始状态估计
    kf.statePre.at<double>(0) = 0;
    kf.statePre.at<double>(1) = 0;
    // 初始状态协方差
    kf.errorCovPre = P;
  }
} // namespace Buff
