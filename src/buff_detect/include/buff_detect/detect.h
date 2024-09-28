# ifndef DETECT
# define DETECT

# include <opencv2/opencv.hpp>
# include <openvino/openvino.hpp>

namespace Buff
{
  // 关键点
  struct KeyPoints
  {
    bool is_detected;
    std::vector<cv::Point2f> point;

    KeyPoints() { is_detected = false; point.resize(5);}
  };
  

  // 能量机关检测
  class BuffDetector
  {
  public:
    BuffDetector(std::string path, double conf=0.8, double nms=0.8);

    // 检测图像
    bool detect(cv::Mat raw_image, KeyPoints& result);
  private:

    // 预处理
    void preprocess(cv::Mat& raw_image);
    // 后处理
    KeyPoints postprocess();
    // 推理模型
    void inference();

    // openvino核心
    ov::Core core;
    // 推理模型
    ov::CompiledModel model;
    // 推理请求
    ov::InferRequest request;
    // 推理图像
    cv::Mat infer_image;
    // 图像数据 float类型
    std::vector<float> imaga_data;
    // 输出
    ov::Tensor output;

    // 置信度
    double conf_threshold;
    // nms抑制阈值
    double nms_threshold;
    // 图像缩放比例
    double scaleRate;
  };

} // namespace Detector


# endif