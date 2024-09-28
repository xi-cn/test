# include "buff_detect/detect.h"

namespace Buff
{
  // 构造函数 
  // path: 模型路径  置信度 nms阈值
  BuffDetector::BuffDetector(std::string path, double conf, double nms)
  {
    // 加载编译模型
    model = core.compile_model("/home/dhu/DIODE_Buff/src/buff_detect/model/best.xml", "GPU");
    // 创建推理请求
    request = model.create_infer_request();
    // 初始化图像数据大小
    imaga_data.resize(640 * 640 * 3);
    // 初始化推理图像大小
    infer_image = cv::Mat::zeros(cv::Size(640, 640), CV_8UC3);

    conf_threshold = conf;
    nms_threshold = nms_threshold;
  }

  // 检测图像
  bool BuffDetector::detect(cv::Mat raw_image, KeyPoints& result)
  {
    // 预处理
    preprocess(raw_image);
    // 推理
    inference();
    // 后处理
    result = postprocess();

    return result.is_detected;
  }

  // 图像预处理
  void BuffDetector::preprocess(cv::Mat& raw_image)
  {
    // 原始图像宽高
    int hight = raw_image.rows;
    int width = raw_image.cols;
    // 缩放比例
    scaleRate = hight > width ? 640.0 / hight : 640.0 / width;

    // 缩放后的图像宽高
    int new_hight = hight * scaleRate;
    int new_width = width * scaleRate;

    // 缩放图像
    cv::Mat resized_img;
    cv::resize(raw_image, resized_img, cv::Size(new_width, new_hight));

    // 拷贝有效数据
    resized_img.copyTo(infer_image(cv::Rect(0, 0, new_width, new_hight)));

    // 将图像数据转为float32
    cv::Mat f32_image;
    infer_image.convertTo(f32_image, CV_32FC3, 1.0 / 255);
    // 图像拷贝到推理图像数据中
    int rc = f32_image.channels();
    for (int i = 0; i < rc; ++i) {
        cv::extractChannel(f32_image, cv::Mat(640, 640, CV_32FC1, imaga_data.data() + i * 640 * 640), i);
    }
  }

  // 后处理
  KeyPoints BuffDetector::postprocess()
  {    
    cv::Mat result(output.get_shape()[1], output.get_shape()[2], CV_32F, output.data<float>());
    cv::transpose(result, result);

    // 记录识别物体的id、置信度、边框、关键点
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Mat> mask_confs;

    for (int i = 0; i < result.rows; ++i) {
      cv::Mat class_conf = result.row(i).colRange(4, 6);
      double max_score;
      cv::Point class_id;
      // 求解置信度最大的类别
      cv::minMaxLoc(class_conf, 0, &max_score, 0, &class_id);

      if (max_score > conf_threshold){
        // 类别id
        class_ids.push_back(class_id.x);
        // 类别的分
        class_scores.push_back(max_score);
        // 矩形框中点和宽高
        float cx = result.at<float>(i, 0);
        float cy = result.at<float>(i, 1);
        float cw = result.at<float>(i, 2);
        float ch = result.at<float>(i, 3);
        // 对矩形框进行缩放 到原始图像
        int left = (cx - cw / 2) / scaleRate;
        int top = (cy - ch / 2) / scaleRate;
        int width = cw / scaleRate;
        int hight = ch / scaleRate;

        boxes.push_back(cv::Rect(left, top, width, hight));
        mask_confs.push_back(result.row(i).colRange(6, 21));
      }
    }

    // 抑制后选取的结果索引
    std::vector<int> indices;
    // 对超过置信度的结果进行非极大值抑制
    cv::dnn::NMSBoxes(boxes, class_scores, conf_threshold, nms_threshold, indices);

    KeyPoints keypoints;

    // 选取结果
    for (int &index : indices){
      keypoints.is_detected = true;
      keypoints.point[0] = cv::Point2f(mask_confs[index].at<float>(0)/scaleRate, mask_confs[index].at<float>(1)/scaleRate);
      keypoints.point[1] = cv::Point2f(mask_confs[index].at<float>(3)/scaleRate, mask_confs[index].at<float>(4)/scaleRate);
      keypoints.point[2] = cv::Point2f(mask_confs[index].at<float>(6)/scaleRate, mask_confs[index].at<float>(7)/scaleRate);
      keypoints.point[3] = cv::Point2f(mask_confs[index].at<float>(9)/scaleRate, mask_confs[index].at<float>(10)/scaleRate);
      keypoints.point[4] = cv::Point2f(mask_confs[index].at<float>(12)/scaleRate, mask_confs[index].at<float>(13)/scaleRate);
    }

    return keypoints;
  }

  // 推理图像
  void BuffDetector::inference()
  {
    auto input_port = model.input();
    // 创建输入张量
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), imaga_data.data());
    // 设置输入张量到推理
    request.set_input_tensor(input_tensor);
    // 推理图像
    request.infer();
    // 获取推理结果
    output = request.get_output_tensor(0);
  }



}