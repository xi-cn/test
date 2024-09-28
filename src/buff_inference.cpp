#include "buff_inference/buff_inference.h"
#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Core>
#include "buff_inference/matplotlibcpp.h"
#include <ceres/ceres.h>

void buffInferenceNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    try  
    {  
        raw_image = cv_bridge::toCvShare(msg, "bgr8")->image;  
        preprocessImage();
    }  
    catch (const cv_bridge::Exception& e)  
    {  
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());  
        return;
    }

    ov::Tensor output = inference();
    postprocess(output);

    std::chrono::high_resolution_clock::time_point end_ = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start).count() / 1e9;
    // std::cout << "inferece: " << duration << "s\n";

    cv::resize(raw_image, raw_image, cv::Size(raw_image.cols/2, raw_image.rows/2));
    cv::imshow("image", raw_image);
    cv::waitKey(1);
}

void buffInferenceNode::initOVModel()
{
    // compile ov model
    model = core.compile_model("/home/a/Buff_Vision/src/buff_inference/model/best.xml", "CPU");
    // create ov request
    request = model.create_infer_request();
}

void buffInferenceNode::preprocessImage()
{
    int hight = raw_image.rows;
    int width = raw_image.cols;
    // 缩放比例
    scaleRate = hight > width ? 640.0 / hight : 640.0 / width;

    int new_hight = hight * scaleRate;
    int new_width = width * scaleRate;

    cv::Mat resized_img;
    cv::resize(raw_image, resized_img, cv::Size(new_width, new_hight));

    resized_img.copyTo(infer_image(cv::Rect(0, 0, new_width, new_hight)));

    // 将图像数据转为float32
    cv::Mat f32_image;
    infer_image.convertTo(f32_image, CV_32FC3, 1.0 / 255);

    int rc = f32_image.channels();
    for (int i = 0; i < rc; ++i) {
        cv::extractChannel(f32_image, cv::Mat(640, 640, CV_32FC1, imaga_data.data() + i * 640 * 640), i);
    }
}

ov::Tensor buffInferenceNode::inference()
{
    auto input_port = model.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), imaga_data.data());
    // memcpy(input_tensor.data(), infer_image.data, infer_image.elemSize() * infer_image.total());
    // Set input tensor for model with one input
    request.set_input_tensor(input_tensor);
    // Start inference
    request.infer();
    // Get the inference result
    ov::Tensor output = request.get_output_tensor(0);

    return output;
}

void buffInferenceNode::postprocess(ov::Tensor &output)
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
            class_ids.push_back(class_id.x);
            class_scores.push_back(max_score);
            float cx = result.at<float>(i, 0);
            float cy = result.at<float>(i, 1);
            float cw = result.at<float>(i, 2);
            float ch = result.at<float>(i, 3);

            int left = (cx - cw / 2) / scaleRate;
            int top = (cy - ch / 2) / scaleRate;
            int width = cw / scaleRate;
            int hight = ch / scaleRate;
            boxes.push_back(cv::Rect(left, top, width, hight));
            mask_confs.push_back(result.row(i).colRange(6, 21));
        }
    }

    std::vector<int> indices;
    // 对超过置信度的结果进行非极大值抑制
    cv::dnn::NMSBoxes(boxes, class_scores, conf_threshold, nms_threshold, indices);

    for (int &index : indices){
        cv::Point p1(int(mask_confs[index].at<float>(0)/scaleRate), int(mask_confs[index].at<float>(1)/scaleRate));
        cv::Point p2(int(mask_confs[index].at<float>(3)/scaleRate), int(mask_confs[index].at<float>(4)/scaleRate));
        cv::Point p3(int(mask_confs[index].at<float>(6)/scaleRate), int(mask_confs[index].at<float>(7)/scaleRate));
        cv::Point p4(int(mask_confs[index].at<float>(9)/scaleRate), int(mask_confs[index].at<float>(10)/scaleRate));
        cv::Point p5(int(mask_confs[index].at<float>(12)/scaleRate), int(mask_confs[index].at<float>(13)/scaleRate));
        cv::line(raw_image, p1, p2, cv::Scalar(255, 0, 0), 3);
        cv::line(raw_image, p2, p4, cv::Scalar(255, 0, 0), 3);
        cv::line(raw_image, p4, p5, cv::Scalar(255, 0, 0), 3);
        cv::line(raw_image, p5, p1, cv::Scalar(255, 0, 0), 3);
        cv::circle(raw_image, p1, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::circle(raw_image, p2, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::circle(raw_image, p4, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::circle(raw_image, p5, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::circle(raw_image, p3, 5, cv::Scalar(0, 255, 0), cv::FILLED);

        std::vector<cv::Point2f> imagePoints;
        imagePoints.push_back(cv::Point2f(mask_confs[index].at<float>(0)/scaleRate, mask_confs[index].at<float>(1)/scaleRate));
        imagePoints.push_back(cv::Point2f(mask_confs[index].at<float>(3)/scaleRate, mask_confs[index].at<float>(4)/scaleRate));
        imagePoints.push_back(cv::Point2f(mask_confs[index].at<float>(6)/scaleRate, mask_confs[index].at<float>(7)/scaleRate));
        imagePoints.push_back(cv::Point2f(mask_confs[index].at<float>(9)/scaleRate, mask_confs[index].at<float>(10)/scaleRate));
        imagePoints.push_back(cv::Point2f(mask_confs[index].at<float>(12)/scaleRate, mask_confs[index].at<float>(13)/scaleRate));

        pnp(imagePoints);
    }
}

void buffInferenceNode::pnp(std::vector<cv::Point2f> imagePoints)
{
    // pnp求解坐标转换
    // cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    
    // cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    rclcpp::Time start = this->get_clock()->now();
    makeTransforms();
    rclcpp::Time end_ = this->get_clock()->now();
    rclcpp::Duration d = end_ - start;
}

void buffInferenceNode::makeTransforms()
{
    // 发布由camera到待击大扇页fan1的坐标转换
    rclcpp::Time now = this->get_clock()->now();
    geometry_msgs::msg::TransformStamped t;

    t.header.stamp = now;
    t.header.frame_id = "camera";
    t.child_frame_id = "fan1";

    t.transform.translation.x = tvec.at<double>(0);
    t.transform.translation.y = tvec.at<double>(1);
    t.transform.translation.z = tvec.at<double>(2);

    tf2::Quaternion q;
    // 相机到扇页的旋转
    q.setRotation(tf2::Vector3(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)),
            cv::norm(rvec));

    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();
    // 发布tf
    tf_pub->sendTransform(t);

    // 相机到扇页中心
    Eigen::Quaterniond camera_fan_q(q.w(), q.x(), q.y(), q.z());
    Eigen::Vector3d camera_fan_v(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // 扇页到能量机关中心
    Eigen::Quaterniond fan_center_q(1, 0, 0, 0);
    Eigen::Vector3d fan_center_v(0, 0.7, 0.05);

    // 相机到能量机关中心
    Eigen::Quaterniond camera_center_q = camera_fan_q * fan_center_q;
    Eigen::Vector3d camera_center_v = camera_fan_q * fan_center_v + camera_fan_v;

    cv::Mat tvec_ = (cv::Mat_<double>(1, 3) << camera_center_v.x(), camera_center_v.y(), camera_center_v.z());
    cv::Mat centerPoint = (cv::Mat_<double>(1, 3) << 0, 0, 0);
    cv::Mat rvec_ = (cv::Mat_<double>(1, 3) << 0, 0, 0);
    cv::Mat imagePoint = cv::Mat_<double>(1, 2);

    // std::cout << "tvec_" << tvec_ << "\n";
    // std::cout << "tvec" << tvec << "\n\n";

    // 将能量机关中心映射到图片上
    cv::projectPoints(centerPoint, rvec_, tvec_, cameraMatrix, distCoeffs, imagePoint);

    cv::Point p(imagePoint.at<double>(0), imagePoint.at<double>(1));
    cv::circle(raw_image, p, 5, cv::Scalar(100, 90, 190), cv::FILLED);

    buffFan fan;
    fan.camera_fan_q = camera_fan_q;
    fan.camera_fan_v = camera_fan_v;
    fan.time_stamp = this->get_clock()->now();

    if (!fans.empty()) {
        // 时间间隔
        rclcpp::Duration duration = fan.time_stamp - fans.back().time_stamp;
        // std::cout << duration.nanoseconds() / 1e9 << "\n";
        
        // 上一次->相机->此次 旋转
        Eigen::Quaterniond pre_now_q = fans.back().camera_fan_q.inverse() * fan.camera_fan_q;
        // 求解旋转角度
        double angle = Eigen::AngleAxisd(pre_now_q).angle();
        // 时间间隔 纳秒
        double time_ = duration.nanoseconds() / 1e9;

        // std::cout << time_<< "\n";

        if (time_recorder.empty()) {
            time_recorder.push_back(time_);
        }
        else {
            time_recorder.push_back(time_ + time_recorder.back());
        }

        angles.push_back(angle / time_);
        if (angles.size() > 50) {
            time_recorder.erase(time_recorder.begin());
            angles.erase(angles.begin());
        }

        // 卡尔曼滤波
        cv::Mat z = (cv::Mat_<double>(1, 1) << angle / time_);
        // 预测
        kf.predict();
        // 估计
        cv::Mat estimateState = kf.correct(z);
        
        filter_angles.push_back(estimateState.at<double>(0, 0));
        if (filter_angles.size() > 50) {
            filter_angles.erase(filter_angles.begin());
        }

        // 初始化参数  
        double param[] = {0.9125, 1.942, 0};

        // 构建问题  
        ceres::Problem problem;  

        for (int i = 0; i < time_recorder.size(); ++i) {  
            ceres::CostFunction* cost_function =  
                new ceres::AutoDiffCostFunction<SinModel, 1, 3>(  
                    new SinModel(time_recorder[i], filter_angles[i]));  
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1.0), param);  
        }  

        // 添加范围限制
        problem.SetParameterLowerBound(param, 0, 0.780);
        problem.SetParameterUpperBound(param, 0, 1.045);

        problem.SetParameterLowerBound(param, 1, 1.884);
        problem.SetParameterUpperBound(param, 1, 2.000);

        problem.SetParameterLowerBound(param, 2, -M_PI);
        problem.SetParameterUpperBound(param, 2, M_PI);

        // 配置求解器并求解  
        ceres::Solver::Options options;  
        options.linear_solver_type = ceres::DENSE_QR;  
        options.minimizer_progress_to_stdout = false;  
        options.gradient_tolerance = 0.1;
        ceres::Solver::Summary summary;  
        ceres::Solve(options, &problem, &summary);  
        // 输出结果  
        // std::cout << summary.BriefReport() << "\n";  
        std::cout << "Estimated a : " << param[0] << "\n";  
        std::cout << "Estimated w : " << param[1] << "\n"; 
        std::cout << "Estimated theta : " << param[2] << "\n"; 

        // 计算拟合后的速度
        std::vector<double> fit_speed;
        fit_speed.resize(time_recorder.size());
        for (int i = 0; i < fit_speed.size(); ++i) {
            fit_speed[i] = param[0]*sin(param[1]*time_recorder[i] + param[2]) + 2.090 - param[0];
        }


        // // 绘制图像
        // matplotlibcpp::clf();
        // matplotlibcpp::subplot2grid(2, 1, 0);
        // matplotlibcpp::scatter(time_recorder, angles);
        // matplotlibcpp::ylim(0, 3);

        // matplotlibcpp::subplot2grid(2, 1, 1);
        // matplotlibcpp::scatter(time_recorder, filter_angles);
        // matplotlibcpp::ylim(0, 3);

        // matplotlibcpp::plot(time_recorder, fit_speed);
        // matplotlibcpp::pause(0.001);

        // a*sin(wt+theta)+b
        // 积分计算角度 dtheta = -(a/w)*cos(w*dt+theta) + b*dt
        // 假设时间为0.1s
        double dt = 0.3;
        double dtheta = -(param[0] / param[1]) * (cos(param[1] * (time_ + dt) + param[2]) - cos(param[1] * (time_) + param[2])) + (2.090 - param[0]) * dt;
        std::cout << "dtheta: " << dtheta << "\n";

        // fan center to next fan center
        Eigen::Vector3d fan_next_v(sin(dtheta)*0.7, (1-cos(dtheta))*0.7, 0);
        
        Eigen::Vector3d camera_next_v = camera_fan_q * fan_next_v + camera_fan_v;
        


        // tf2::Vector3 rvec_next(pre_now_q.x(), pre_now_q.y(), pre_now_q.z());
        // tf2::Quaternion q_next;
        // // 此次到预测的旋转
        // q_next.setRotation(rvec_next.normalize(), dtheta);
        // // 转换到Eigen
        // Eigen::Quaterniond eigen_next(q_next.getW(), q_next.getX(), q_next.getY(), q_next.getZ()); 

        // // 相机到下一次能量机关中心的旋转
        // Eigen::Quaterniond next_center_q = camera_center_q*eigen_next;

        // // 相机到下一次扇页中心的转换
        // Eigen::Quaterniond next_fan_q = next_center_q;
        // // q3(q2(X + v1)) + v3
        // // q3*q2X + q3q2v1 + v3
        // Eigen::Vector3d next_fan_v = next_fan_q * eigen_next * (-fan_center_v) + camera_center_v;

        // // 相机到下一次的扇页的旋转轴
        // tf2::Vector3 camera_next_rvec(next_fan_q.x(), next_fan_q.y(), next_fan_q.z());
        // camera_next_rvec = camera_next_rvec.normalize();
        // cv::Mat rvec_ = (cv::Mat_<double>(1, 3) << camera_next_rvec.x(), camera_next_rvec.y(), camera_next_rvec.z());
        // // 相机到下一次的扇页的平移
        // cv::Mat tvec_ = (cv::Mat_<double>(1, 3) << next_fan_v.x(), next_fan_v.y(), next_fan_v.z());
        cv::Mat centerPoint = (cv::Mat_<double>(1, 3) << fan_next_v.x(), fan_next_v.y(), 0);
        
        cv::Mat predictPoint = cv::Mat_<double>(1, 2);

        // 将相机中心映射到图片上
        cv::projectPoints(centerPoint, rvec, tvec, cameraMatrix, distCoeffs, predictPoint);

        cv::Point p(predictPoint.at<double>(0), predictPoint.at<double>(1));
        std::cout << "point: " << p << "\n";
        cv::circle(raw_image, p, 10, cv::Scalar(180, 90, 100), cv::FILLED);




        // // 发布由camera到待击大扇页fan1的坐标转换
        // rclcpp::Time now = this->get_clock()->now();
        // geometry_msgs::msg::TransformStamped t_pred;

        // t_pred.header.stamp = now;
        // t_pred.header.frame_id = "camera";
        // t_pred.child_frame_id = "fan_pre";

        // t_pred.transform.translation.x = tvec_.at<double>(0);
        // t_pred.transform.translation.y = tvec_.at<double>(1);
        // t_pred.transform.translation.z = tvec_.at<double>(2);

        // tf2::Quaternion q;
        // // 相机到扇页的旋转
        // q.setRotation(tf2::Vector3(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)),
        //         cv::norm(rvec));

        // t_pred.transform.rotation.x = next_fan_q.x();
        // t_pred.transform.rotation.y = next_fan_q.y();
        // t_pred.transform.rotation.z = next_fan_q.z();
        // t_pred.transform.rotation.w = next_fan_q.w();
        // // 发布tf
        // tf_pub->sendTransform(t_pred);
    }

    fans.push_back(fan);
  }

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<buffInferenceNode>());
    rclcpp::shutdown();
    return 0;
}