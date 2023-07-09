#include "multi_calib.h"
#include "cout_style.h"

std::vector<std::string> camera_name{"front", "right", "rear", "left"};

MultiCalib::MultiCalib(std::vector<TripleSphereCamera> cameras, const std::vector<cv::Point3d>& worlds)
{
    worlds_ = worlds;
    int camera_num = cameras.size();
    int chessboard_num = cameras[0].has_chessboard().size();
    cameras_.resize(camera_num);
    chessboards_.resize(chessboard_num);
    // 粗略的估计所有相机和棋盘格的位姿
    // 首先计算相机的位姿
    for(int i = 0; i < camera_num; i++)
    {
        cv::Mat R, t;
        if(i == 0)
        {
            // 第一个相机假设为世界坐标系原点
            R = (cv::Mat_<double>(3,3) << 1,0,0, 0,1,0, 0,0,1);
            t = (cv::Mat_<double>(3,1) << 0,0,0);
        }
        else
        {
            // 根据上一个初始化的相机，计算这个相机所有可能的位姿
            std::vector<cv::Mat> Rs;
            std::vector<cv::Mat> ts;
            for(int j = 0; j < chessboard_num; j++)
            {
                if(cameras_[i-1].is_initial() == false)
                {
                    std::cout << "请按相邻顺序输出相机" << std::endl;
                    return;
                }
                if(cameras[i-1].has_chessboard(j) == false || cameras[i].has_chessboard(j) == false) continue;
                cv::Mat chess_Rt = cameras[i].Rt(j);
                cv::Mat chess_R_i, chess_t_i, chess_R_k, chess_t_k;
                Rt_to_R_t(chess_Rt, chess_R_i, chess_t_i);
                chess_Rt = cameras[i-1].Rt(j);
                Rt_to_R_t(chess_Rt, chess_R_k, chess_t_k);
                cv::Mat camera_R_k = cameras_[i-1].R();
                cv::Mat camera_t_k = cameras_[i-1].t();
                cv::Mat R_ik = chess_R_i * chess_R_k.t();
                cv::Mat t_ik = chess_t_i - R_ik * chess_t_k;
                Rs.push_back(R_ik * cameras_[i-1].R());
                ts.push_back(R_ik * cameras_[i-1].t() + t_ik);
            }
            // 从这些位姿中选取重投影误差最小的位姿
            double min_error = 1e10;
            double min_id = -1;
            for(int j = 0; j < Rs.size(); j++)
            {
                double error = 0;
                for(int k = 0; k < chessboard_num; k++)
                {
                    if(cameras[i-1].has_chessboard(k) == false || cameras[i].has_chessboard(k) == false) continue;
                    cv::Mat chess_Rt = cameras[i].Rt(k);
                    cv::Mat chess_R_i, chess_t_i, chess_R_k, chess_t_k;
                    Rt_to_R_t(chess_Rt, chess_R_i, chess_t_i);
                    cv::Mat camera_R_k = cameras_[i-1].R();
                    cv::Mat camera_t_k = cameras_[i-1].t();
                    cv::Mat R_ki = camera_R_k * Rs[j].t();
                    cv::Mat t_ki = camera_t_k - R_ki * ts[j];
                    chess_R_k = R_ki * chess_R_i;
                    chess_t_k = R_ki * chess_t_i + t_ki;
                    double e = cameras[i-1].ReprojectError(cameras[i-1].pixels()[k], worlds, chess_R_k, chess_t_k);
                    // std::cout << e << std::endl;
                    error += e;
                    chess_Rt = cameras[i-1].Rt(k);
                    Rt_to_R_t(chess_Rt, chess_R_k, chess_t_k);
                    R_ki = Rs[j] * camera_R_k.t();
                    t_ki = ts[j] - R_ki * camera_t_k;
                    chess_R_i = R_ki * chess_R_k;
                    chess_t_i = R_ki * chess_t_k + t_ki;
                    e = cameras[i].ReprojectError(cameras[i].pixels()[k], worlds, chess_R_i, chess_t_i);
                    // std::cout << e << std::endl;
                    error += e;
                }
                if(error < min_error)
                {
                    min_error = error;
                    min_id = j;
                }
            }
            R = Rs[min_id];
            t = ts[min_id];
        }
        cameras_[i] = MultiCalib_camera(cameras[i].cx(), cameras[i].cy(), cameras[i].fx(), cameras[i].fy(), cameras[i].xi(), cameras[i].lamda(), cameras[i].alpha(),
                                        cameras[i].b(), cameras[i].c(), 
                                        R, t, cameras[i].has_chessboard(), cameras[i].pixels());
    }
    // 然后计算棋盘的位姿
    for(int i = 0; i < chessboard_num; i++)
    {
        cv::Mat R, t;
        std::vector<int> camera_ids;
        for(int j = 0; j < cameras.size(); j++)
        {
            if(cameras[j].has_chessboard(i)) camera_ids.push_back(j);
        }
        if(camera_ids.size() == 0) continue;
        if(camera_ids.size() == 1)
        {
            cv::Mat camera_R = cameras_[camera_ids[0]].R();
            cv::Mat camera_t = cameras_[camera_ids[0]].t();
            cv::Mat chess_Rt = cameras[camera_ids[0]].Rt(i);
            cv::Mat chess_R, chess_t;
            Rt_to_R_t(chess_Rt, chess_R, chess_t);
            R = camera_R.t() * chess_R;
            t = camera_R.t() * (chess_t - camera_t);
        }
        else
        {
            // 计算棋盘所有可能的位姿
            std::vector<cv::Mat> Rs, ts;
            for(int j = 0; j < camera_ids.size(); j++)
            {
                cv::Mat camera_R = cameras_[camera_ids[j]].R();
                cv::Mat camera_t = cameras_[camera_ids[j]].t();
                cv::Mat chess_Rt = cameras[camera_ids[j]].Rt(i);
                cv::Mat chess_R, chess_t;
                Rt_to_R_t(chess_Rt, chess_R, chess_t);
                Rs.push_back(camera_R.t() * chess_R);
                ts.push_back(camera_R.t() * (chess_t - camera_t));
            }
            // 选取重投影最小的位姿
            double min_error = 1e10;
            double min_id = -1;
            for(int j = 0; j < Rs.size(); j++)
            {
                double error = 0;
                for(int k = 0; k < camera_ids.size(); k++)
                {
                    cv::Mat camera_R = cameras_[camera_ids[k]].R();
                    cv::Mat camera_t = cameras_[camera_ids[k]].t();
                    cv::Mat chess_R = camera_R * Rs[j];
                    cv::Mat chess_t = camera_R * ts[j] + camera_t;
                    double e = cameras[camera_ids[k]].ReprojectError(cameras[camera_ids[k]].pixels()[i], worlds, chess_R, chess_t);
                    error += e;
                }
                if(error < min_error)
                {
                    min_error = error;
                    min_id = j;
                }
            }
            R = Rs[min_id];
            t = ts[min_id];
        }
        chessboards_[i] = MultiCalib_chessboard(R, t);
    }
}

void MultiCalib::calibrate()
{
    int camera_num = cameras_.size();
    int chessboard_num = chessboards_.size();
    
    ceres::Problem problem;

    for(int m = 0; m < camera_num; m++)
    {
        std::vector<std::vector<cv::Point2d>> pixels = cameras_[m].pixels();
        for(int i = 0; i < chessboard_num; i++)
        {
            if(chessboards_[i].is_initial())
            {
                for(int j = 0; j < pixels[i].size(); j++)
                {
                    if(m == 0)
                    {
                        ReprojectionError *cost_function =
                            new ReprojectionError(pixels[i][j], worlds_[j]);

                        problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<
                                ReprojectionError,
                                2,  // num_residuals
                                6,6,9>(cost_function),
                            NULL,
                            cameras_[m].rt_.data(),
                            chessboards_[i].rt_.data(),
                            cameras_[m].intrinsic_.data()
                        );
                        problem.SetParameterBlockConstant(cameras_[m].rt_.data());
                    }
                    else
                    {
                        ReprojectionError *cost_function =
                            new ReprojectionError(pixels[i][j], worlds_[j]);

                        problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<
                                ReprojectionError,
                                2,  // num_residuals
                                6,6,9>(cost_function),
                            NULL,
                            cameras_[m].rt_.data(),
                            chessboards_[i].rt_.data(),
                            cameras_[m].intrinsic_.data()
                        );
                    }
                }
            }
        }
    }
    // Configure the solver.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout =false;
    // options.max_num_iterations = 100;

    // Solve!
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    // 更新相机和棋盘格的参数
    for(int i = 0; i < camera_num; i++)
    {
        if(cameras_[i].is_initial() == false)
            continue;
        cameras_[i].update_param();
    }
    for(int i = 0; i < chessboard_num; i++)
    {
        if(chessboards_[i].is_initial() == false)
            continue;
        chessboards_[i].update_param();
    }
    // 计算重投影误差
    std::cout << blue << "多相机标定结束" << reset << std::endl;
    double error_sum = 0;
    int cntt = 0;
    for(int m = 0; m < camera_num; m++)
    {
        std::vector<std::vector<cv::Point2d>> pixels = cameras_[m].pixels();
        double error = 0;
        double cnt = 0;
        for(int i = 0; i < chessboard_num; i++)
        {
            if(chessboards_[i].is_initial())
            {
                for(int j = 0; j < pixels[i].size(); j++)
                {
                    cv::Mat R = chessboards_[i].R();
                    cv::Mat t = chessboards_[i].t();
                    cv::Mat p = (cv::Mat_<double>(3, 1) << worlds_[j].x, worlds_[j].y, worlds_[j].z);
                    p = R * p + t;
                    R = cameras_[m].R();
                    t = cameras_[m].t();
                    p = R * p + t;
                    double fx = cameras_[m].fx();
                    double fy = cameras_[m].fy();
                    double cx = cameras_[m].cx();
                    double cy = cameras_[m].cy();
                    double alpha = cameras_[m].alpha();
                    double lamda = cameras_[m].lamda();
                    double xi = cameras_[m].xi();
                    double b = cameras_[m].b();
                    double c = cameras_[m].c();
                    double X = p.at<double>(0,0);
                    double Y = p.at<double>(1,0);
                    double Z = p.at<double>(2,0);
                    double d1 = std::sqrt(X*X+Y*Y+Z*Z);
                    double d2 = std::sqrt(X*X+Y*Y+std::pow(Z+xi*d1,2));
                    double d3 = std::sqrt(X*X+Y*Y+std::pow(Z+xi*d1+lamda*d2,2));
                    double ksai = Z+xi*d1+lamda*d2+alpha/(1-alpha)*d3;
                    double pixel_x = fx * X/ksai + b * Y/ksai + cx;
                    double pixel_y = c * X/ksai + fy * Y/ksai + cy;
                    error += std::sqrt((pixels[i][j].x-pixel_x)*(pixels[i][j].x-pixel_x) + (pixels[i][j].y-pixel_y)*(pixels[i][j].y-pixel_y));
                    cnt++;
                    cntt++;
                }
            }
        }
        error_sum += error;
        error /= cnt;
        std::cout << "camera_" << m << " 重投影误差：" << error << std::endl;
    }
    std::cout << yellow << "average reproject error: " << error_sum/cntt << reset << std::endl;
}

//////////////////////////////////////////////
//               可视化部分
//////////////////////////////////////////////

double position_x = 0, position_y = 0, last_position_x = 0, last_position_y = 0, position_z = 500;
double angle_x = 0, angle_y = 0, last_angle_x = 0, last_angle_y = 0;

bool mouse_l_down = false;
bool mouse_r_down = false;
cv::Point last_position(-1, -1);
void on_MouseHandle(int event, int x, int y, int flag, void *param)
{
    float value;
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        mouse_l_down = true;
        last_position.x = x;
        last_position.y = y;
        break;
    case cv::EVENT_LBUTTONUP:
        mouse_l_down = false;
        last_angle_x = last_angle_x + (x - last_position.x) * 0.01;
        last_angle_y = last_angle_y + (y - last_position.y) * 0.01;
        break;
    case cv::EVENT_RBUTTONDOWN:
        mouse_r_down = true;
        last_position.x = x;
        last_position.y = y;
        break;
    case cv::EVENT_RBUTTONUP:
        mouse_r_down = false;
        last_position_x = last_position_x + x - last_position.x;
        last_position_y = last_position_y - (y - last_position.y);
        break;
    case cv::EVENT_MOUSEMOVE:
        if (mouse_l_down)
        {
            angle_x = last_angle_x + (x - last_position.x) * 0.01;
            angle_y = last_angle_y + (y - last_position.y) * 0.01;
        }
        else if (mouse_r_down)
        {
            position_x = last_position_x + x - last_position.x;
            position_y = last_position_y - (y - last_position.y);
        }
        
        break;
    case cv::EVENT_MOUSEWHEEL:
        value = cv::getMouseWheelDelta(flag);
        position_z += value*20;
        break;
    default:;
    }
}

void MultiCalib::show_result()
{
    // 根据姿态计算点的世界坐标
    std::vector<std::vector<cv::Point3d>> camera_points;
    std::vector<std::vector<cv::Point3d>> chessboard_points;
    std::vector<cv::Point3d> axis_points;
    axis_points.push_back(cv::Point3d(0,0,0));
    axis_points.push_back(cv::Point3d(50,0,0));
    axis_points.push_back(cv::Point3d(0,50,0));
    axis_points.push_back(cv::Point3d(0,0,50));
    for(int i = 0; i < cameras_.size(); i++)
    {
        if(cameras_[i].is_initial())
        {
            std::vector<cv::Point3d> points;
            cv::Mat R, t, p;
            R = cameras_[i].R();
            t = cameras_[i].t();
            std::cout << "=================" << std::endl;
            std::cout << yellow << "[camera_" << i << "]: " << std::endl << t << std::endl;
            std::cout << R << std::endl;
            std::cout << "fx: " << cameras_[i].fx() << ", fy: " << cameras_[i].fy() << ", cx: " << cameras_[i].cx() << ", cy: " << cameras_[i].cy() << 
                         ", xi: " << cameras_[i].xi() << ", lamda: " << cameras_[i].lamda() << ", alpha: " << cameras_[i].alpha() << 
                         ", b: " << cameras_[i].b() << ", c: " << cameras_[i].c() << std::endl;
            std::cout << reset;
            R = R.inv();
            t = -R * t;
            std::cout << "*****************" << std::endl;
            std::cout <<R << std::endl << t << std::endl;
            p = (cv::Mat_<double>(3,1) << -10, -10, 0);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << 10, -10, 0);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << 10, 10, 0);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << -10, 10, 0);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << -20, -20, 30);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << 20, -20, 30);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << 20, 20, 30);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << -20, 20, 30);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            camera_points.push_back(points);
        }
    }
    for(int i = 0; i < chessboards_.size(); i++)
    {
        if(chessboards_[i].is_initial())
        {
            std::vector<cv::Point3d> points;
            cv::Mat R, t, p;
            R = chessboards_[i].R();
            t = chessboards_[i].t();
            p = (cv::Mat_<double>(3,1) << -80, -50, 0);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << 80, -50, 0);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << 80, 50, 0);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            p = (cv::Mat_<double>(3,1) << -80, 50, 0);
            p = R * p + t;
            points.push_back(cv::Point3d(p.at<double>(0,0), p.at<double>(1,0), p.at<double>(2,0)));
            chessboard_points.push_back(points);
        }
    }
    cv::Mat show_img = cv::Mat(cv::Size(800, 800), CV_8UC3, cv::Scalar(255,255,255));
	cv::imshow("show_window", show_img);
	cv::setMouseCallback("show_window", on_MouseHandle);
    while(true)
    {
        cv::Mat show_img = cv::Mat(cv::Size(800, 800), CV_8UC3, cv::Scalar(255,255,255));
        cv::Mat r1 = (cv::Mat_<double>(3, 3) << cos(angle_x), 0, sin(angle_x),
            0, 1, 0,
            -sin(angle_x), 0, cos(angle_x));
        cv::Mat r2 = (cv::Mat_<double>(3, 3) << 1, 0, 0,
            0, cos(angle_y), sin(angle_y),
            0, -sin(angle_y), cos(angle_y));
        cv::Mat K = (cv::Mat_<double>(3,3) << 500, 0, 400, 
                                                0, 500, 400, 
                                                0, 0, 1);
        r1 = r1 * r2;
        cv::Mat Rt = (cv::Mat_<double>(3,4) << r1.at<double>(0,0), r1.at<double>(0,1), r1.at<double>(0,2), position_x,
                                               r1.at<double>(1,0), r1.at<double>(1,1), r1.at<double>(1,2), -position_y,
                                               r1.at<double>(2,0), r1.at<double>(2,1), r1.at<double>(2,2), position_z);
        K = K * Rt;
        for (int k = 0; k < chessboard_points.size(); k++)
        {
            std::vector<cv::Point2d> points_i;
            bool valid = true;
            for(int l = 0; l < chessboard_points[k].size(); l++)
            {
                cv::Mat p = (cv::Mat_<double>(4, 1) << chessboard_points[k][l].x, chessboard_points[k][l].y, chessboard_points[k][l].z, 1);
                p = K * p;
                p /= p.at<double>(2,0);
                points_i.push_back(cv::Point2d(p.at<double>(0,0), p.at<double>(1,0)));
                if(p.at<double>(0,0) < 0 || p.at<double>(0,0) > 800 || p.at<double>(1,0) < 0 || p.at<double>(1,0) > 800)
                    valid = false;
            }
            if(valid)
            {
                cv::line(show_img, points_i[0], points_i[1], cv::Scalar(255,0,0), 2);
                cv::line(show_img, points_i[1], points_i[2], cv::Scalar(255,0,0), 2);
                cv::line(show_img, points_i[2], points_i[3], cv::Scalar(255,0,0), 2);
                cv::line(show_img, points_i[3], points_i[0], cv::Scalar(255,0,0), 2);
            }
        }
        std::vector<cv::Point2d> points_copy;
        for(int i = 0; i < axis_points.size(); i++)
        {
            cv::Mat p = (cv::Mat_<double>(4, 1) << axis_points[i].x, axis_points[i].y, axis_points[i].z, 1);
            p = K * p;
            p /= p.at<double>(2,0);
            points_copy.push_back(cv::Point2d(p.at<double>(0,0), p.at<double>(1,0)));
        }
        cv::arrowedLine(show_img, points_copy[0], points_copy[1], cv::Scalar(255,0,0), 2);
        cv::arrowedLine(show_img, points_copy[0], points_copy[2], cv::Scalar(0,255,0), 2);
        cv::arrowedLine(show_img, points_copy[0], points_copy[3], cv::Scalar(0,0,255), 2);
        for (int k = 0; k < camera_points.size(); k++)
        {
            std::vector<cv::Point2d> points_i;
            for(int l = 0; l < camera_points[k].size(); l++)
            {
                cv::Mat p = (cv::Mat_<double>(4, 1) << camera_points[k][l].x, camera_points[k][l].y, camera_points[k][l].z, 1);
                p = K * p;
                p /= p.at<double>(2,0);
                points_i.push_back(cv::Point2d(p.at<double>(0,0), p.at<double>(1,0)));
            }
            cv::line(show_img, points_i[0], points_i[1], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[1], points_i[2], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[2], points_i[3], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[3], points_i[0], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[4], points_i[5], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[5], points_i[6], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[6], points_i[7], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[7], points_i[4], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[0], points_i[4], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[1], points_i[5], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[2], points_i[6], cv::Scalar(0,255,0), 2);
            cv::line(show_img, points_i[3], points_i[7], cv::Scalar(0,255,0), 2);
        }
        
        
        cv::imshow("show_window", show_img);
        int key = cv::waitKey(10);
        if(key == 'q' || key == 27 || cv::getWindowProperty("show_window", cv::WND_PROP_AUTOSIZE) < 0)
            break;
    }
}
