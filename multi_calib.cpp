#include "multi_calib.hpp"

MultiCalib::MultiCalib(std::vector<DoubleSphereCamera> cameras)
{
    int camera_num = cameras.size();
    int chessboard_num = cameras[0].has_chessboard().size();
    cameras_.resize(camera_num);
    chessboards_.resize(chessboard_num);
    camera_rt_.resize(camera_num);
    intrinsic_.resize(camera_num);
    chessboard_rt_.resize(chessboard_num);
    // 先粗略的估计所有相机和棋盘格的位姿
    for(int i = 0; i < camera_num; i++)
    {
        cv::Mat R, t;
        if(i == 0)
        {
            // 第一个相机假设为世界坐标系原点
            R = (cv::Mat_<double>(3,3) << 1,0,0, 0,1,0, 0,0,1);
            t = (cv::Mat_<double>(3,1) << 0,0,0);
            for(int j = 0; j < chessboard_num; j++)
            {
                // 第一个相机观察到的棋盘格的位姿可以直接求出来
                if(cameras[i].has_chessboard(j))
                {
                    // 如果第一个相机观察到了这个棋盘格
                    cv::Mat Rt = cameras[i].Rt(j);
                    cv::Vec3f r1(Rt.at<double>(0,0),Rt.at<double>(1,0),Rt.at<double>(2,0));
                    cv::Vec3f r2(Rt.at<double>(0,1),Rt.at<double>(1,1),Rt.at<double>(2,1));
                    cv::Vec3f r3 = r1.cross(r2);
                    cv::Mat R = (cv::Mat_<double>(3,3) << r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]);
                    cv::Mat t = (cv::Mat_<double>(3,1) << Rt.at<double>(0,2), Rt.at<double>(1,2), Rt.at<double>(2,2));
                    chessboards_[j] = MultiCalib_chessboard(R, t);
                }
            }
            cameras_[i] = MultiCalib_camera(cameras[i].cx(), cameras[i].cy(), cameras[i].fx(), cameras[i].fy(), cameras[i].xi(), cameras[i].alpha(),
                                            R, t, cameras[i].has_chessboard(), cameras[i].pixels());
        }
        else
        {
            // 如果不是第一个相机，就判断自己观察到的棋盘格中
            // 那个棋盘格被其他已知姿态的相机观察到
            for(int j = 0; j < chessboard_num; j++)
            {
                if(cameras_[i].is_initial() == false)
                {
                    // 非第一个相机姿态初始化
                    // 如果自己观察到的棋盘格的姿态已知
                    if(cameras[i].has_chessboard(j) && chessboards_[j].is_initial())
                    {
                        cv::Mat Rt = cameras[i].Rt(j);
                        cv::Vec3f r1(Rt.at<double>(0,0),Rt.at<double>(1,0),Rt.at<double>(2,0));
                        cv::Vec3f r2(Rt.at<double>(0,1),Rt.at<double>(1,1),Rt.at<double>(2,1));
                        cv::Vec3f r3 = r1.cross(r2);
                        cv::Mat camera_R = (cv::Mat_<double>(3,3) << r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]);
                        cv::Mat camera_t = (cv::Mat_<double>(3,1) << Rt.at<double>(0,2), Rt.at<double>(1,2), Rt.at<double>(2,2));
                        cv::Mat chess_R = chessboards_[j].R();
                        cv::Mat chess_t = chessboards_[j].t();
                        R = camera_R * chess_R.inv();
                        t = camera_t - R * chess_t;
                        cameras_[i] = MultiCalib_camera(cameras[i].cx(), cameras[i].cy(), cameras[i].fx(), cameras[i].fy(), cameras[i].xi(), cameras[i].alpha(),
                                            R, t, cameras[i].has_chessboard(), cameras[i].pixels());
                        for(int k = 0; k < chessboard_num; k++)
                        {
                            // 自己姿态已知时，可以计算棋盘格的姿态
                            if(cameras[i].has_chessboard(k) && chessboards_[k].is_initial() == false)
                            {
                                cv::Mat Rt = cameras[i].Rt(k);
                                cv::Vec3f r1(Rt.at<double>(0,0),Rt.at<double>(1,0),Rt.at<double>(2,0));
                                cv::Vec3f r2(Rt.at<double>(0,1),Rt.at<double>(1,1),Rt.at<double>(2,1));
                                cv::Vec3f r3 = r1.cross(r2);
                                cv::Mat camera_R = (cv::Mat_<double>(3,3) << r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]);
                                cv::Mat camera_t = (cv::Mat_<double>(3,1) << Rt.at<double>(0,2), Rt.at<double>(1,2), Rt.at<double>(2,2));
                                R = cameras_[i].R();
                                t = cameras_[i].t();
                                cv::Mat chess_R = R.inv() * camera_R;
                                cv::Mat chess_t = R.inv() * (camera_t - t);
                                chessboards_[k] = MultiCalib_chessboard(chess_R, chess_t);
                            }
                        }
                    }
                }
                else
                {
                    continue;
                }
            }
        }
        if(cameras_[i].is_initial() == false)
            cameras_[i] = MultiCalib_camera(cameras[i].cx(), cameras[i].cy(), cameras[i].fx(), cameras[i].fy(), cameras[i].xi(), cameras[i].alpha(),
                                            R, t, cameras[i].has_chessboard(), cameras[i].pixels());
    }
    // 细化相机的位姿
    for(int i = 1; i < camera_num; i++)
    {
        cv::Mat r_avg = cv::Mat::zeros(3, 1, CV_64F), t_avg = cv::Mat::zeros(3, 1, CV_64F);
        int num = 0;
        for(int j = 0; j < chessboard_num; j++)
        {
            if(cameras[i].has_chessboard(j) && cameras_[i].is_initial() && chessboards_[j].is_initial())
            {
                cv::Mat Rt = cameras[i].Rt(j);
                cv::Vec3f r1(Rt.at<double>(0,0),Rt.at<double>(1,0),Rt.at<double>(2,0));
                cv::Vec3f r2(Rt.at<double>(0,1),Rt.at<double>(1,1),Rt.at<double>(2,1));
                cv::Vec3f r3 = r1.cross(r2);
                cv::Mat camera_R = (cv::Mat_<double>(3,3) << r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]);
                cv::Mat camera_t = (cv::Mat_<double>(3,1) << Rt.at<double>(0,2), Rt.at<double>(1,2), Rt.at<double>(2,2));
                cv::Mat chess_R = chessboards_[j].R();
                cv::Mat chess_t = chessboards_[j].t();
                cv::Mat R = camera_R * chess_R.inv();
                cv::Mat t = camera_t - R * chess_t;
                cv::Mat r;
                cv::Rodrigues(R, r);
                r_avg += r;
                t_avg += t;
                num++;
            }
        }
        r_avg /= num;
        t_avg /= num;
        cv::Mat R;
        cv::Rodrigues(r_avg, R);
        cameras_[i].update_Rt(R, t_avg);
    }
    // 细化棋盘格的位姿
    for(int j = 0; j < chessboard_num; j++)
    {
        cv::Mat r_avg = cv::Mat::zeros(3, 1, CV_64F), t_avg = cv::Mat::zeros(3, 1, CV_64F);
        int num = 0;
        for(int i = 0; i < camera_num; i++)
        {
            if(cameras[i].has_chessboard(j) && cameras_[i].is_initial() && chessboards_[j].is_initial())
            {
                cv::Mat Rt = cameras[i].Rt(j);
                cv::Vec3f r1(Rt.at<double>(0,0),Rt.at<double>(1,0),Rt.at<double>(2,0));
                cv::Vec3f r2(Rt.at<double>(0,1),Rt.at<double>(1,1),Rt.at<double>(2,1));
                cv::Vec3f r3 = r1.cross(r2);
                cv::Mat camera_R = (cv::Mat_<double>(3,3) << r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]);
                cv::Mat camera_t = (cv::Mat_<double>(3,1) << Rt.at<double>(0,2), Rt.at<double>(1,2), Rt.at<double>(2,2));
                cv::Mat R = cameras_[i].R();
                cv::Mat t = cameras_[i].t();
                cv::Mat chess_R = R.inv() * camera_R;
                cv::Mat chess_t = R.inv() * (camera_t - t);
                cv::Mat r;
                cv::Rodrigues(chess_R, r);
                r_avg += r;
                t_avg += chess_t;
                num++;
            }
        }
        r_avg /= num;
        t_avg /= num;
        cv::Mat R;
        cv::Rodrigues(r_avg, R);
        chessboards_[j].update_Rt(R, t_avg);
    }
    // 准备被优化的参数
    for(int i = 0; i < cameras_.size(); i++)
    {
        if(cameras_[i].is_initial() == false)
            continue;
        cv::Mat r, t;
        cv::Rodrigues(cameras_[i].R(), r);
        t = cameras_[i].t();
        std::vector<double> rt{r.at<double>(0,0), r.at<double>(1,0), r.at<double>(2,0), t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0)};
        camera_rt_[i] = rt;
        intrinsic_[i] = std::vector<double>{cameras_[i].fx(), cameras_[i].fy(), cameras_[i].cx(), cameras_[i].cy(), cameras_[i].xi(), cameras_[i].alpha()};
    }
    for(int i = 0; i < chessboards_.size(); i++)
    {
        if(chessboards_[i].is_initial() == false)
            continue;
        cv::Mat r, t;
        cv::Rodrigues(chessboards_[i].R(), r);
        t = chessboards_[i].t();
        std::vector<double> rt{r.at<double>(0,0), r.at<double>(1,0), r.at<double>(2,0), t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0)};
        chessboard_rt_[i] = rt;
    }
}

void MultiCalib::calibrate(const std::vector<cv::Point3d>& worlds)
{
    int camera_num = cameras_.size();
    int chessboard_num = chessboards_.size();

    for(int m = 0; m < camera_num; m++)
    {
        std::vector<std::vector<cv::Point2d>> pixels = cameras_[m].pixels();
        double error = 0;
        for(int i = 0; i < chessboard_num; i++)
        {
            if(chessboards_[i].is_initial())
            {
                for(int j = 0; j < pixels[i].size(); j++)
                {
                    cv::Mat R = chessboards_[i].R();
                    cv::Mat t = chessboards_[i].t();
                    cv::Mat p = (cv::Mat_<double>(3, 1) << worlds[j].x, worlds[j].y, worlds[j].z);
                    p = R * p + t;
                    R = cameras_[m].R();
                    t = cameras_[m].t();
                    p = R * p + t;
                    double fx = cameras_[m].fx();
                    double fy = cameras_[m].fy();
                    double cx = cameras_[m].cx();
                    double cy = cameras_[m].cy();
                    double alpha = cameras_[m].alpha();
                    double xi = cameras_[m].xi();
                    double d1 = std::sqrt(p.at<double>(0,0)*p.at<double>(0,0)+p.at<double>(1,0)*p.at<double>(1,0)+p.at<double>(2,0)*p.at<double>(2,0));
                    double d2 = std::sqrt(p.at<double>(0,0)*p.at<double>(0,0)+p.at<double>(1,0)*p.at<double>(1,0)+std::pow(p.at<double>(2,0)+xi*d1,2));
                    double pixel_x = fx * p.at<double>(0,0)/(alpha*d2+(1-alpha)*(xi*d1+p.at<double>(2,0))) + cx;
                    double pixel_y = fy * p.at<double>(1,0)/(alpha*d2+(1-alpha)*(xi*d1+p.at<double>(2,0))) + cy;
                    error += std::sqrt((pixels[i][j].x-pixel_x)*(pixels[i][j].x-pixel_x) + (pixels[i][j].y-pixel_y)*(pixels[i][j].y-pixel_y));
                }
            }
        }
        std::cout << "my own calculate: " << error << std::endl;
    }
    
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
                        NoRtReprojectionError *cost_function = 
                            new NoRtReprojectionError(pixels[i][j], worlds[j], camera_rt_[m]);
                        problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<
                                NoRtReprojectionError,
                                2,  // num_residuals
                                6,6>(cost_function),
                            NULL,
                            chessboard_rt_[i].data(), 
                            intrinsic_[m].data());
                    }
                    else
                    {
                        ReprojectionError *cost_function =
                            new ReprojectionError(pixels[i][j], worlds[j]);

                        problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<
                                ReprojectionError,
                                2,  // num_residuals
                                6,6,6>(cost_function),
                            NULL,
                            camera_rt_[m].data(),
                            chessboard_rt_[i].data(),
                            intrinsic_[m].data());
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
}

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
        std::cout << value << std::endl;
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
            cv::Mat r = (cv::Mat_<double>(3, 1) << camera_rt_[i][0], camera_rt_[i][1], camera_rt_[i][2]);
            cv::Rodrigues(r, R);
            t = (cv::Mat_<double>(3,1) << camera_rt_[i][3], camera_rt_[i][4], camera_rt_[i][5]);
            std::cout << "=================" << std::endl;
            std::cout << "[camera_" << i << "]: " << std::endl << t << std::endl;
            std::cout << R << std::endl;
            std::cout << "fx: " << intrinsic_[i][0] << ", fy: " << intrinsic_[i][1] << ", cx: " << intrinsic_[i][2] << ", cy: " << intrinsic_[i][3] << ", alpha: " << intrinsic_[i][5] << ", xi: " << intrinsic_[i][4] << std::endl;
            R = R.inv();
            t = -R * t;
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
            cv::Mat r = (cv::Mat_<double>(3, 1) << chessboard_rt_[i][0], chessboard_rt_[i][1], chessboard_rt_[i][2]);
            cv::Rodrigues(r, R);
            t = (cv::Mat_<double>(3,1) << chessboard_rt_[i][3], chessboard_rt_[i][4], chessboard_rt_[i][5]);
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
    cv::Mat show_img = cv::Mat::zeros(cv::Size(800, 800), CV_8UC3);
	cv::imshow("show_window", show_img);
	cv::setMouseCallback("show_window", on_MouseHandle);
    while(true)
    {
        cv::Mat show_img = cv::Mat::zeros(cv::Size(800, 800), CV_8UC3);
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
        
        cv::imshow("show_window", show_img);
        int key = cv::waitKey(10);
        if(key == 'q')
            break;
    }
}
