#include "DS.h"

DoubleSphereCamera::DoubleSphereCamera()
{
    has_init_guess_ = false;
}

DoubleSphereCamera::DoubleSphereCamera(double cx, double cy, double fx, double fy, double xi, double alpha)
{
    has_init_guess_= true;
    cx_ = cx;
    cy_ = cy;
    fx_ = fx;
    fy_ = fy;
    xi_ = xi;
    alpha_ = alpha;
}

void DoubleSphereCamera::calibrate(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds, const cv::Size img_size)
{
    int img_num = pixels.size();
    r1.resize(img_num);
    r2.resize(img_num);
    r3.resize(img_num);
    t.resize(img_num);
    if(has_init_guess_ == false)
    {
        cx_ = img_size.width / 2 - 0.5;
        cy_ = img_size.height / 2 - 0.5;
        xi_ = 0;
        alpha_ = 0.5;
        initialize_param(pixels, worlds);
    }
}

void DoubleSphereCamera::initialize_param(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds)
{
    // 将像素坐标原点移到(cx, cy)
    std::vector<std::vector<cv::Point2d>> pixels_center;
    for(int i = 0; i < pixels.size(); i++)
    {
        std::vector<cv::Point2d> pixels_tmp;
        for(int j = 0; j < pixels[i].size(); j++)
        {
            cv::Point2d p(pixels[i][j].x - cx_, pixels[i][j].y - cy_);
            pixels_tmp.push_back(p);
        }
        pixels_center.push_back(pixels_tmp);
    }
    for(int i = 0; i < pixels_center.size(); i++)
    {
        int point_num = pixels_center[i].size();
        cv::Mat A(cv::Size(6, point_num), CV_64F);
        cv::Mat x(cv::Size(1, 6), CV_64F);
        for(int j = 0; j < point_num; j++)
        {
            A.at<double>(j, 0) = -pixels_center[i][j].y * worlds[j].x;
            A.at<double>(j, 1) = -pixels_center[i][j].y * worlds[j].y;
            A.at<double>(j, 2) = pixels_center[i][j].x * worlds[j].x;
            A.at<double>(j, 3) = pixels_center[i][j].x * worlds[j].y;
            A.at<double>(j, 4) = -pixels_center[i][j].y;
            A.at<double>(j, 5) = pixels_center[i][j].y;
        }
        cv::SVD::solveZ(A, x);
    }
}