#ifndef DS_H
#define DS_H

#include <opencv2/opencv.hpp>

class DoubleSphereCamera
{
    public:
    DoubleSphereCamera();
    DoubleSphereCamera(double cx, double cy, double fx, double fy, double xi, double alpha);
    void calibrate(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds, const cv::Size img_size);
    double cx() {return cx_;}
    double cy() {return cy_;}
    double fx() {return fx_;}
    double fy() {return fy_;}
    double xi() {return xi_;}
    double alpha() {return alpha_;}
    private:
    void initialize_param(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds);
    bool has_init_guess_;
    double cx_;
    double cy_;
    double fx_;
    double fy_;
    double xi_;
    double alpha_;
    std::vector<cv::Point3d> t;
    std::vector<cv::Point3d> r1;
    std::vector<cv::Point3d> r2;
    std::vector<cv::Point3d> r3;
};

#endif