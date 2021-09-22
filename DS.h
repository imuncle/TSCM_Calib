#ifndef DS_H
#define DS_H

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class DoubleSphereCamera
{
    public:
    DoubleSphereCamera();
    DoubleSphereCamera(double cx, double cy, double fx, double fy, double xi, double alpha);
    void calibrate(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds, const cv::Size img_size);
    void Reproject(const std::vector<cv::Point3d>& worlds, cv::Mat Rt, std::vector<cv::Point2d>& pixels);
    void undistort(double fx, double fy, double cx, double cy, cv::Size img_size, cv::Mat& mapx, cv::Mat& mapy);
    double cx() {return cx_;}
    double cy() {return cy_;}
    double fx() {return fx_;}
    double fy() {return fy_;}
    double xi() {return xi_;}
    double alpha() {return alpha_;}
    std::vector<cv::Mat> Rt() {return Rt_;}
    private:
    void initialize_param(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds);
    double ReprojectError(const std::vector<cv::Point2d>& pixels, const std::vector<cv::Point3d>& worlds, cv::Mat Rt,
                          double cx, double cy, double fx, double fy, double xi, double alpha);
    void refinement(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds);
    bool has_init_guess_;
    double cx_;
    double cy_;
    double fx_;
    double fy_;
    double xi_;
    double alpha_;
    std::vector<cv::Mat> Rt_;
    std::vector<double> gammas_;
    std::vector<std::vector<double>> rt_; 
    std::vector<double> intrinsic_;

    struct ReprojectionError
    {
        ReprojectionError(const cv::Point2d img_pts_, const cv::Point3d board_pts_)
            :_img_pts(img_pts_), _board_pts(board_pts_)
        {
        }

        template<typename T>
        bool operator()(const T* const intrinsic_,
            const T* const rt_,//6 : angle axis and translation
            T* residuls)const
        {
            // intrinsic: fx fy cx cy xi alpha
            T p[3];
            p[0] = T(_board_pts.x);
            p[1] = T(_board_pts.y);
            p[2] = T(0.0);
            T one = T(1.0);
            T P[3];
            ceres::AngleAxisRotatePoint(rt_, p, P);
            P[0] += rt_[3];
            P[1] += rt_[4];
            P[2] += rt_[5];

            T d1 = ceres::sqrt(P[0]*P[0]+P[1]*P[1]+P[2]*P[2]);
            T d2 = ceres::sqrt(P[0]*P[0]+P[1]*P[1]+(P[2]+intrinsic_[4]*d1)*(P[2]+intrinsic_[4]*d1));
            T pixel_x = intrinsic_[0] * P[0]/(intrinsic_[5]*d2+(one-intrinsic_[5])*(intrinsic_[4]*d1+P[2])) + intrinsic_[2];
            T pixel_y = intrinsic_[1] * P[1]/(intrinsic_[5]*d2+(one-intrinsic_[5])*(intrinsic_[4]*d1+P[2])) + intrinsic_[3];

            // residuls
            residuls[0] = T(_img_pts.x) - pixel_x;
            residuls[1] = T(_img_pts.y) - pixel_y;
            return true;
        }
        const cv::Point2d _img_pts;
        const cv::Point3d _board_pts;
	};
};

#endif