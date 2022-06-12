#ifndef DS_H
#define DS_H

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class DoubleSphereCamera
{
    public:
    DoubleSphereCamera();
    DoubleSphereCamera(double fx, double fy, double cx, double cy, double alpha, double xi);
    bool calibrate(const std::vector<std::vector<cv::Point2d>> pixels, 
                   std::vector<bool> has_chessboard, 
                   const std::vector<cv::Point3d>& worlds, 
                   const cv::Size img_size, 
                   const cv::Size chessboard_num);
    void Reproject(const std::vector<cv::Point3d>& worlds, cv::Mat Rt, std::vector<cv::Point2d>& pixels);
    void undistort(double fx, double fy, double cx, double cy, cv::Size img_size, cv::Mat& mapx, cv::Mat& mapy);
    double cx() {return cx_;}
    double cy() {return cy_;}
    double fx() {return fx_*(1-alpha_);}
    double fy() {return fy_*(1-alpha_);}
    double xi() {return xi_;}
    double alpha() {return alpha_;}
    std::vector<cv::Mat> Rt() {return Rt_;}
    cv::Mat Rt(int id) {return Rt_[id];}
    void setRt(std::vector<cv::Mat> Rts) {Rt_ = Rts;}
    void setPixels(std::vector<std::vector<cv::Point2d>> pixels) {pixels_ = pixels;}
    std::vector<std::vector<cv::Point2d>> pixels() {return pixels_;}
    std::vector<bool> has_chessboard() {return has_chessboard_;}
    bool has_chessboard(int id) {return has_chessboard_[id];}
    void setHasChessboard(std::vector<bool> has_chessboard) {has_chessboard_ = has_chessboard;}
    cv::Mat undistort_chessboard(cv::Mat src, int index, cv::Size chessboard, double chessboard_size);
    cv::Point2d project(cv::Mat P);
    cv::Point3d get_unit_sphere_coordinate(cv::Point2d pixel, cv::Mat transform=cv::Mat::eye(3,3,CV_64F))
    {
        float mx = (pixel.x - cx_) / fx_ / (1-alpha_);
        float my = (pixel.y - cy_) / fy_ / (1-alpha_);
        float r_square = mx*mx + my*my;
        float mz = (1-alpha_*alpha_*r_square)/(alpha_*std::sqrt(1-(2*alpha_-1)*r_square)+1-alpha_);
        float ksai = alpha_ / (1 - alpha_);
        float x = (mz*xi_+std::sqrt(mz*mz+(1-xi_*xi_)*r_square))/(mz*mz+r_square);
        cv::Mat p = (cv::Mat_<double>(3,1) << x*mx, x*my, x*mz-xi_);
        p = transform * p;
        return cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2));
    }
    double ReprojectError(const std::vector<cv::Point2d>& pixels, const std::vector<cv::Point3d>& worlds, cv::Mat R, cv::Mat t)
    {
        double error = 0;
        for(int i = 0; i < worlds.size(); i++)
        {
            cv::Mat P = (cv::Mat_<double>(3,1) << worlds[i].x, worlds[i].y, worlds[i].z);
            P = R * P + t;
            cv::Point2d project_p = project(P);
            error += std::sqrt((pixels[i].x-project_p.x)*(pixels[i].x-project_p.x) + (pixels[i].y-project_p.y)*(pixels[i].y-project_p.y));
        }
        return error;
    }
    private:
    void estimate_focal(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds, cv::Size img_size, const cv::Size chessboard_num);
    void estimate_extrinsic(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds, const cv::Size chessboard_num);
    void initialize_param(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds);
    double ReprojectError(const std::vector<cv::Point2d>& pixels, const std::vector<cv::Point3d>& worlds, cv::Mat Rt,
                          double cx, double cy, double fx, double fy, double xi, double alpha);
    bool refinement(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds);
    bool has_init_guess_;
    double cx_;
    double cy_;
    double fx_;
    double fy_;
    double xi_;
    double alpha_;
    std::vector<cv::Mat> Rt_;
    std::vector<std::vector<double>> rt_; 
    std::vector<std::vector<cv::Point2d>> pixels_;
    std::vector<double> intrinsic_;
    std::vector<bool> has_chessboard_;

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
            T pixel_x = intrinsic_[0] * (one-intrinsic_[5])*P[0]/(intrinsic_[5]*d2+(one-intrinsic_[5])*(intrinsic_[4]*d1+P[2])) + intrinsic_[2];
            T pixel_y = intrinsic_[1] * (one-intrinsic_[5])*P[1]/(intrinsic_[5]*d2+(one-intrinsic_[5])*(intrinsic_[4]*d1+P[2])) + intrinsic_[3];

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
