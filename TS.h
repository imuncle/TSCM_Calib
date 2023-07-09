#ifndef TS_H
#define TS_H

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class TripleSphereCamera
{
    public:
    TripleSphereCamera();
    TripleSphereCamera(double fx, double fy, double cx, double cy, double xi, double lamda, double alpha);
    bool calibrate(const std::vector<std::vector<cv::Point2d>> pixels, 
                   std::vector<bool> has_chessboard, 
                   const std::vector<cv::Point3d>& worlds, 
                   const cv::Size img_size, 
                   const cv::Size chessboard_num);
    void Reproject(const std::vector<cv::Point3d>& worlds, cv::Mat Rt, std::vector<cv::Point2d>& pixels);
    void undistort(double fx, double fy, double cx, double cy, cv::Size img_size, cv::Mat& mapx, cv::Mat& mapy);
    double cx() {return cx_;}
    double cy() {return cy_;}
    double fx() {return fx_;}
    double fy() {return fy_;}
    double xi() {return xi_;}
    double lamda() {return lamda_;}
    double alpha() {return alpha_;}
    double b() {return b_;}
    double c() {return c_;}
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
        double x = pixel.x - cx_;
        double y = pixel.y - cy_;
        double mx = (fy_*x - b_*y)/(fx_*fy_-b_*c_);
        double my = (-c_*x + fx_*y)/(fx_*fy_-b_*c_);
        double ksai = alpha_ / (1 - alpha_);
        double r_square = mx*mx + my*my;
        double gamma = (ksai+std::sqrt(1+(1-ksai*ksai)*r_square))/(r_square+1);
        double yita = lamda_*(gamma-ksai)+std::sqrt(((gamma-ksai)*(gamma-ksai)-1)*lamda_*lamda_+1);
        double mz = yita*(gamma-ksai);
        double mu = xi_*(mz-lamda_)+std::sqrt(xi_*xi_*((mz-lamda_)*(mz-lamda_)-1)+1);
        x = mu*yita*gamma*mx;
        y = mu*yita*gamma*my;
        double z = mu*(mz-lamda_) - xi_;
        cv::Mat p = (cv::Mat_<double>(3,1) << x, y, z);
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
                          double cx, double cy, double fx, double fy, double xi, double lamda, double alpha, double b, double c);
    bool refinement(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds);
    bool has_init_guess_;
    double cx_;
    double cy_;
    double fx_;
    double fy_;
    double xi_;
    double lamda_;
    double alpha_;
    double b_;
    double c_;
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
            // intrinsic: fx fy cx cy xi lambda alpha
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
            T d3 = ceres::sqrt(P[0]*P[0]+P[1]*P[1]+(P[2]+intrinsic_[4]*d1+intrinsic_[5]*d2)*(P[2]+intrinsic_[4]*d1+intrinsic_[5]*d2));
            T ksai = P[2]+intrinsic_[4]*d1+intrinsic_[5]*d2+intrinsic_[6]/(one-intrinsic_[6])*d3;    // TS Model
            // T ksai = P[2]+intrinsic_[4]*d1+intrinsic_[6]/(one-intrinsic_[6])*d2;    // DS Model
            // T pixel_x = intrinsic_[0] * P[0]/ksai + intrinsic_[7] * P[1]/ksai + intrinsic_[2];
            // T pixel_y = intrinsic_[8] * P[0]/ksai + intrinsic_[1] * P[1]/ksai + intrinsic_[3];
            T pixel_x = intrinsic_[0] * P[0]/ksai + intrinsic_[2];
            T pixel_y = intrinsic_[1] * P[1]/ksai + intrinsic_[3];

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
