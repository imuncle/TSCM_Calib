#ifndef MULTI_CALIB_H
#define MULTI_CALIB_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include "DS.h"

class MultiCalib_camera
{
    public:
    MultiCalib_camera(){}
    MultiCalib_camera(double cx, double cy, double fx, double fy, double xi, double alpha,
                      cv::Mat R, cv::Mat t, std::vector<bool> has_chessboard,
                      std::vector<std::vector<cv::Point2d>> pixel_coordinates){
        cv::Mat r;
        cv::Rodrigues(R, r);
        std::vector<double> rt{r.at<double>(0,0), r.at<double>(1,0), r.at<double>(2,0), t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0)};
        rt_ = rt;
        R_ = R;
        t_ = t;
        intrinsic_ = std::vector<double>{fx, fy, cx, cy, xi, alpha};
        cx_ = intrinsic_[2];
        cy_ = intrinsic_[3];
        fx_ = intrinsic_[0];
        fy_ = intrinsic_[1];
        xi_ = intrinsic_[4];
        alpha_ = intrinsic_[5];
        has_chessboard_ = has_chessboard;
        pixel_coordinates_ = pixel_coordinates;
        is_initial_ = true;
    }
    ~MultiCalib_camera(){}
    bool is_initial() {return is_initial_;}
    bool has_chessboard(int id) {return has_chessboard_[id];}
    std::vector<std::vector<cv::Point2d>> pixels() {return pixel_coordinates_;}
    void update_Rt(cv::Mat R, cv::Mat t) {R_ = R; t_ = t;}
    void update_param()
    {
        cx_ = intrinsic_[2];
        cy_ = intrinsic_[3];
        fx_ = intrinsic_[0];
        fy_ = intrinsic_[1];
        xi_ = intrinsic_[4];
        alpha_ = intrinsic_[5];
        cv::Mat r = (cv::Mat_<double>(3, 1) << rt_[0], rt_[1], rt_[2]);
        cv::Rodrigues(r, R_);
        t_ = (cv::Mat_<double>(3,1) << rt_[3], rt_[4], rt_[5]);
    }
    cv::Mat R() {return R_;}
    cv::Mat t() {return t_;}
    double cx() {return cx_;}
    double cy() {return cy_;}
    double fx() {return fx_*(1-alpha_);}
    double fy() {return fy_*(1-alpha_);}
    double xi() {return xi_;}
    double alpha() {return alpha_;}
    std::vector<double> intrinsic_;
    std::vector<double> rt_;
    private:
    std::vector<bool> has_chessboard_;
    std::vector<std::vector<cv::Point2d>> pixel_coordinates_;
    cv::Mat R_;
    cv::Mat t_;
    double cx_;
    double cy_;
    double fx_;
    double fy_;
    double xi_;
    double alpha_;
    bool is_initial_ = false;
};

class MultiCalib_chessboard
{
    public:
    MultiCalib_chessboard(){}
    MultiCalib_chessboard(cv::Mat R, cv::Mat t){
        cv::Mat r;
        cv::Rodrigues(R, r);
        std::vector<double> rt{r.at<double>(0,0), r.at<double>(1,0), r.at<double>(2,0), t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0)};
        rt_ = rt;
        R_ = R;
        t_ = t;
        is_initial_ = true;
    }
    ~MultiCalib_chessboard(){}
    bool is_initial() {return is_initial_;}
    void update_param() {
        cv::Mat r = (cv::Mat_<double>(3, 1) << rt_[0], rt_[1], rt_[2]);
        cv::Rodrigues(r, R_);
        t_ = (cv::Mat_<double>(3,1) << rt_[3], rt_[4], rt_[5]);
    }
    cv::Mat R() {return R_;}
    cv::Mat t() {return t_;}
    std::vector<double> rt_;
    private:
    std::vector<int> camera_id;
    cv::Mat R_;
    cv::Mat t_;
    bool is_initial_ = false;
};

class MultiCalib
{
    public:
    MultiCalib(std::vector<DoubleSphereCamera> cameras, const std::vector<cv::Point3d>& worlds);
    ~MultiCalib(){}
    void calibrate();
    void show_result();
    std::vector<MultiCalib_camera> cameras_;
    std::vector<MultiCalib_chessboard> chessboards_;
    std::vector<cv::Point3d> worlds_;
    private:
    void Rt_to_R_t(cv::Mat Rt, cv::Mat& R, cv::Mat& t)
    {
        cv::Vec3f r1(Rt.at<double>(0,0),Rt.at<double>(1,0),Rt.at<double>(2,0));
        cv::Vec3f r2(Rt.at<double>(0,1),Rt.at<double>(1,1),Rt.at<double>(2,1));
        cv::Vec3f r3 = r1.cross(r2);
        R = (cv::Mat_<double>(3,3) << r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]);
        t = (cv::Mat_<double>(3,1) << Rt.at<double>(0,2), Rt.at<double>(1,2), Rt.at<double>(2,2));
    }
    struct ReprojectionError
    {
        ReprojectionError(const cv::Point2d observ_p_, 
                          const cv::Point3d board_pts_)
            :observ_p(observ_p_), board_pts(board_pts_)
        {
        }

        template<typename T>
        bool operator()(const T* const camera_rt_,
            const T* const chessbaord_rt_,
            const T* const intrinsic_,
            T* residuls)const
        {
            // 先根据棋盘的外参和角点编号计算角点的世界坐标
            T chessboard_p[3];
            chessboard_p[0] = T(board_pts.x);
            chessboard_p[1] = T(board_pts.y);
            chessboard_p[2] = T(0.0);
            T chessboard_world_p[3];
            ceres::AngleAxisRotatePoint(chessbaord_rt_, chessboard_p, chessboard_world_p);
            chessboard_world_p[0] += chessbaord_rt_[3];
            chessboard_world_p[1] += chessbaord_rt_[4];
            chessboard_world_p[2] += chessbaord_rt_[5];
            // 根据相机的外参计算角点在相机坐标系下的坐标
            T chessboard_camera_p[3];
            ceres::AngleAxisRotatePoint(camera_rt_, chessboard_world_p, chessboard_camera_p);
            chessboard_camera_p[0] += camera_rt_[3];
            chessboard_camera_p[1] += camera_rt_[4];
            chessboard_camera_p[2] += camera_rt_[5];
            // 根据相机内参计算角点的投影坐标
            T one = T(1.0);
            T d1 = ceres::sqrt(chessboard_camera_p[0]*chessboard_camera_p[0]+chessboard_camera_p[1]*chessboard_camera_p[1]+chessboard_camera_p[2]*chessboard_camera_p[2]);
            T d2 = ceres::sqrt(chessboard_camera_p[0]*chessboard_camera_p[0]+chessboard_camera_p[1]*chessboard_camera_p[1]+(chessboard_camera_p[2]+intrinsic_[4]*d1)*(chessboard_camera_p[2]+intrinsic_[4]*d1));
            T pixel_x = intrinsic_[0] * (one-intrinsic_[5])*chessboard_camera_p[0]/(intrinsic_[5]*d2+(one-intrinsic_[5])*(intrinsic_[4]*d1+chessboard_camera_p[2])) + intrinsic_[2];
            T pixel_y = intrinsic_[1] * (one-intrinsic_[5])*chessboard_camera_p[1]/(intrinsic_[5]*d2+(one-intrinsic_[5])*(intrinsic_[4]*d1+chessboard_camera_p[2])) + intrinsic_[3];
            // 计算残差
            residuls[0] = T(observ_p.x) - pixel_x;
            residuls[1] = T(observ_p.y) - pixel_y;
            return true;
        }
        private:
        const cv::Point2d observ_p;
        const cv::Point3d board_pts;
	};
};

#endif
