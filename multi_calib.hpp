#ifndef MULTI_CALIB_H
#define MULTI_CALIB_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace multi_calib
{
    struct ReprojectionError
    {
        ReprojectionError(const cv::Point2d observ_p_, 
                          const cv::Point corner_id_, 
                          const double chessboard_size_, 
                          const double fx_, 
                          const double fy_, 
                          const double cx_, 
                          const double cy_, 
                          const double xi_, 
                          const double alpha_)
            :observ_p(observ_p_), corner_id(corner_id_), chessboard_size(chessboard_size_), fx(fx_), fy(fy_), cx(cx_), cy(cy_), xi(xi_), alpha(alpha_)
        {
        }

        template<typename T>
        bool operator()(const T* const camera_rt_,
            const T* const chessbaord_rt_,
            T* residuls)const
        {
            // 先根据棋盘的外参和角点编号计算角点的世界坐标
            // 根据相机的外参计算角点在相机坐标系下的坐标
            // 根据相机内参计算角点的投影坐标
            // 计算残差
            // residuls
            residuls[0] = T(0);
            residuls[1] = T(0);
            return true;
        }
        private:
        const cv::Point2d observ_p;
        const cv::Point corner_id;
        const double chessboard_size;
        const double fx;
        const double fy;
        const double cx;
        const double cy;
        const double xi;
        const double alpha;
	};
}

#endif