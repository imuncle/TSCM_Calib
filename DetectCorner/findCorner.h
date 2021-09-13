#ifndef FIND_CORNER_H
#define FIND_CORNER_H

#include <opencv2/opencv.hpp>
#include "chessboard.h"

struct Template_t
{
    cv::Mat a1;
    cv::Mat a2;
    cv::Mat b1;
    cv::Mat b2;
};

struct Chessboarder_t
{
    struct Corner_t corners;
    std::vector<cv::Mat> chessboard;
};

struct Chessboarder_t findCorner(cv::Mat img, int sigma);
void secondDerivCornerMetric(cv::Mat I, int sigma, cv::Mat* cxy, cv::Mat* c45, cv::Mat* Ix, cv::Mat* Iy, cv::Mat* Ixy, cv::Mat* I_45_45);
std::vector<cv::Point2d> nonMaximumSuppression(cv::Mat img, int n, double tau, int margin);
struct Corner_t getOrientations(cv::Mat img_angle, cv::Mat img_weight, struct Corner_t corners, int r);
std::vector<cv::Point2d> edgeOrientations(cv::Mat img_angle, cv::Mat img_weight);
std::vector<cv::Point3d> findModesMeanShift(cv::Mat hist, int sigma);
double cornerCorrelationScore(cv::Mat img, cv::Mat img_weight, cv::Point2d v1, cv::Point2d v2);
struct Corner_t scoreCorners(cv::Mat img, cv::Mat img_weight, struct Corner_t corners, int* radius);
struct Corner_t subPixelLocation(cv::Mat metric, struct Corner_t corners);

#endif
