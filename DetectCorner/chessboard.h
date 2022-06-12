#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <opencv2/opencv.hpp>

struct Corner_t
{
    std::vector<cv::Point2d> p;
    std::vector<cv::Point2d> v1;
    std::vector<cv::Point2d> v2;
    std::vector<double> score;
};

std::vector<cv::Mat> chessboardsFromCorners(struct Corner_t corners);
cv::Mat initChessboard(struct Corner_t corners, int idx);
double average(double* a, int length);
double stdd(double* a, int length, double mean);
void directionalNeighbor(int idx, cv::Point2d v, cv::Mat chessboard, struct Corner_t corners, int* neighbor_idx, double* min_dist);
double chessboardEnergy(cv::Mat chessboard, struct Corner_t corners);
cv::Mat growChessboard(cv::Mat chessboard, struct Corner_t corners, int boarder_type);
cv::Point2d predictCorners(cv::Point2d p1, cv::Point2d p2, cv::Point2d p3);
std::vector<int> assignClosestCorners(std::vector<cv::Point2d> cand, std::vector<cv::Point2d> pred);
void plotChessboards(cv::Mat img, std::vector<cv::Mat> chessboards, struct Corner_t corners);

#endif
