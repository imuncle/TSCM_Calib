#include "findCorner.h"
#include <opencv2/opencv.hpp>
#include "DS.h"

int main(int argc, char **argv)
{
    int chessboard_size = 20;
    std::vector<std::vector<cv::Point2d>> pixel_coordinates;
    std::vector<cv::Point3d> world_coordinates;
    struct Chessboarder_t find_corner_chessboard;
    char str[30];
    cv::Size img_size;
    for(int i = 0; i < 11; i++)
    {
        sprintf(str, "../gopro/gopro%02d.jpg", i+1);
        cv::Mat img = cv::imread(str);
        img_size = img.size();
        find_corner_chessboard = findCorner(img, 2);
        if(find_corner_chessboard.chessboard.size() != 1) continue;
        std::vector<cv::Point2d> pixels_per_image;
        for (int u = 0; u < find_corner_chessboard.chessboard[0].rows; u++)
        {
            for (int v = 0; v < find_corner_chessboard.chessboard[0].cols; v++)
            {
                cv::circle(img, find_corner_chessboard.corners.p[find_corner_chessboard.chessboard[0].at<uint16_t>(u, v)], 5, cv::Scalar(0,0,255), 2);
                pixels_per_image.push_back(find_corner_chessboard.corners.p[find_corner_chessboard.chessboard[0].at<uint16_t>(u, v)]);
            }
        }
        pixel_coordinates.push_back(pixels_per_image);
        cv::line(img, find_corner_chessboard.corners.p[find_corner_chessboard.chessboard[0].at<uint16_t>(0, 0)],
                      find_corner_chessboard.corners.p[find_corner_chessboard.chessboard[0].at<uint16_t>(0, 1)], cv::Scalar(0,255,0), 2);
        cv::line(img, find_corner_chessboard.corners.p[find_corner_chessboard.chessboard[0].at<uint16_t>(0, 0)],
                      find_corner_chessboard.corners.p[find_corner_chessboard.chessboard[0].at<uint16_t>(1, 0)], cv::Scalar(255,0,0), 2);
        cv::imshow("img", img);
        int key = cv::waitKey(0);
        if(key == 'q')
            break;
    }
    for (int u = 0; u < find_corner_chessboard.chessboard[0].rows; u++)
    {
        for (int v = 0; v < find_corner_chessboard.chessboard[0].cols; v++)
        {
            world_coordinates.push_back(cv::Point3d(v * chessboard_size, u * chessboard_size, 0));
        }
    }
    DoubleSphereCamera ds_camera;
    ds_camera.calibrate(pixel_coordinates, world_coordinates, img_size);
    return 0;
}