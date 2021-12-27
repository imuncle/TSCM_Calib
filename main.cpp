#include "findCorner.h"
#include <opencv2/opencv.hpp>
#include "DS.h"
#include "multi_calib.hpp"

std::vector<cv::Point3d> monocular_calib(std::vector<std::string> filenames, double chessboard_size, cv::Size chessboard_num, DoubleSphereCamera& ds_camera)
{
    std::vector<bool> has_chessboard;
    cv::Size img_size;
    std::vector<std::vector<cv::Point2d>> pixel_coordinates;
    std::vector<cv::Point3d> world_coordinates;
    struct Chessboarder_t find_corner_chessboard;
    for(int i = 0; i < filenames.size(); i++)
    {
        std::cout << filenames[i].data() << std::endl;
        cv::Mat img = cv::imread(filenames[i].data());
        img_size = img.size();
        std::vector<cv::Point2d> pixels_per_image;
        find_corner_chessboard = findCorner(img, 2);
        if((find_corner_chessboard.chessboard.size() == 0) || 
            (find_corner_chessboard.chessboard.size() > 1 || find_corner_chessboard.chessboard[0].rows != chessboard_num.height || find_corner_chessboard.chessboard[0].cols != chessboard_num.width))
        {
            has_chessboard.push_back(false);
            pixel_coordinates.push_back(pixels_per_image);
            continue;
        }
        has_chessboard.push_back(true);
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
        int key = cv::waitKey(1);
        if(key == 'q')
            break;
    }
    for (int u = 0; u < chessboard_num.height; u++)
    {
        for (int v = 0; v < chessboard_num.width; v++)
        {
            world_coordinates.push_back(cv::Point3d(v * chessboard_size, u * chessboard_size, 0));
        }
    }
    ds_camera.calibrate(pixel_coordinates, has_chessboard, world_coordinates, img_size);
    std::cout << "fx:" << ds_camera.fx() << ", fy:" << ds_camera.fy() << ", cx:" << ds_camera.cx() << ", cy:" << ds_camera.cy() << ", alpha:" << ds_camera.alpha() << ", xi:" << ds_camera.xi() << std::endl;
    std::vector<cv::Mat> Rt = ds_camera.Rt();
    double error = 0;
    for(int i = 0; i < Rt.size(); i++)
    {
        if(has_chessboard[i] == false)
            continue;
        cv::Mat img = cv::imread(filenames[i].data());
        std::vector<cv::Point2d> pixels_reproject;
        ds_camera.Reproject(world_coordinates, Rt[i], pixels_reproject);
        for (int j = 0; j < pixels_reproject.size(); j++)
        {
            cv::circle(img, pixels_reproject[j], 5, cv::Scalar(0,0,255), 2);
            error += std::sqrt((pixels_reproject[j].x-pixel_coordinates[i][j].x)*(pixels_reproject[j].x-pixel_coordinates[i][j].x)+
                               (pixels_reproject[j].y-pixel_coordinates[i][j].y)*(pixels_reproject[j].y-pixel_coordinates[i][j].y));
        }
        cv::imshow("img", img);
        int key = cv::waitKey(0);
    }
    error /= (Rt.size() * world_coordinates.size());
    std::cout << "mean error: " << error << std::endl;
    cv::Mat mapx, mapy;
    ds_camera.undistort(ds_camera.fx(), ds_camera.fy(), ds_camera.cx(), ds_camera.cy(), img_size, mapx, mapy);
    for(int i = 0; i < Rt.size(); i++)
    {
        cv::Mat img = cv::imread(filenames[i].data());
        cv::remap(img, img, mapx, mapy, cv::INTER_LINEAR);
        cv::imshow("img", img);
        int key = cv::waitKey(0);
        if(key == 'q')
            break;
    }
    return world_coordinates;
}

int main(int argc, char **argv)
{
    int chessboard_size = 30;
    cv::Size chessboard_num(11, 8);
    char str[40];
    std::vector<std::string> filenames;
    for(int i = 0; i < 81; i++)
    {
        sprintf(str, "../mul/front/%02d.jpg", i);
        std::string name = str;
        filenames.push_back(name);
    }
    DoubleSphereCamera camera_1;
    std::vector<cv::Point3d> world_coordinates = monocular_calib(filenames, chessboard_size, chessboard_num, camera_1);
    for(int i = 0; i < 81; i++)
    {
        sprintf(str, "../mul/left/%02d.jpg", i);
        std::string name = str;
        filenames[i] = name;
    }
    DoubleSphereCamera camera_2;
    monocular_calib(filenames, chessboard_size, chessboard_num, camera_2);
    for(int i = 0; i < 81; i++)
    {
        sprintf(str, "../mul/right/%02d.jpg", i);
        std::string name = str;
        filenames[i] = name;
    }
    DoubleSphereCamera camera_4;
    monocular_calib(filenames, chessboard_size, chessboard_num, camera_4);
    std::vector<DoubleSphereCamera> cameras;
    cameras.push_back(camera_1);
    cameras.push_back(camera_2);
    cameras.push_back(camera_4);
    MultiCalib mul_calib = MultiCalib(cameras);
    std::cout << "start multi-camera calibration..." << std::endl;
    mul_calib.calibrate(world_coordinates);
    mul_calib.show_result();
    return 0;
}
