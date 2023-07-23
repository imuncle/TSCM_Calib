#include "findCorner.h"
#include <opencv2/opencv.hpp>
#include "TS.h"
#include "multi_calib.h"
#include "cout_style.h"
#include <time.h>

std::vector<cv::Point3d> monocular_calib(std::vector<std::string> filenames, double chessboard_size, cv::Size chessboard_num, TripleSphereCamera& ds_camera)
{
    
    std::vector<cv::Point3d> world_coordinates;
    for (int u = 0; u < chessboard_num.height; u++)
    {
        for (int v = 0; v < chessboard_num.width; v++)
        {
            world_coordinates.push_back(cv::Point3d(v * chessboard_size, u * chessboard_size, 0));
        }
    }
    std::vector<bool> has_chessboard;
    cv::Size img_size;
    std::vector<std::vector<cv::Point2d>> pixel_coordinates;
    std::vector<std::string> imgs_filename;
    std::cout << blue << "检测角点..." << reset << std::endl;
    struct Chessboarder_t find_corner_chessboard;
    for(int i = 0; i < filenames.size(); i++)
    {
        cv::Mat img = cv::imread(filenames[i].data());
        if(img.empty()) continue;
        imgs_filename.push_back(filenames[i]);
        img_size = img.size();
        std::vector<cv::Point2d> pixels_per_image;
        find_corner_chessboard = findCorner(img, 4);
        if(find_corner_chessboard.chessboard.size() != 1 || find_corner_chessboard.chessboard[0].size() != chessboard_num)
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
        cv::line(img, pixels_per_image[0],
                      pixels_per_image[1], cv::Scalar(0,255,0), 2);
        cv::line(img, pixels_per_image[0],
                      pixels_per_image[chessboard_num.width], cv::Scalar(255,0,0), 2);
        cv::imshow("img", img);
        cv::waitKey(1);
    }
    ds_camera.calibrate(pixel_coordinates, has_chessboard, world_coordinates, img_size, chessboard_num);
    // 精细化角点
    std::vector<cv::Mat> Rts;
    Rts = ds_camera.Rt();
    std::cout << blue << "精细化角点..." << reset << std::endl;
    for(int i = 0; i < imgs_filename.size(); i++)
    {
        cv::Mat img = cv::imread(imgs_filename[i].data());
        cv::Mat chessboard_img = ds_camera.undistort_chessboard(img, i, chessboard_num, chessboard_size);
        if(chessboard_img.empty()) continue;
        for (int u = 0; u < pixel_coordinates[i].size(); u++)
        {
            cv::circle(img, pixel_coordinates[i][u], 5, cv::Scalar(0,0,255), 2);
        }
        struct Chessboarder_t find_chessboard = findCorner(chessboard_img, 4);
        if(find_chessboard.chessboard.size() == 0 || find_chessboard.chessboard[0].size() != chessboard_num)
        {
            // 纠正翻转的棋盘，保证棋盘左上角是黑块（要求棋盘格行列数分别为奇数和偶数）
            cv::Mat gray;
            cv::cvtColor(chessboard_img, gray, cv::COLOR_BGR2GRAY);
            cv::Point2d p1(chessboard_size/2, chessboard_size/2);
            cv::Point2d p2(chessboard_size*3/2, chessboard_size/2);
            cv::Point2d p3(chessboard_size*3/2, chessboard_size*3/2);
            cv::Point2d p4(chessboard_size/2, chessboard_size*3/2);
            int gray1 = gray.at<uchar>(int(p1.y), int(p1.x));
            int gray2 = gray.at<uchar>(int(p2.y), int(p2.x));
            int gray3 = gray.at<uchar>(int(p3.y), int(p3.x));
            int gray4 = gray.at<uchar>(int(p4.y), int(p4.x));
            if(gray1 + gray3 > gray2 + gray4)
            {
                cv::flip(chessboard_img, chessboard_img, -1);
                std::vector<cv::Point2d> tmp = pixel_coordinates[i];
                for(int k = 0; k < pixel_coordinates[i].size(); k++)
                    pixel_coordinates[i][k] = tmp[pixel_coordinates[i].size()-1-k];
            }
            continue;
        }
        for (int u = 0; u < find_chessboard.chessboard[0].rows; u++)
        {
            for (int v = 0; v < find_chessboard.chessboard[0].cols; v++)
            {
                cv::Point2d new_corner = find_chessboard.corners.p[find_chessboard.chessboard[0].at<uint16_t>(u, v)];
                cv::circle(chessboard_img, new_corner, 5, cv::Scalar(0,0,255), 2);
                cv::Mat p = (cv::Mat_<double>(3,1) << new_corner.x-chessboard_size, new_corner.y-chessboard_size, 1);
                p = Rts[i]*p;
                new_corner = ds_camera.project(p);
                pixel_coordinates[i][v+u*chessboard_num.width] = new_corner;
                cv::circle(img, new_corner, 2, cv::Scalar(0,255,0), 2);
            }
        }
        // 纠正翻转的棋盘，保证棋盘左上角是黑块（要求棋盘格行列数分别为奇数和偶数）
        cv::Mat gray;
        cv::cvtColor(chessboard_img, gray, cv::COLOR_BGR2GRAY);
        cv::Point2d p1(chessboard_size/2, chessboard_size/2);
        cv::Point2d p2(chessboard_size*3/2, chessboard_size/2);
        cv::Point2d p3(chessboard_size*3/2, chessboard_size*3/2);
        cv::Point2d p4(chessboard_size/2, chessboard_size*3/2);
        int gray1 = gray.at<uchar>(int(p1.y), int(p1.x));
        int gray2 = gray.at<uchar>(int(p2.y), int(p2.x));
        int gray3 = gray.at<uchar>(int(p3.y), int(p3.x));
        int gray4 = gray.at<uchar>(int(p4.y), int(p4.x));
        if(gray1 + gray3 > gray2 + gray4)
        {
            cv::flip(chessboard_img, chessboard_img, -1);
            std::vector<cv::Point2d> tmp = pixel_coordinates[i];
            for(int k = 0; k < pixel_coordinates[i].size(); k++)
                pixel_coordinates[i][k] = tmp[pixel_coordinates[i].size()-1-k];
        }
        cv::imshow("chessboard", chessboard_img);
        cv::imshow("img", img);
        cv::waitKey(1);
    }
    ds_camera.calibrate(pixel_coordinates, has_chessboard, world_coordinates, img_size, chessboard_num);
    std::cout << yellow;
    std::cout << "fx:" << ds_camera.fx() << ", fy:" << ds_camera.fy() << ", cx:" << ds_camera.cx() << ", cy:" << ds_camera.cy() << ", xi:" << ds_camera.xi() 
              << ", lamda:" << ds_camera.lamda() << ", alpha:" << ds_camera.alpha() 
              << ", b:" << ds_camera.b() << ", c:" << ds_camera.c() << std::endl;
    std::cout << reset;
    
    Rts = ds_camera.Rt();
    double error = 0;
    int cnt = 0;
    for(int i = 0; i < Rts.size(); i++)
    {
        if(has_chessboard[i] == false)
            continue;
        cv::Mat img = cv::imread(filenames[i].data());
        cv::Mat chessboard_img = ds_camera.undistort_chessboard(img, i, chessboard_num, chessboard_size);
        for(int m = 0; m < chessboard_num.width; m++)
        {
            cv::line(chessboard_img, cv::Point(chessboard_size*(m+1), 0), 
                                     cv::Point(chessboard_size*(m+1), chessboard_size*(chessboard_num.height+1)), cv::Scalar(0,255,0), 1);
        }
        for(int m = 0; m < chessboard_num.height; m++)
        {
            cv::line(chessboard_img, cv::Point(0, chessboard_size*(m+1)), 
                                     cv::Point(chessboard_size*(chessboard_num.width+1), chessboard_size*(m+1)), cv::Scalar(0,255,0), 1);
        }
        for(int j = 0; j < pixel_coordinates[i].size(); j++)
        {
            cv::Point3d p = ds_camera.get_unit_sphere_coordinate(pixel_coordinates[i][j]);
            cv::Mat Rt = Rts[i];
            cv::Mat pp = (cv::Mat_<double>(3,1) << p.x, p.y, p.z);
            pp = Rt.inv() * pp;
            cv::Point2d point(pp.at<double>(0)/pp.at<double>(2), pp.at<double>(1)/pp.at<double>(2));
            cv::circle(chessboard_img, point+cv::Point2d(chessboard_size, chessboard_size), 5, cv::Scalar(0,0,255), 2);
            int x = j%chessboard_num.width;
            int y = (j-x)/chessboard_num.width;
        }
        std::vector<cv::Point2d> pixels_reproject;
        ds_camera.Reproject(world_coordinates, Rts[i], pixels_reproject);
        for (int j = 0; j < pixels_reproject.size(); j++)
        {
            cv::circle(img, pixel_coordinates[i][j], 5, cv::Scalar(0,0,255), 2);
            cv::circle(img, pixels_reproject[j], 5, cv::Scalar(0,255,255), 2);
            cv::line(img, pixel_coordinates[i][0],
                        pixel_coordinates[i][1], cv::Scalar(0,255,0), 2);
            cv::line(img, pixel_coordinates[i][0],
                        pixel_coordinates[i][chessboard_num.width], cv::Scalar(255,0,0), 2);
            error += std::sqrt((pixels_reproject[j].x-pixel_coordinates[i][j].x)*(pixels_reproject[j].x-pixel_coordinates[i][j].x)+
                               (pixels_reproject[j].y-pixel_coordinates[i][j].y)*(pixels_reproject[j].y-pixel_coordinates[i][j].y));
            cnt++;
        }
        cv::imshow("img", img);
        cv::imshow("chessboard", chessboard_img);
        cv::waitKey(1);
    }
    std::cout << yellow << "mean error: " << error/cnt << reset << std::endl;
    return world_coordinates;
}

int main(int argc, char **argv)
{
    int chessboard_size = 45;
    cv::Size chessboard_num(11, 8);
    char str[80];
    std::vector<std::string> filenames1;
    for(int i = 0; i < 185; i++)
    {
        sprintf(str, "/home/xiesheng/Downloads/calib_4/video/front/%d.jpg", i);
        std::string name = str;
        filenames1.push_back(name);
    }
    TripleSphereCamera camera_1;
    std::cout << green << "开始标定相机1..." << reset << std::endl;
    std::vector<cv::Point3d> world_coordinates = monocular_calib(filenames1, chessboard_size, chessboard_num, camera_1);

    std::vector<std::string> filenames2;
    for(int i = 0; i < 185; i++)
    {
        sprintf(str, "/home/xiesheng/Downloads/calib_4/video/right/%d.jpg", i);
        std::string name = str;
        filenames2.push_back(name);
    }
    TripleSphereCamera camera_2;
    std::cout << green << "开始标定相机2..." << reset << std::endl;
    monocular_calib(filenames2, chessboard_size, chessboard_num, camera_2);

    std::vector<std::string> filenames3;
    for(int i = 0; i < 185; i++)
    {
        sprintf(str, "/home/xiesheng/Downloads/calib_4/video/rear/%d.jpg", i);
        std::string name = str;
        filenames3.push_back(name);
    }
    TripleSphereCamera camera_3;
    std::cout << green << "开始标定相机3..." << reset << std::endl;
    monocular_calib(filenames3, chessboard_size, chessboard_num, camera_3);

    std::vector<std::string> filenames4;
    for(int i = 0; i < 185; i++)
    {
        sprintf(str, "/home/xiesheng/Downloads/calib_4/video/left/%d.jpg", i);
        std::string name = str;
        filenames4.push_back(name);
    }
    TripleSphereCamera camera_4;
    std::cout << green << "开始标定相机4..." << reset << std::endl;
    monocular_calib(filenames4, chessboard_size, chessboard_num, camera_4);
    
    std::vector<TripleSphereCamera> cameras;
    
    cameras.push_back(camera_1);
    cameras.push_back(camera_2);
    cameras.push_back(camera_3);
    cameras.push_back(camera_4);
    MultiCalib mul_calib = MultiCalib(cameras, world_coordinates);
    std::cout << green <<"开始联合标定..." << reset << std::endl;
    std::cout << "多相机标定前..." << std::endl;
    for(int m = 0; m < mul_calib.cameras_.size(); m++)
    {
        std::vector<std::vector<cv::Point2d>> pixels = mul_calib.cameras_[m].pixels();
        double error = 0;
        int cnt = 0;
        for(int i = 0; i < mul_calib.chessboards_.size(); i++)
        {
            if(mul_calib.chessboards_[i].is_initial())
            {
                for(int j = 0; j < pixels[i].size(); j++)
                {
                    cv::Mat R = mul_calib.chessboards_[i].R();
                    cv::Mat t = mul_calib.chessboards_[i].t();
                    cv::Mat p = (cv::Mat_<double>(3, 1) << world_coordinates[j].x, world_coordinates[j].y, world_coordinates[j].z);
                    p = R * p + t;
                    R = mul_calib.cameras_[m].R();
                    t = mul_calib.cameras_[m].t();
                    p = R * p + t;
                    double fx = mul_calib.cameras_[m].fx();
                    double fy = mul_calib.cameras_[m].fy();
                    double cx = mul_calib.cameras_[m].cx();
                    double cy = mul_calib.cameras_[m].cy();
                    double alpha = mul_calib.cameras_[m].alpha();
                    double lamda = mul_calib.cameras_[m].lamda();
                    double xi = mul_calib.cameras_[m].xi();
                    double b = mul_calib.cameras_[m].b();
                    double c = mul_calib.cameras_[m].c();
                    double X = p.at<double>(0,0);
                    double Y = p.at<double>(1,0);
                    double Z = p.at<double>(2,0);
                    double d1 = std::sqrt(X*X+Y*Y+Z*Z);
                    double d2 = std::sqrt(X*X+Y*Y+std::pow(Z+xi*d1,2));
                    double d3 = std::sqrt(X*X+Y*Y+std::pow(Z+xi*d1+lamda*d2,2));
                    double ksai = Z+xi*d1+lamda*d2+alpha/(1-alpha)*d3;
                    double pixel_x = fx * X/ksai + b * Y/ksai + cx;
                    double pixel_y = c * X/ksai + fy * Y/ksai + cy;
                    error += std::sqrt((pixels[i][j].x-pixel_x)*(pixels[i][j].x-pixel_x) + (pixels[i][j].y-pixel_y)*(pixels[i][j].y-pixel_y));
                    cnt++;
                }
            }
        }
        std::cout << "camera_" << m << " 重投影误差：" << error/cnt << std::endl;
    }
    mul_calib.calibrate();
    mul_calib.show_result();

    // save calibration result
    time_t tt;
    time( &tt );
    tt = tt + 8*3600;  // transform the time zone
    tm* t= gmtime( &tt );
    char yaml_filename[100];
    sprintf(yaml_filename, "%d-%02d-%02d %02d-%02d-%02d.yaml",
            t->tm_year + 1900,
            t->tm_mon + 1,
            t->tm_mday,
            t->tm_hour,
            t->tm_min,
            t->tm_sec);
    cv::FileStorage fs_write(yaml_filename, cv::FileStorage::WRITE);
    for(size_t i = 0; i < mul_calib.cameras_.size(); i++)
    {
        char cam_name[10];
        sprintf(cam_name, "cam%d", (int)i);
        fs_write << cam_name << mul_calib.cameras_[i].intrinsic_matrix_;
        sprintf(cam_name, "Twc%d", (int)i);
        cv::Mat_<double> R = mul_calib.cameras_[i].R();
        cv::Mat_<double> t = mul_calib.cameras_[i].t();
        cv::Mat T = (cv::Mat_<double>(3,4) << R(0,0), R(0,1), R(0,2), t(0),
                                              R(1,0), R(1,1), R(1,2), t(1),
                                              R(2,0), R(2,1), R(2,2), t(2));
        fs_write << cam_name << T;
    }
    fs_write.release();
    return 0;
}
