#include "TS.h"

TripleSphereCamera::TripleSphereCamera()
{
    has_init_guess_ = false;
    cx_ = 0;
    cy_ = 0;
    fx_ = 0;
    fy_ = 0;
    xi_ = 0;
    alpha_ = 0;
    lamda_ = 0;
    b_ = c_ = 0;
    intrinsic_.resize(9);
}

TripleSphereCamera::TripleSphereCamera(double fx, double fy, double cx, double cy, double xi, double lamda, double alpha)
{
    has_init_guess_= true;
    cx_ = cx;
    cy_ = cy;
    fx_ = fx;
    fy_ = fy;
    xi_ = xi;
    lamda_ = lamda;
    alpha_ = alpha;
    intrinsic_.resize(9);
}

bool TripleSphereCamera::calibrate(const std::vector<std::vector<cv::Point2d>> pixels, 
                                   std::vector<bool> has_chessboard, 
                                   const std::vector<cv::Point3d>& worlds, 
                                   const cv::Size img_size, 
                                   const cv::Size chessboard_num)
{
    pixels_ = pixels;
    has_chessboard_ = has_chessboard;
    int img_num = pixels.size();
    Rt_.resize(img_num);
    rt_.resize(img_num);
    if(has_init_guess_ == false)
    {
        cx_ = img_size.width / 2 - 0.5;
        cy_ = img_size.height / 2 - 0.5;
        xi_ = 0.0;
        lamda_ = 0.0;
        alpha_ = 0.5;
        estimate_focal(pixels, worlds, img_size, chessboard_num);
        std::cout << "[initialize] focal:" << fx_ << ", cx:" << cx_ << ", cy:" << cy_ << ", xi:" << xi_ << ", lamda:" << lamda_ << ", alpha:" << alpha_ << std::endl;
        if(fx_ == 0) return false;
    }
    estimate_extrinsic(pixels, worlds, chessboard_num);
    intrinsic_[0] = fx_;
    intrinsic_[1] = fy_;
    intrinsic_[2] = cx_;
    intrinsic_[3] = cy_;
    intrinsic_[4] = xi_;
    intrinsic_[5] = lamda_;
    intrinsic_[6] = alpha_;
    intrinsic_[7] = b_;
    intrinsic_[8] = c_;
    for(int i = 0; i < Rt_.size(); i++)
    {
        if(has_chessboard_[i] == false)
            continue;
        cv::Vec3f r1(Rt_[i].at<double>(0,0),Rt_[i].at<double>(1,0),Rt_[i].at<double>(2,0));
        cv::Vec3f r2(Rt_[i].at<double>(0,1),Rt_[i].at<double>(1,1),Rt_[i].at<double>(2,1));
        cv::Vec3f r3 = r1.cross(r2);
        cv::Mat R = (cv::Mat_<double>(3,3) << r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]);
        cv::Mat r;
        cv::Rodrigues(R, r);
        std::vector<double> rt{r.at<double>(0,0), r.at<double>(1,0), r.at<double>(2,0), Rt_[i].at<double>(0,2), Rt_[i].at<double>(1,2), Rt_[i].at<double>(2,2)};
        rt_[i] = rt;
    }

    bool status = false;
    status = refinement(pixels, worlds);
    if(status) has_init_guess_ = true;

    fx_ = intrinsic_[0];
    fy_ = intrinsic_[1];
    cx_ = intrinsic_[2];
    cy_ = intrinsic_[3];
    xi_ = intrinsic_[4];
    lamda_ = intrinsic_[5];
    alpha_ = intrinsic_[6];
    b_ = intrinsic_[7];
    c_ = intrinsic_[8];
    for(int i = 0; i < Rt_.size(); i++)
    {
        if(has_chessboard_[i] == false)
            continue;
        cv::Mat R;
        cv::Mat r = (cv::Mat_<double>(3, 1) << rt_[i][0], rt_[i][1], rt_[i][2]);
        cv::Rodrigues(r, R);
        Rt_[i].at<double>(0,0) = R.at<double>(0,0);
        Rt_[i].at<double>(1,0) = R.at<double>(1,0);
        Rt_[i].at<double>(2,0) = R.at<double>(2,0);
        Rt_[i].at<double>(0,1) = R.at<double>(0,1);
        Rt_[i].at<double>(1,1) = R.at<double>(1,1);
        Rt_[i].at<double>(2,1) = R.at<double>(2,1);
        Rt_[i].at<double>(0,2) = rt_[i][3];
        Rt_[i].at<double>(1,2) = rt_[i][4];
        Rt_[i].at<double>(2,2) = rt_[i][5];
    }

    return status;
}

void TripleSphereCamera::estimate_focal(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds, cv::Size img_size, const cv::Size chessboard_num)
{
    double focal_ = 0;
    // 将像素坐标原点移到(cx, cy)
    std::vector<std::vector<cv::Point2d>> pixels_center;
    for(int i = 0; i < pixels.size(); i++)
    {
        std::vector<cv::Point2d> pixels_tmp;
        for(int j = 0; j < pixels[i].size(); j++)
        {
            cv::Point2d p(pixels[i][j].x - cx_, pixels[i][j].y - cy_);
            pixels_tmp.push_back(p);
        }
        pixels_center.push_back(pixels_tmp);
    }
    int total_num = 0;
    for(int k = 0; k < pixels_center.size(); k++)
    {
        if(pixels_center[k].size() == 0) continue;
        for(int i = 0; i < chessboard_num.height; i++)
        {
            cv::Mat P(cv::Size(4, chessboard_num.width), CV_64F);
            for(int j = 0; j < chessboard_num.width; j++)
            {
                double x = pixels_center[k][i*chessboard_num.width+j].x;
                double y = pixels_center[k][i*chessboard_num.width+j].y;
                P.at<double>(j, 0) = x;
                P.at<double>(j, 1) = y;
                P.at<double>(j, 2) = 0.5;
                P.at<double>(j, 3) = -0.5*(x*x+y*y);
            }
            cv::Mat C;
            cv::SVD::solveZ(P, C);
            double c1 = C.at<double>(0);
            double c2 = C.at<double>(1);
            double c3 = C.at<double>(2);
            double c4 = C.at<double>(3);
            double t = c1*c1 + c2*c2 + c3*c4;
            if(t < 0) continue;
            double d = std::sqrt(1/t);
            double nx = c1 * d;
            double ny = c2 * d;
            if(nx*nx+ny*ny > 0.95) continue;
            double nz = std::sqrt(1-nx*nx-ny*ny);
            double gamma = fabs(c3*d/nz);
            focal_ += gamma;
            total_num++;
        }
    }
    if(total_num > 0)
    {
        focal_ /= total_num;
    }
    else
    {
        std::cout << "焦距估计失败" << std::endl;
    }
    fx_ = fy_ = focal_;
}

void TripleSphereCamera::estimate_extrinsic(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds, const cv::Size chessboard_num)
{
    for(int k = 0; k < pixels.size(); k++)
    {
        if(has_chessboard_[k] == false) continue;
        std::vector<cv::Point2d> pixels_normalize;
        cv::Mat transform = cv::Mat::eye(3,3,CV_64F);
        std::vector<cv::Point2d> pixel = pixels[k];
        cv::Point3d p = get_unit_sphere_coordinate(pixel[pixel.size()/2-chessboard_num.width/2-1], transform);
        double alpha_angle = std::atan2(p.x, p.z);
        double beta_angle = std::asin(p.y);
        cv::Mat R1 = (cv::Mat_<double>(3,3) << cos(alpha_angle), 0, -sin(alpha_angle),
                                            0,                1,                 0,
                                            sin(alpha_angle), 0,  cos(alpha_angle));
        cv::Mat R2 = (cv::Mat_<double>(3,3) << 1, 0,                0,
                                            0, cos(beta_angle), -sin(beta_angle),
                                            0, sin(beta_angle),  cos(beta_angle));
        transform = R2 * R1;
        for(int i = 0; i < pixels[k].size(); i++)
        {
            cv::Point3d p = get_unit_sphere_coordinate(pixels[k][i], transform);
            pixels_normalize.push_back(cv::Point2d(p.x/p.z, p.y/p.z));
        }
        cv::Mat rvec, tvec, Rt;
        cv::solvePnPRansac(worlds, pixels_normalize, cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(4,0,CV_64F), rvec, tvec);
        cv::Rodrigues(rvec, Rt);
        Rt = transform.t() * Rt;
        tvec = transform.t() * tvec;
        Rt.at<double>(0,2) = tvec.at<double>(0);
        Rt.at<double>(1,2) = tvec.at<double>(1);
        Rt.at<double>(2,2) = tvec.at<double>(2);
        Rt_[k] = Rt;
    }
}

double TripleSphereCamera::ReprojectError(const std::vector<cv::Point2d>& pixels, const std::vector<cv::Point3d>& worlds, cv::Mat Rt,
                          double cx, double cy, double fx, double fy, double xi, double lamda, double alpha, double b, double c)
{
    double error = 0;
    for(int i = 0; i < worlds.size(); i++)
    {
        cv::Mat P = (cv::Mat_<double>(3,1) << worlds[i].x, worlds[i].y, 1);
        P = Rt * P;
        double X = P.at<double>(0,0);
        double Y = P.at<double>(1,0);
        double Z = P.at<double>(2,0);
        double d1 = std::sqrt(X*X+Y*Y+Z*Z);
        double d2 = std::sqrt(X*X+Y*Y+std::pow(Z+xi*d1,2));
        double d3 = std::sqrt(X*X+Y*Y+std::pow(Z+xi*d1+lamda*d2,2));
        double ksai = Z+xi*d1+lamda*d2+alpha/(1-alpha)*d3;
        double pixel_x = fx * X/ksai + b * Y/ksai + cx;
        double pixel_y = c * X/ksai + fy * Y/ksai + cy;
        error += std::sqrt((pixels[i].x-pixel_x)*(pixels[i].x-pixel_x) + (pixels[i].y-pixel_y)*(pixels[i].y-pixel_y));
    }
    return error / worlds.size();
}

void TripleSphereCamera::Reproject(const std::vector<cv::Point3d>& worlds, cv::Mat Rt, std::vector<cv::Point2d>& pixels)
{
    pixels.resize(worlds.size());
    for(int i = 0; i < worlds.size(); i++)
    {
        cv::Mat P = (cv::Mat_<double>(3,1) << worlds[i].x, worlds[i].y, 1);
        P = Rt * P;
        double X = P.at<double>(0,0);
        double Y = P.at<double>(1,0);
        double Z = P.at<double>(2,0);
        double d1 = std::sqrt(X*X+Y*Y+Z*Z);
        double d2 = std::sqrt(X*X+Y*Y+std::pow(Z+xi_*d1,2));
        double d3 = std::sqrt(X*X+Y*Y+std::pow(Z+xi_*d1+lamda_*d2,2));
        double ksai = Z+xi_*d1+lamda_*d2+alpha_/(1-alpha_)*d3;
        double pixel_x = fx_ * X/ksai + b_ * Y/ksai + cx_;
        double pixel_y = c_ * X/ksai + fy_ * Y/ksai + cy_;
        pixels[i] = cv::Point2d(pixel_x, pixel_y);
    }
}

bool TripleSphereCamera::refinement(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds)
{
    ceres::Problem problem;

    for (int i = 0; i < pixels.size(); i++)
    {
        if(has_chessboard_[i] == false)
            continue;
        for(int j = 0; j < pixels[i].size(); j++)
        {
            ReprojectionError *cost_function =
                new ReprojectionError(pixels[i][j], worlds[j]);

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<
                    ReprojectionError,
                    2,  // num_residuals
                    9,6>(cost_function),
                NULL,
                intrinsic_.data(),
                rt_[i].data());
        }
    }
    // Configure the solver.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout =false;
    options.max_num_iterations = 100;

    // Solve!
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    return summary.termination_type == ceres::CONVERGENCE;
}

void TripleSphereCamera::undistort(double fx, double fy, double cx, double cy, cv::Size img_size, cv::Mat& mapx, cv::Mat& mapy)
{
    mapx = cv::Mat(img_size, CV_32FC1);
    mapy = cv::Mat(img_size, CV_32FC1);
    for(int i = 0; i < img_size.height; i++)
    {
        for(int j = 0; j < img_size.width; j++)
        {
            // 先根据针孔相机模型计算入射光线向量
            double x = (j - cx) / fx;
            double y = (i - cy) / fy;
            double z = 1.0;
            double d1 = std::sqrt(x*x+y*y+z*z);
            double d2 = std::sqrt(x*x+y*y+std::pow(z+xi_*d1,2));
            double d3 = std::sqrt(x*x+y*y+std::pow(z+xi_*d1+lamda_*d2,2));
            double ksai = z+xi_*d1+lamda_*d2+alpha_/(1-alpha_)*d3;
            double pixel_x = fx_ * x/ksai + b_ * y/ksai + cx_;
            double pixel_y = c_ * x/ksai + fy_ * y/ksai + cy_;
            mapx.at<float>(i, j) = pixel_x;
            mapy.at<float>(i, j) = pixel_y;
        }
    }
}

cv::Mat TripleSphereCamera::undistort_chessboard(cv::Mat src, int index, cv::Size chessboard, double chessboard_size)
{
    cv::Mat dst;
    if(has_chessboard_[index] == false)
        return dst;
    cv::Size img_size((chessboard.width+1)*chessboard_size, (chessboard.height+1)*chessboard_size);
    cv::Mat mapx = cv::Mat(img_size, CV_32FC1);
    cv::Mat mapy = cv::Mat(img_size, CV_32FC1);
    cv::Mat Rt = Rt_[index];
    for(int i = 0; i < img_size.height; i++)
    {
        for(int j = 0; j < img_size.width; j++)
        {
            cv::Mat P = (cv::Mat_<double>(3,1) << (j-chessboard_size), (i-chessboard_size), 1);
            P = Rt*P;
            cv::Point2d pixel = project(P);
            mapx.at<float>(i, j) = pixel.x;
            mapy.at<float>(i, j) = pixel.y;
        }
    }
    cv::remap(src, dst, mapx, mapy, cv::INTER_LINEAR);
    return dst;
}

cv::Point2d TripleSphereCamera::project(cv::Mat P)
{
    double X = P.at<double>(0,0);
    double Y = P.at<double>(1,0);
    double Z = P.at<double>(2,0);
    double d1 = std::sqrt(X*X+Y*Y+Z*Z);
    double d2 = std::sqrt(X*X+Y*Y+std::pow(Z+xi_*d1,2));
    double d3 = std::sqrt(X*X+Y*Y+std::pow(Z+xi_*d1+lamda_*d2,2));
    double ksai = Z+xi_*d1+lamda_*d2+alpha_/(1-alpha_)*d3;
    double pixel_x = fx_ * X/ksai + b_ * Y/ksai + cx_;
    double pixel_y = c_ * X/ksai + fy_ * Y/ksai + cy_;
    return cv::Point2d(pixel_x, pixel_y);
}