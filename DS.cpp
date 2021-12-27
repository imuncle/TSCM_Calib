#include "DS.h"
#include "multi_calib.hpp"

DoubleSphereCamera::DoubleSphereCamera()
{
    has_init_guess_ = false;
    cx_ = 0;
    cy_ = 0;
    fx_ = 0;
    fy_ = 0;
    xi_ = 0;
    alpha_ = 0;
    intrinsic_.resize(6);
}

DoubleSphereCamera::DoubleSphereCamera(double fx, double fy, double cx, double cy, double alpha, double xi)
{
    has_init_guess_= true;
    cx_ = cx;
    cy_ = cy;
    fx_ = fx;
    fy_ = fy;
    xi_ = xi;
    alpha_ = alpha;
    intrinsic_.resize(6);
}

void DoubleSphereCamera::calibrate(const std::vector<std::vector<cv::Point2d>> pixels, std::vector<bool> has_chessboard, const std::vector<cv::Point3d>& worlds, const cv::Size img_size)
{
    pixels_ = pixels;
    has_chessboard_ = has_chessboard;
    int img_num = pixels.size();
    Rt_.resize(img_num);
    gammas_.resize(img_num);
    rt_.resize(img_num);
    if(has_init_guess_ == false)
    {
        cx_ = img_size.width / 2 - 0.5;
        cy_ = img_size.height / 2 - 0.5;
        xi_ = 0;
        alpha_ = 0.5;
        initialize_param(pixels_, worlds);
        has_init_guess_ = true;
    }
    else
    {
        initialize_param(pixels_, worlds);
    }
    intrinsic_[0] = fx_;
    intrinsic_[1] = fy_;
    intrinsic_[2] = cx_;
    intrinsic_[3] = cy_;
    intrinsic_[4] = xi_;
    intrinsic_[5] = alpha_;
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
    refinement(pixels_, worlds);

    fx_ = intrinsic_[0];
    fy_ = intrinsic_[1];
    cx_ = intrinsic_[2];
    cy_ = intrinsic_[3];
    xi_ = intrinsic_[4];
    alpha_ = intrinsic_[5];
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
}

void DoubleSphereCamera::initialize_param(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds)
{
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
    for(int i = 0; i < pixels_center.size(); i++)
    {
        if(has_chessboard_[i] == false)
            continue;
        int point_num = pixels_center[i].size();
        cv::Mat A(cv::Size(6, point_num), CV_64F);
        // x = [r11, r12, r21, r22, t1, t2]
        cv::Mat x(cv::Size(1, 6), CV_64F);
        for(int j = 0; j < point_num; j++)
        {
            A.at<double>(j, 0) = -pixels_center[i][j].y * worlds[j].x;
            A.at<double>(j, 1) = pixels_center[i][j].x * worlds[j].x;
            A.at<double>(j, 2) = -pixels_center[i][j].y * worlds[j].y;
            A.at<double>(j, 3) = pixels_center[i][j].x * worlds[j].y;
            A.at<double>(j, 4) = -pixels_center[i][j].y;
            A.at<double>(j, 5) = pixels_center[i][j].x;
        }
        cv::SVD::solveZ(A, x);
        double r11 = x.at<double>(0, 0);
        double r12 = x.at<double>(1, 0);
        double r21 = x.at<double>(2, 0);
        double r22 = x.at<double>(3, 0);
        double t1 = x.at<double>(4, 0);
        double t2 = x.at<double>(5, 0);
        double AA = std::pow(r11*r21 + r12*r22, 2);
        double BB = r11*r11 + r12*r12;
        double CC = r21*r21 + r22*r22;

        // r11^2 + r12^2 + r13^2 = 1
        // r21^2 + r22^2 + r23^2 = 1
        // r11*r21 + r12*r22 + r13*r23 = 0
        // ==> (r11*r21 + r12*r22)^2 = r13^2 * r23^2 = (1 - (r11^2 + r12^2)) * r23^2 = ((r21^2 + r22^2) + r23^2- (r11^2 + r12^2)) * r23^2
        // ==> AA = (CC + r23^2 - BB) * r23^2
        // ==> r23^4 + (CC - BB) * r23^2 - AA = 0
        // 求解上述一元二次方程
        std::vector<double> r23, r13, r23_squre;
        double tmp = std::sqrt(std::pow(CC-BB, 2) + 4*AA);
        if(BB - CC - tmp >= 0)
        {
            r23_squre.push_back((BB-CC-tmp)/2);
        }
        if(BB - CC + tmp >= 0)
        {
            r23_squre.push_back((BB-CC+tmp)/2);
        }
        if(r23_squre.size() == 0)
        {
            has_chessboard_[i] = false;
            continue;
        }
        total_num++;
        int sign[2] = {-1, 1};
        for(int j = 0; j < r23_squre.size(); j++)
        {
            for(int k = 0; k < 2; k++)
            {
                double r23_ = sign[k]*std::sqrt(r23_squre[j]);
                r23.push_back(r23_);
                if(r23_squre[j] < 1e-5)
                {
                    r13.push_back(std::sqrt(CC+r23_squre[j]-BB));
                    r13.push_back(-std::sqrt(CC+r23_squre[j]-BB));
                    r23.push_back(r23_);
                }
                else
                {
                    r13.push_back(-(r11*r21 + r12*r22) / r23_);
                }
            }
            
        }
        
        std::vector<cv::Mat> Rt;
        for(int j = 0; j < r13.size(); j++)
        {
            double len = std::sqrt(r11*r11 + r12*r12 + r13[j]*r13[j]);
            cv::Mat tmp = (cv::Mat_<double>(3,3) << r11, r21, t1 , r12, r22, t2, r13[j], r23[j], 0);
            tmp /= len;
            Rt.push_back(tmp);
            Rt.push_back(-tmp);
        }
        // 对于每一个r13 r23，计算gamma和t3
        for(int j = 0; j < Rt.size(); j++)
        {
            double r11_ = Rt[j].at<double>(0, 0);
            double r12_ = Rt[j].at<double>(1, 0);
            double r13_ = Rt[j].at<double>(2, 0);
            double r21_ = Rt[j].at<double>(0, 1);
            double r22_ = Rt[j].at<double>(1, 1);
            double r23_ = Rt[j].at<double>(2, 1);
            double t1_ = Rt[j].at<double>(0, 2);
            double t2_ = Rt[j].at<double>(1, 2);
            A = cv::Mat(cv::Size(3, point_num*2), CV_64F);
            x = cv::Mat(cv::Size(1, 3), CV_64F);
            cv::Mat b(cv::Size(1, point_num*2), CV_64F);
            for(int k = 0; k < point_num; k++)
            {
                double u = pixels_center[i][k].x;
                double v = pixels_center[i][k].y;
                double rpho = u*u+v*v;
                double A_ = r12_*worlds[k].x + r22_*worlds[k].y + t2_;
                double B_ = v * (r13_*worlds[k].x + r23_*worlds[k].y);
                double C_ = r11_*worlds[k].x + r21_*worlds[k].y + t1_;
                double D_ = u * (r13_*worlds[k].x + r23_*worlds[k].y);
                A.at<double>(2*k, 0) = A_/2;
                A.at<double>(2*k, 1) = -A_*rpho/2;
                A.at<double>(2*k, 2) = -v;
                A.at<double>(2*k+1, 0) = C_/2;
                A.at<double>(2*k+1, 1) = -C_*rpho/2;
                A.at<double>(2*k+1, 2) = -u;
                b.at<double>(2*k, 0) = B_;
                b.at<double>(2*k+1, 0) = D_;
            }
            cv::solve(A.t()*A, A.t()*b, x, cv::DECOMP_LU);
            Rt[j].at<double>(2,2) = x.at<double>(2,0);
            if(x.at<double>(0,0)/x.at<double>(1,0) >= 0)
                gammas_[j] = std::sqrt(x.at<double>(0,0)/x.at<double>(1,0));
            else
                gammas_[j] = 0;
        }
        double min_error = 1e10;
        int min_id = 0;
        for(int j = 0; j < Rt.size(); j++)
        {
            double error = ReprojectError(pixels[i], worlds, Rt[j], cx_, cy_, gammas_[j]/2, gammas_[j]/2, xi_, alpha_);
            if(error < min_error)
            {
                min_error = error;
                min_id = j;
            }
        }
        fx_ += gammas_[min_id] / 2;
        fy_ += gammas_[min_id] / 2;
        Rt_[i] = Rt[min_id];
    }
    fx_ /= total_num;
    fy_ /= total_num;
    std::cout << "[initialize] fx:" << fx_ << ", fy:" << fy_ << ", cx:" << cx_ << ", cy:" << cy_ << ", alpha:" << alpha_ << ", xi:" << xi_ << std::endl;
}

double DoubleSphereCamera::ReprojectError(const std::vector<cv::Point2d>& pixels, const std::vector<cv::Point3d>& worlds, cv::Mat Rt,
                          double cx, double cy, double fx, double fy, double xi, double alpha)
{
    double error = 0;
    for(int i = 0; i < worlds.size(); i++)
    {
        cv::Mat P = (cv::Mat_<double>(3,1) << worlds[i].x, worlds[i].y, 1);
        P = Rt * P;
        double d1 = std::sqrt(P.at<double>(0,0)*P.at<double>(0,0)+P.at<double>(1,0)*P.at<double>(1,0)+P.at<double>(2,0)*P.at<double>(2,0));
        double d2 = std::sqrt(P.at<double>(0,0)*P.at<double>(0,0)+P.at<double>(1,0)*P.at<double>(1,0)+std::pow(P.at<double>(2,0)+xi*d1,2));
        double pixel_x = fx * P.at<double>(0,0)/(alpha*d2+(1-alpha)*(xi*d1+P.at<double>(2,0))) + cx;
        double pixel_y = fy * P.at<double>(1,0)/(alpha*d2+(1-alpha)*(xi*d1+P.at<double>(2,0))) + cy;
        error += std::sqrt((pixels[i].x-pixel_x)*(pixels[i].x-pixel_x) + (pixels[i].y-pixel_y)*(pixels[i].y-pixel_y));
    }
    return error / worlds.size();
}

void DoubleSphereCamera::Reproject(const std::vector<cv::Point3d>& worlds, cv::Mat Rt, std::vector<cv::Point2d>& pixels)
{
    pixels.resize(worlds.size());
    for(int i = 0; i < worlds.size(); i++)
    {
        cv::Mat P = (cv::Mat_<double>(3,1) << worlds[i].x, worlds[i].y, 1);
        P = Rt * P;
        double d1 = std::sqrt(P.at<double>(0,0)*P.at<double>(0,0)+P.at<double>(1,0)*P.at<double>(1,0)+P.at<double>(2,0)*P.at<double>(2,0));
        double d2 = std::sqrt(P.at<double>(0,0)*P.at<double>(0,0)+P.at<double>(1,0)*P.at<double>(1,0)+std::pow(P.at<double>(2,0)+xi_*d1,2));
        double pixel_x = fx_ * P.at<double>(0,0)/(alpha_*d2+(1-alpha_)*(xi_*d1+P.at<double>(2,0))) + cx_;
        double pixel_y = fy_ * P.at<double>(1,0)/(alpha_*d2+(1-alpha_)*(xi_*d1+P.at<double>(2,0))) + cy_;
        pixels[i] = cv::Point2d(pixel_x, pixel_y);
    }
}

void DoubleSphereCamera::refinement(const std::vector<std::vector<cv::Point2d>>& pixels, const std::vector<cv::Point3d>& worlds)
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
                    6,6>(cost_function),
                NULL,
                intrinsic_.data(),
                rt_[i].data());
        }
    }
    // Configure the solver.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout =false;
    options.max_num_iterations = 50;

    // Solve!
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
}

void DoubleSphereCamera::undistort(double fx, double fy, double cx, double cy, cv::Size img_size, cv::Mat& mapx, cv::Mat& mapy)
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
            double pixel_x = fx_ * x/(alpha_*d2+(1-alpha_)*(xi_*d1+z)) + cx_;
            double pixel_y = fy_ * y/(alpha_*d2+(1-alpha_)*(xi_*d1+z)) + cy_;
            mapx.at<float>(i, j) = pixel_x;
            mapy.at<float>(i, j) = pixel_y;
        }
    }
}
