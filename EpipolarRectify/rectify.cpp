#include <opencv2/opencv.hpp>

class TScamera
{
    public:
    TScamera(double fx, double fy, double cx, double cy, double xi, double lambda, double alpha, double b, double c)
        :fx_(fx),fy_(fy),cx_(cx),cy_(cy),xi_(xi),lamda_(lambda),alpha_(alpha),b_(b),c_(c){w2_ = 0.42399;}
    cv::Point3d get_unit_sphere_coordinate(cv::Point2d pixel)
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
        return cv::Point3d(mu*yita*gamma*mx, mu*yita*gamma*my, mu*(mz-lamda_) - xi_);
    }
    cv::Point2d project(cv::Point3d p)
    {
        double X = p.x;
        double Y = p.y;
        double Z = p.z;
        double d1 = std::sqrt(X*X+Y*Y+Z*Z);
        if(Z <= -w2_*d1) return cv::Point2d(-1, -1);
        double d2 = std::sqrt(X*X+Y*Y+std::pow(Z+xi_*d1,2));
        double d3 = std::sqrt(X*X+Y*Y+std::pow(Z+xi_*d1+lamda_*d2,2));
        double ksai = Z+xi_*d1+lamda_*d2+alpha_/(1-alpha_)*d3;
        double pixel_x = fx_ * X/ksai + b_ * Y/ksai + cx_;
        double pixel_y = c_ * X/ksai + fy_ * Y/ksai + cy_;
        return cv::Point2d(pixel_x, pixel_y);
    }
    private:
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    double xi_;
    double lamda_;
    double alpha_;
    double b_;
    double c_;
    double w2_;
};

// 双目极线矫正
class Remap
{
    public:
    Remap(cv::Mat& front, cv::Mat Tfront, cv::Mat& right, cv::Mat& Tright, cv::Mat& rear, cv::Mat& Trear, cv::Mat& left, cv::Mat& Tleft){
        front_cam = std::make_shared<TScamera>(front.at<double>(0), front.at<double>(1), front.at<double>(2),
                            front.at<double>(3), front.at<double>(4), front.at<double>(5),
                            front.at<double>(6), front.at<double>(7), front.at<double>(8));
        right_cam = std::make_shared<TScamera>(right.at<double>(0), right.at<double>(1), right.at<double>(2),
                            right.at<double>(3), right.at<double>(4), right.at<double>(5),
                            right.at<double>(6), right.at<double>(7), right.at<double>(8));
        rear_cam = std::make_shared<TScamera>(rear.at<double>(0), rear.at<double>(1), rear.at<double>(2),
                            rear.at<double>(3), rear.at<double>(4), rear.at<double>(5),
                            rear.at<double>(6), rear.at<double>(7), rear.at<double>(8));
        left_cam = std::make_shared<TScamera>(left.at<double>(0), left.at<double>(1), left.at<double>(2),
                            left.at<double>(3), left.at<double>(4), left.at<double>(5),
                            left.at<double>(6), left.at<double>(7), left.at<double>(8));
        R_front = Tfront.colRange(0,3);
        t_front = std::vector<double>{Tfront.at<double>(0,3), Tfront.at<double>(1,3), Tfront.at<double>(2,3)};
        R_right = Tright.colRange(0,3);
        t_right = std::vector<double>{Tright.at<double>(0,3), Tright.at<double>(1,3), Tright.at<double>(2,3)};
        R_rear = Trear.colRange(0,3);
        t_rear = std::vector<double>{Trear.at<double>(0,3), Trear.at<double>(1,3), Trear.at<double>(2,3)};
        R_left = Tleft.colRange(0,3);
        t_left = std::vector<double>{Tleft.at<double>(0,3), Tleft.at<double>(1,3), Tleft.at<double>(2,3)};

        image_size = cv::Size(400, 400);

        right_mapx = cv::Mat(cv::Size(400,1600), CV_32FC1);
        right_mapy = cv::Mat(cv::Size(400,1600), CV_32FC1);
        left_mapx = cv::Mat(cv::Size(400,1600), CV_32FC1);
        left_mapy = cv::Mat(cv::Size(400,1600), CV_32FC1);

        cx = cy = 200;
        fx = fy = 200;
    }
    
    void init_remap()
    {
        cv::Mat R_front_right = calc_R(t_front, t_right);
        cv::Mat R_right_rear = calc_R(t_right, t_rear);
        cv::Mat R_rear_left = calc_R(t_rear, t_left);
        cv::Mat R_left_front = calc_R(t_left, t_front);
        cv::Mat R;

        // front and right
        R = R_front.t() * R_front_right;
        for(int i = 0; i < image_size.height; i++)
        {
            for(int j = 0; j < image_size.width; j++)
            {
                cv::Mat p = (cv::Mat_<double>(3,1) << (j-cx)/fx, (i-cy)/fy, 1);
                p = R * p;
                cv::Point2d pixel = front_cam->project(cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2)));
                left_mapx.at<float>(i,j) = pixel.x;
                left_mapy.at<float>(i,j) = pixel.y;
            }
        }
        R = R_right.t() * R_front_right;
        for(int i = 0; i < image_size.height; i++)
        {
            for(int j = 0; j < image_size.width; j++)
            {
                cv::Mat p = (cv::Mat_<double>(3,1) << (j-cx)/fx, (i-cy)/fy, 1);
                p = R * p;
                cv::Point2d pixel = right_cam->project(cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2)));
                right_mapx.at<float>(i,j) = pixel.x+1280;
                right_mapy.at<float>(i,j) = pixel.y;
            }
        }

        // right and rear
        R = R_right.t() * R_right_rear;
        for(int i = 0; i < image_size.height; i++)
        {
            for(int j = 0; j < image_size.width; j++)
            {
                cv::Mat p = (cv::Mat_<double>(3,1) << (j-cx)/fx, (i-cy)/fy, 1);
                p = R * p;
                cv::Point2d pixel = right_cam->project(cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2)));
                left_mapx.at<float>(i+image_size.height,j) = pixel.x+1280;
                left_mapy.at<float>(i+image_size.height,j) = pixel.y;
            }
        }
        R = R_rear.t() * R_right_rear;
        for(int i = 0; i < image_size.height; i++)
        {
            for(int j = 0; j < image_size.width; j++)
            {
                cv::Mat p = (cv::Mat_<double>(3,1) << (j-cx)/fx, (i-cy)/fy, 1);
                p = R * p;
                cv::Point2d pixel = rear_cam->project(cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2)));
                right_mapx.at<float>(i+image_size.height,j) = pixel.x;
                right_mapy.at<float>(i+image_size.height,j) = pixel.y+1080;
            }
        }

        // rear and left
        R = R_rear.t() * R_rear_left;
        for(int i = 0; i < image_size.height; i++)
        {
            for(int j = 0; j < image_size.width; j++)
            {
                cv::Mat p = (cv::Mat_<double>(3,1) << (j-cx)/fx, (i-cy)/fy, 1);
                p = R * p;
                cv::Point2d pixel = rear_cam->project(cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2)));
                left_mapx.at<float>(i+image_size.height*2,j) = pixel.x;
                left_mapy.at<float>(i+image_size.height*2,j) = pixel.y+1080;
                
            }
        }
        R = R_left.t() * R_rear_left;
        for(int i = 0; i < image_size.height; i++)
        {
            for(int j = 0; j < image_size.width; j++)
            {
                cv::Mat p = (cv::Mat_<double>(3,1) << (j-cx)/fx, (i-cy)/fy, 1);
                p = R * p;
                cv::Point2d pixel = left_cam->project(cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2)));
                right_mapx.at<float>(i+image_size.height*2,j) = pixel.x+1280;
                right_mapy.at<float>(i+image_size.height*2,j) = pixel.y+1080;
            }
        }

        // left and front
        R = R_left.t() * R_left_front;
        for(int i = 0; i < image_size.height; i++)
        {
            for(int j = 0; j < image_size.width; j++)
            {
                cv::Mat p = (cv::Mat_<double>(3,1) << (j-cx)/fx, (i-cy)/fy, 1);
                p = R * p;
                cv::Point2d pixel = left_cam->project(cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2)));
                left_mapx.at<float>(i+image_size.height*3,j) = pixel.x+1280;
                left_mapy.at<float>(i+image_size.height*3,j) = pixel.y+1080;
                
            }
        }
        R = R_front.t() * R_left_front;
        for(int i = 0; i < image_size.height; i++)
        {
            for(int j = 0; j < image_size.width; j++)
            {
                cv::Mat p = (cv::Mat_<double>(3,1) << (j-cx)/fx, (i-cy)/fy, 1);
                p = R * p;
                cv::Point2d pixel = front_cam->project(cv::Point3d(p.at<double>(0), p.at<double>(1), p.at<double>(2)));
                right_mapx.at<float>(i+image_size.height*3,j) = pixel.x;
                right_mapy.at<float>(i+image_size.height*3,j) = pixel.y;
            }
        }
    }
    void remap(cv::Mat& src, cv::Mat& dst, int code)
    {
        switch (code)
        {
        case 0:
            cv::remap(src, dst, left_mapx, left_mapy, cv::INTER_LINEAR);break;
        case 1:
            cv::remap(src, dst, right_mapx, right_mapy, cv::INTER_LINEAR);break;
        
        default:
            break;
        }
    }

    private:
    void normalize(double* vector)
    {
        double norm = std::sqrt(vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]);
        if(norm == 0) return;
        vector[0] /= norm;
        vector[1] /= norm;
        vector[2] /= norm;
    }

    void product(double* a, double* b, double* c)
    {
        // 0 -a[2] a[1]   b[0] = c[0]
        // a[2] 0 -a[0] * b[1] = c[1]
        // -a[1] a[0] 0   b[2] = c[2]
        c[0] = -a[2]*b[1] + a[1]*b[2];
        c[1] = a[2]*b[0] - a[0]*b[2];
        c[2] = -a[1]*b[0] + a[0]*b[1];
    }

    cv::Mat calc_R(std::vector<double>& t1, std::vector<double>& t2)
    {
        double x_axis[3] = {t2[0]-t1[0], t2[1]-t1[1], t2[2]-t1[2]};
        normalize(x_axis);
        // z轴始终在x-z平面
        double z_axis[3] = {-x_axis[2], 0, x_axis[0]};
        normalize(z_axis);
        double y_axis[3];
        product(z_axis, x_axis, y_axis);
        normalize(y_axis);
        cv::Mat R = (cv::Mat_<double>(3,3) << x_axis[0], y_axis[0], z_axis[0],
                                            x_axis[1], y_axis[1], z_axis[1],
                                            x_axis[2], y_axis[2], z_axis[2]);
        return R;
    }
    
    cv::Mat R_front, R_right, R_rear, R_left;
    std::vector<double> t_front, t_right, t_rear, t_left;
    cv::Size image_size;
    std::shared_ptr<TScamera> front_cam, right_cam, rear_cam, left_cam;
    cv::Mat right_mapx, right_mapy;
    cv::Mat left_mapx, left_mapy;
    double fx, fy, cx, cy;
};

int main(int argc, char** argv)
{
    cv::Mat front, Tfront, right, Tright, rear, Trear, left, Tleft;
    cv::FileStorage yaml_read("../calib.yaml", cv::FileStorage::READ);
    yaml_read["cam0"] >> front;
    yaml_read["Twc0"] >> Tfront;
    yaml_read["cam1"] >> right;
    yaml_read["Twc1"] >> Tright;
    yaml_read["cam2"] >> rear;
    yaml_read["Twc2"] >> Trear;
    yaml_read["cam3"] >> left;
    yaml_read["Twc3"] >> Tleft;
    Remap remap(front, Tfront, right, Tright, rear, Trear, left, Tleft);
    remap.init_remap();

    cv::Mat img = cv::imread("../test_img.jpg");
    cv::Mat left_img, right_img;
    remap.remap(img, left_img, 0);
    remap.remap(img, right_img, 1);
    cv::imshow("left", left_img);
    cv::imshow("right", right_img);
    cv::waitKey(0);
    return 0;
}