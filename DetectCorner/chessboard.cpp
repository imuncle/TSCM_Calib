#include "chessboard.h"

std::vector<cv::Mat> chessboardsFromCorners(struct Corner_t corners)
{
    std::vector<cv::Mat> chessboards;
    cv::Mat chessboard = cv::Mat::zeros(cv::Size(3, 3), CV_16U);
    for (unsigned int i = 0; i < corners.p.size(); i++)
    {
        chessboard = initChessboard(corners, i);
        if (chessboard.at<uint16_t>(0, 0) == 0 && chessboard.at<uint16_t>(0, 1) == 0)
            continue;
        if (chessboardEnergy(chessboard, corners) > 0)
            continue;
        while (1)
        {
            double enegy = chessboardEnergy(chessboard, corners);
            std::vector<cv::Mat> proposal;
            std::vector<double> p_enegy;
            for (int j = 0; j < 4; j++)
            {
                cv::Mat chess = growChessboard(chessboard, corners, j);
                proposal.push_back(chess);
                p_enegy.push_back(chessboardEnergy(chess, corners));
            }
            int min_idx = min_element(p_enegy.begin(), p_enegy.end()) - p_enegy.begin();
            if (p_enegy[min_idx] < enegy)
                chessboard = proposal[min_idx];
            else
                break;
        }
        if (chessboardEnergy(chessboard, corners) < -10)
        {
            std::vector<cv::Point2d> overlap;
            if (chessboards.size() > 0)
            {
                overlap.resize(chessboards.size());
                for (unsigned int j = 0; j < chessboards.size(); j++)
                {
                    auto ptr = chessboards[j].begin<uint16_t>();
                    for (int k = 0; k < chessboards[j].rows*chessboards[j].cols; k++)
                    {
                        auto ret = std::find(chessboard.begin<uint16_t>(), chessboard.end<uint16_t>(), (*ptr));
                        ptr++;
                        if (ret != chessboard.end<uint16_t>())
                        {
                            overlap[j].x = 1;
                            overlap[j].y = chessboardEnergy(chessboards[j], corners);
                            break;
                        }
                    }
                }
                bool overlaped = false;
                bool lower_enegy = false;
                for (unsigned int j = 0; j < overlap.size(); j++)
                {
                    if (overlap[j].x == 1)
                    {
                        overlaped = true;
                        if (overlap[j].y > chessboardEnergy(chessboard, corners))
                        {
                            chessboards[j] = cv::Mat::zeros(cv::Size(2, 2), CV_64F);
                            lower_enegy = true;
                        }
                    }
                }
                if (overlaped == false || lower_enegy == true)
                    chessboards.push_back(chessboard);
                for (auto iter = chessboards.begin(); iter != chessboards.end(); iter++)
                {
                    if (iter->at<double>(0, 0) == 0 && iter->at<double>(0, 1) == 0)
                    {
                        chessboards.erase(iter);
                        iter--;
                    }
                }
            }
            else
            {
                chessboards.push_back(chessboard);
            }
        }
    }
    for (unsigned int i = 0; i < chessboards.size(); i++)
    {
        cv::Mat chess = chessboards[i];
        cv::Mat new_chess;
        // 保证列数不小于行数
        if (chess.cols < chess.rows)
        {
            new_chess = cv::Mat::zeros(cv::Size(chess.rows, chess.cols), CV_16U);
            for (int j = 0; j < new_chess.rows; j++)
            {
                for (int k = 0; k < new_chess.cols; k++)
                {
                    new_chess.at<uint16_t>(j, k) = chess.at<uint16_t>(chess.rows - k - 1, j);
                }
            }
            chess = new_chess.clone();
        }
        chessboards[i] = chess;
    }
    return chessboards;
}

cv::Mat initChessboard(struct Corner_t corners, int idx)
{
    cv::Mat chessboard = cv::Mat::zeros(cv::Size(3, 3), CV_16U);
    if (corners.p.size() < 9)
        return chessboard;
    cv::Point2d v1 = corners.v1[idx];
    cv::Point2d v2 = corners.v2[idx];
    cv::Point2d minus_v1 = cv::Point2d(-corners.v1[idx].x, -corners.v1[idx].y);
    cv::Point2d minus_v2 = cv::Point2d(-corners.v2[idx].x, -corners.v2[idx].y);
    chessboard.at<uint16_t>(1, 1) = idx;
    double dist1[2] = { 0 };
    double dist2[6] = { 0 };
    int neighbor_idx = 0;
    // find left/right/top/bottom neighbors
    directionalNeighbor(idx, v1, chessboard, corners, &neighbor_idx, &dist1[0]);
    chessboard.at<uint16_t>(1, 2) = neighbor_idx;
    directionalNeighbor(idx, minus_v1, chessboard, corners, &neighbor_idx, &dist1[1]);
    chessboard.at<uint16_t>(1, 0) = neighbor_idx;
    directionalNeighbor(idx, v2, chessboard, corners, &neighbor_idx, &dist2[0]);
    chessboard.at<uint16_t>(2, 1) = neighbor_idx;
    directionalNeighbor(idx, minus_v2, chessboard, corners, &neighbor_idx, &dist2[1]);
    chessboard.at<uint16_t>(0, 1) = neighbor_idx;
    // find top-left/top-right/bottom-left/bottom-right neighbors
    directionalNeighbor(chessboard.at<uint16_t>(1, 0), minus_v2, chessboard, corners, &neighbor_idx, &dist2[2]);
    chessboard.at<uint16_t>(0, 0) = neighbor_idx;
    directionalNeighbor(chessboard.at<uint16_t>(1, 0), v2, chessboard, corners, &neighbor_idx, &dist2[3]);
    chessboard.at<uint16_t>(2, 0) = neighbor_idx;
    directionalNeighbor(chessboard.at<uint16_t>(1, 2), minus_v2, chessboard, corners, &neighbor_idx, &dist2[4]);
    chessboard.at<uint16_t>(0, 2) = neighbor_idx;
    directionalNeighbor(chessboard.at<uint16_t>(1, 2), v2, chessboard, corners, &neighbor_idx, &dist2[5]);
    chessboard.at<uint16_t>(2, 2) = neighbor_idx;
    double ave_1 = average(dist1, 2);
    double std_1 = stdd(dist1, 2, ave_1);
    if (std_1 / ave_1 > 0.3)
    {
        return cv::Mat::zeros(cv::Size(3, 3), CV_16U);
    }
    double ave_2 = average(dist2, 6);
    double std_2 = stdd(dist2, 6, ave_2);
    if (std_2 / ave_2 > 0.3)
    {
        return cv::Mat::zeros(cv::Size(3, 3), CV_16U);
    }
    return chessboard;
}

double average(double* a, int length)
{
    double sum = 0;
    for (int i = 0; i < length; i++)
    {
        sum += a[i];
    }
    return sum / length;
}

double stdd(double* a, int length, double mean)
{
    double sum = 0;
    for (int i = 0; i < length; i++)
    {
        sum += (a[i] - mean)*(a[i] - mean);
    }
    sum =  sum / (length-1);
    return sqrt(sum);
}

void directionalNeighbor(int idx, cv::Point2d v, cv::Mat chessboard, struct Corner_t corners, int* neighbor_idx, double* min_dist)
{

    std::vector<int> unused;
    std::vector<int> used;
    unused.resize(corners.p.size());
    for (unsigned int i = 0; i < unused.size(); i++)
        unused[i] = i;
    for (int i = 0; i < chessboard.rows; i++)
    {
        for (int j = 0; j < chessboard.cols; j++)
        {
            if (chessboard.at<uint16_t>(i, j) != 0)
            {
                used.push_back(chessboard.at<uint16_t>(i, j));
            }
        }
    }
    for (auto iterator = unused.begin(); iterator != unused.end(); iterator++)
    {
        for (unsigned int i = 0; i < used.size(); i++)
        {
            if (*iterator == used[i])
            {
                unused.erase(iterator);
                iterator--;
            }
        }
    }
    std::vector<double> dist;
    dist.resize(unused.size());
    for (unsigned int i = 0; i < unused.size(); i++)
    {
        cv::Point2d dir = corners.p[unused[i]] - corners.p[idx];
        dist[i] = dir.x*v.x + dir.y*v.y;
        cv::Point2d edge = dir - dist[i] * v;
        double dist_edge = sqrt(edge.x*edge.x+edge.y*edge.y);
        if (dist[i] < 0) dist[i] = 1e10;
        dist[i] = dist[i] + 5 * dist_edge;
    }
    int min_idx = min_element(dist.begin(), dist.end()) - dist.begin();
    *min_dist = dist[min_idx];
    *neighbor_idx = unused[min_idx];
}

double chessboardEnergy(cv::Mat chessboard, struct Corner_t corners)
{
    double E_structure = 0;
    for (int j = 0; j < chessboard.rows; j++)
    {
        for (int k = 0; k < chessboard.cols-2; k++)
        {
            std::vector<cv::Point2d> x;
            for (int i = k; i <= k + 2; i++)
            {
                x.push_back(corners.p[chessboard.at<uint16_t>(j, i)]);
            }
            cv::Point2d x_ = x[0] + x[2] - 2 * x[1];
            double x_norm1 = sqrt(x_.x*x_.x+x_.y*x_.y);
            x_ = x[0] - x[2];
            double x_norm2 = sqrt(x_.x*x_.x + x_.y*x_.y);
            E_structure = std::max(E_structure, x_norm1/x_norm2);
        }
    }
    for (int j = 0; j < chessboard.cols; j++)
    {
        for (int k = 0; k < chessboard.rows - 2; k++)
        {
            std::vector<cv::Point2d> x;
            for (int i = k; i <= k + 2; i++)
            {
                x.push_back(corners.p[chessboard.at<uint16_t>(i, j)]);
            }
            cv::Point x_ = x[0] + x[2] - 2 * x[1];
            double x_norm1 = sqrt(x_.x*x_.x + x_.y*x_.y);
            x_ = x[0] - x[2];
            double x_norm2 = sqrt(x_.x*x_.x + x_.y*x_.y);
            E_structure = std::max(E_structure, x_norm1 / x_norm2);
        }
    }
    return chessboard.rows*chessboard.cols*(E_structure - 1);
}

cv::Mat growChessboard(cv::Mat chessboard, struct Corner_t corners, int boarder_type)
{
    if (chessboard.at<uint16_t>(0, 0) == 0 && chessboard.at<uint16_t>(0, 1) == 0)
        return chessboard;
    std::vector<int> unused;
    std::vector<int> used;
    unused.resize(corners.p.size());
    for (unsigned int i = 0; i < unused.size(); i++)
        unused[i] = i;
    for (int i = 0; i < chessboard.rows; i++)
    {
        for (int j = 0; j < chessboard.cols; j++)
        {
            if (chessboard.at<uint16_t>(i, j) != 0)
            {
                used.push_back(chessboard.at<uint16_t>(i, j));
            }
        }
    }
    for (auto iterator = unused.begin(); iterator != unused.end(); iterator++)
    {
        for (unsigned int i = 0; i < used.size(); i++)
        {
            if (*iterator == used[i])
            {
                unused.erase(iterator);
                iterator--;
            }
        }
    }
    std::vector<cv::Point2d> cand;
    for (unsigned int i = 0; i < unused.size(); i++)
    {
        cand.push_back(corners.p[unused[i]]);
    }
    switch (boarder_type)
    {
    case 0:
    {
        std::vector<cv::Point2d> pred;
        for (int i = 0; i < chessboard.rows; i++)
        {
            pred.push_back(predictCorners(corners.p[chessboard.at<uint16_t>(i, chessboard.cols-3)],
                corners.p[chessboard.at<uint16_t>(i, chessboard.cols - 2)],
                corners.p[chessboard.at<uint16_t>(i, chessboard.cols - 1)]));
        }
        std::vector<int> idx = assignClosestCorners(cand, pred);
        if (idx.size() > 0)
        {
            cv::Mat new_chess = cv::Mat::zeros(cv::Size(1, chessboard.rows), CV_16U);
            for (unsigned int j = 0; j < idx.size(); j++)
            {
                new_chess.at<uint16_t>(j, 0) = unused[idx[j]];
            }
            cv::Mat new_chessboard = cv::Mat::zeros(cv::Size(chessboard.cols+1, chessboard.rows), CV_16U);
            cv::Mat new_roi = new_chessboard(cv::Rect(0, 0, chessboard.cols, chessboard.rows));
            chessboard.convertTo(new_roi, new_roi.type());
            new_roi = new_chessboard(cv::Rect(chessboard.cols, 0, 1, chessboard.rows));
            new_chess.convertTo(new_roi, new_roi.type());
            chessboard = new_chessboard;
        }
        break;
    }
    case 1:
    {
        std::vector<cv::Point2d> pred;
        for (int i = 0; i < chessboard.cols; i++)
        {
            pred.push_back(predictCorners(corners.p[chessboard.at<uint16_t>(chessboard.rows - 3, i)],
                corners.p[chessboard.at<uint16_t>(chessboard.rows - 2, i)],
                corners.p[chessboard.at<uint16_t>(chessboard.rows - 1, i)]));
        }
        std::vector<int> idx = assignClosestCorners(cand, pred);
        if (idx.size() > 0)
        {
            cv::Mat new_chess = cv::Mat::zeros(cv::Size(chessboard.cols, 1), CV_16U);
            for (unsigned int j = 0; j < idx.size(); j++)
            {
                new_chess.at<uint16_t>(0, j) = unused[idx[j]];
            }
            cv::Mat new_chessboard = cv::Mat::zeros(cv::Size(chessboard.cols, chessboard.rows + 1), CV_16U);
            cv::Mat new_roi = new_chessboard(cv::Rect(0, 0, chessboard.cols, chessboard.rows));
            chessboard.convertTo(new_roi, new_roi.type());
            new_roi = new_chessboard(cv::Rect(0, chessboard.rows, chessboard.cols, 1));
            new_chess.convertTo(new_roi, new_roi.type());
            chessboard = new_chessboard;
        }
        break;
    }
    case 2:
    {
        std::vector<cv::Point2d> pred;
        for (int i = 0; i < chessboard.rows; i++)
        {
            pred.push_back(predictCorners(corners.p[chessboard.at<uint16_t>(i, 2)],
                corners.p[chessboard.at<uint16_t>(i, 1)],
                corners.p[chessboard.at<uint16_t>(i, 0)]));
        }
        std::vector<int> idx = assignClosestCorners(cand, pred);
        if (idx.size() > 0)
        {
            cv::Mat new_chess = cv::Mat::zeros(cv::Size(1, chessboard.rows), CV_16U);
            for (unsigned int j = 0; j < idx.size(); j++)
            {
                new_chess.at<uint16_t>(j, 0) = unused[idx[j]];
            }
            cv::Mat new_chessboard = cv::Mat::zeros(cv::Size(chessboard.cols + 1, chessboard.rows), CV_16U);
            cv::Mat new_roi = new_chessboard(cv::Rect(1, 0, chessboard.cols, chessboard.rows));
            chessboard.convertTo(new_roi, new_roi.type());
            new_roi = new_chessboard(cv::Rect(0, 0, 1, chessboard.rows));
            new_chess.convertTo(new_roi, new_roi.type());
            chessboard = new_chessboard;
        }
        break;
    }
    case 3:
    {
        std::vector<cv::Point2d> pred;
        for (int i = 0; i < chessboard.cols; i++)
        {
            pred.push_back(predictCorners(corners.p[chessboard.at<uint16_t>(2, i)],
                corners.p[chessboard.at<uint16_t>(1, i)],
                corners.p[chessboard.at<uint16_t>(0, i)]));
        }
        std::vector<int> idx = assignClosestCorners(cand, pred);
        if (idx.size() > 0)
        {
            cv::Mat new_chess = cv::Mat::zeros(cv::Size(chessboard.cols, 1), CV_16U);
            for (unsigned int j = 0; j < idx.size(); j++)
            {
                new_chess.at<uint16_t>(0, j) = unused[idx[j]];
            }
            cv::Mat new_chessboard = cv::Mat::zeros(cv::Size(chessboard.cols, chessboard.rows + 1), CV_16U);
            cv::Mat new_roi = new_chessboard(cv::Rect(0, 1, chessboard.cols, chessboard.rows));
            chessboard.convertTo(new_roi, new_roi.type());
            new_roi = new_chessboard(cv::Rect(0, 0, chessboard.cols, 1));
            new_chess.convertTo(new_roi, new_roi.type());
            chessboard = new_chessboard;
        }
        break;
    }
    }
    return chessboard;
}

cv::Point2d predictCorners(cv::Point2d p1, cv::Point2d p2, cv::Point2d p3)
{
    cv::Point2d v1 = p2 - p1;
    cv::Point2d v2 = p3 - p2;
    double a1 = atan2(v1.y, v1.x);
    double a2 = atan2(v2.y, v2.x);
    double a3 = 2 * a2 - a1;
    double s1 = sqrt(v1.x*v1.x+v1.y*v1.y);
    double s2 = sqrt(v2.x*v2.x+v2.y*v2.y);
    double s3 = 2 * s2 - s1;
    cv::Point2d pred;
    pred.x = p3.x + 0.75*s3*cos(a3);
    pred.y = p3.y + 0.75*s3*sin(a3);
    return pred;
}

std::vector<int> assignClosestCorners(std::vector<cv::Point2d> cand, std::vector<cv::Point2d> pred)
{
    std::vector<int> idx;
    if (cand.size() < pred.size())
        return idx;
    idx.resize(pred.size());
    cv::Mat D = cv::Mat::zeros(cv::Size(pred.size(), cand.size()), CV_64F);
    for (unsigned int i = 0; i < pred.size(); i++)
    {
        for (unsigned int j = 0; j < cand.size(); j++)
        {
            cv::Point2d delta = cand[j] - pred[i];
            D.at<double>(j, i) = sqrt(delta.x*delta.x+delta.y*delta.y);
        }
    }
    for (unsigned int i = 0; i < pred.size(); i++)
    {
        double minVal, maxVal;
        int    minIdx[2] = {}, maxIdx[2] = {};	// minnimum Index, maximum Index
        cv::minMaxIdx(D, &minVal, &maxVal, minIdx, maxIdx);
        idx[minIdx[1]] = minIdx[0];
        for (int j = 0; j < D.cols; j++)
        {
            D.at<double>(minIdx[0], j) = 1e10;
        }
        for (int j = 0; j < D.rows; j++)
        {
            D.at<double>(j, minIdx[1]) = 1e10;
        }
    }
    return idx;
}

void plotChessboards(cv::Mat img, std::vector<cv::Mat> chessboards, struct Corner_t corners)
{
    for (unsigned int i = 0; i < chessboards.size(); i++)
    {
        cv::Mat chessboard = chessboards[i];
        cv::Point2d o = corners.p[chessboard.at<uint16_t>(0, 0)];
        cv::Point2d o1 = corners.p[chessboard.at<uint16_t>(0, 1)];
        cv::Point2d o2 = corners.p[chessboard.at<uint16_t>(1, 0)];
        if (img.channels() == 3)
        {
            for (int j = 0; j < chessboard.rows; j++)
            {
                for (int k = 1; k < chessboard.cols; k++)
                {
                    cv::line(img, corners.p[chessboard.at<uint16_t>(j, k - 1)], corners.p[chessboard.at<uint16_t>(j, k)], cv::Scalar(0, 0, 255), 2);
                }
            }
            for (int j = 0; j < chessboard.cols; j++)
            {
                for (int k = 1; k < chessboard.rows; k++)
                {
                    cv::line(img, corners.p[chessboard.at<uint16_t>(k - 1, j)], corners.p[chessboard.at<uint16_t>(k, j)], cv::Scalar(0, 0, 255), 2);
                }
            }
            cv::line(img, o, o1, cv::Scalar(255, 0, 0), 2);
            cv::line(img, o, o2, cv::Scalar(0, 255, 0), 2);
        }
        else
        {
            cv::line(img, o, o1, cv::Scalar(100), 2);
            cv::line(img, o, o2, cv::Scalar(100), 2);
        }
    }
}
