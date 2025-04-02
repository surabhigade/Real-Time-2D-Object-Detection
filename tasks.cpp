#include "tasks.h"
#include <iostream>
#include <vector> 
#include <cmath>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "kmeans.h"
// this function is to blur a single channel grayscale image 
int blur5x5(cv::Mat &src, cv::Mat &dst) {
    int arr[5] = {1, 2, 4, 2, 1};     
    cv::Mat src2 = src.clone();
    for(int y = 2; y < src.rows-2; y++) {         
        for(int x = 2; x < src.cols-2; x++) {             
            int count = 0, clr = 0;             
            int x_value;             
            for(int i = x-2; i<= x+2; i++) {                 
                x_value = arr[i-x+2];                 
                uchar pixel = src.at<uchar>(y, i);
                clr += pixel*x_value;                 
                count += x_value;
            }
            src2.at<uchar>(y, x) = clr/count;
        }
    }
    for(int y = 2; y < src.rows-2; y++) {         
        for(int x = 2; x < src.cols-2; x++) {             
            int count = 0, clr = 0;             
            int y_value;             
            for(int j = y-2; j<= y+2; j++) {                 
                y_value = arr[j-y+2];                 
                uchar pixel = src2.at<uchar>(j, x);
                clr += pixel*y_value;
                count += y_value;
            }
        dst.at<uchar>(y, x) = clr/count;
        }
    }
    return(0);
}

// this function is to blur the colored image
int blur5x5_clr(cv::Mat &src, cv::Mat &dst) {
    int arr[5] = {1, 2, 4, 2, 1};
    cv::Mat src2 = src.clone();
    for(int y = 2; y < src.rows-2; y++) {
        for(int x = 2; x < src.cols-2; x++) {
            int count = 0, b = 0, g = 0, r = 0;
            int x_value;
            for(int i = x-2; i<= x+2; i++) {
                x_value = arr[i-x+2];
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, i);
                b += pixel[0]*x_value;
                g += pixel[1]*x_value;
                r += pixel[2]*x_value;
                count += x_value;
            }
            src2.at<cv::Vec3b>(y, x)[0] = b/count;
            src2.at<cv::Vec3b>(y, x)[1] = g/count;
            src2.at<cv::Vec3b>(y, x)[2] = r/count;
        }
    }
    for(int y = 2; y < src.rows-2; y++) {
        for(int x = 2; x < src.cols-2; x++) {
            int count = 0, b = 0, g = 0, r = 0;
            int y_value;
            for(int j = y-2; j<= y+2; j++) {
                y_value = arr[j-y+2];
                cv::Vec3b pixel = src2.at<cv::Vec3b>(j, x);
                b += pixel[0]*y_value;
                g += pixel[1]*y_value;
                r += pixel[2]*y_value;
                count += y_value;
            }
            dst.at<cv::Vec3b>(y, x)[0] = b/count;
            dst.at<cv::Vec3b>(y, x)[1] = g/count;
            dst.at<cv::Vec3b>(y, x)[2] = r/count;
        }
    }
    return(0);
}

// i think its is self-explanatory - task_1
void threshold(cv::Mat &src, cv::Mat &dst, int thresh) {
    cv::Mat blur = src.clone();
    blur5x5(src, blur);
    dst = blur.clone();
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            uchar pixel =  blur.at<uchar>(i, j);
            if(pixel < thresh) {
                dst.at<uchar>(i, j) = 255;
            }
            else {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
    return;
}
// part of prof's code
cv::Vec3b closestColor( cv::Vec3b &pix, std::vector<cv::Vec3b> &colors ) {
    int mincolor = 0;
    int mindist = SSD( pix, colors[0] );
    for(int i=1;i<colors.size();i++) {
        int sse = SSD( pix, colors[i] );
        if( sse < mindist ) {
            mindist = sse;
            mincolor = i;
        }
    }
    return( colors[mincolor] );
}
// dilate and erosion function built from scratch
// if connect8 is true then 8 connectivity else 4 connectivity
// if inverse is true then it'll works like dilate else works like erosion
void dilate_built(cv::Mat &src, cv::Mat &dst, bool connect8 = true, bool inverse = false, int iteration = 1) {
    dst = src.clone();
    int value = 255;
    if(inverse == true) {
        value = 0;
    }
    for(int i{0}; i < iteration; i++) {
        for (int y = 0; y < src.rows; ++y) {
            for (int x = 0; x < src.cols; ++x) {
                if(src.at<uchar>(y, x) == value) { // in both 4 and 8 connectivity these 4 operations are same
                    if (y > 0) {
                        dst.at<uchar>(y-1, x) = value;
                    }
                    if (y < src.rows - 1) {
                        dst.at<uchar>(y+1, x) = value;
                    }
                    if (x > 0) {
                        dst.at<uchar>(y, x-1) = value;
                    }
                    if (x < src.cols - 1) {
                        dst.at<uchar>(y, x+1) = value;
                    }
                    if (connect8 == true) { // this 4 operations are only for 8 connectivity
                        if (y > 0 && x > 0) {
                            dst.at<uchar>(y-1, x-1) = value;
                        }
                        if (y < src.rows -1 && x > 0) {
                            dst.at<uchar>(y+1, x-1) = value;
                        }
                        if (x < src.cols -1 && y > 0) {
                            dst.at<uchar>(y-1, x+1) = value;
                        }
                        if (x < src.cols -1 and y < src.rows -1) {
                            dst.at<uchar>(y+1, x+1) = value;
                        }
                    }
                }
            }
        }
        src = dst.clone();
    }
    return;
}
// part of prof's code with little bit modification - task_1
int threshold_kmeans(cv::Mat &src, cv::Mat &dst) {
    blur5x5_clr(src, src);
    int ncolors = 2;
    int B = 4;
    std::vector<cv::Vec3b> data;
    for(int i=0;i<src.rows - B;i += B) {
        for(int j=0;j<src.cols - B;j += B) {
            int jx = rand() % B;
            int jy = rand() % B;
            data.push_back( src.at<cv::Vec3b>(i+jy, j+jx) );
        }
    }
    std::vector<cv::Vec3b> means;
    int *labels = new int[data.size()];
    if(kmeans( data, means, labels, ncolors ) ) {
        printf("Error using kmeans\n");
        return(-1);
    }
    for(int i=0;i<src.rows;i++) {
        for(int j=0;j<src.cols;j++) {
            dst.at<cv::Vec3b>(i,j) = closestColor( src.at<cv::Vec3b>(i,j), means );
        }
    }
    delete[] labels;
    return(0);
}
// this function is to perform image segmentation and for computing features for each major region
// if task_3 = true then task_3 will be executed else task_4/5 wil be executed according to wht name has
// if name is empty then task_4 else task_5 will be executed
void image_segment(cv::Mat &src, cv::Mat &dst, std::string name, int thresh, bool task_3 = true) {
    cv::Mat labels, centroid, stats;
    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    int numlabels = cv::connectedComponentsWithStats(src, labels, stats, centroid);
    std::vector<cv::Vec3b> colors = {cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255),
                                     cv::Vec3b(255, 255, 0), cv::Vec3b(0, 255, 255), cv::Vec3b(255, 0, 255)};
    std::vector<int> accepted_label(numlabels, 0);
    int maxval = 0;
    for(int i{1}; i < numlabels; i++) { // if any object touches the corner/ if the area is below a specific value it'll remove the object
        if(stats.at<int>(i, cv::CC_STAT_LEFT) <= 0 or stats.at<int>(i, cv::CC_STAT_TOP) <= 0) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH) >= src.cols) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT) >= src.rows) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_AREA) < thresh) {
            continue;
        }
        maxval += 1;
        accepted_label[i] = maxval;
    }
    for(int j{0}; j < src.rows; j++) {
        for(int k{0}; k < src.cols; k++) {
            int index = labels.at<int>(j, k);
            if(accepted_label[index] != 0) {
                dst.at<cv::Vec3b>(j, k) = colors[accepted_label[index]-1];
            }
        }
    }
    if(task_3 == true) { // if we're performing task_3 then we'll break here
        return;
    }
    float area;
    float ht_wt_ratio;
    for(int i{1}; i < numlabels; i++) {
        if(accepted_label[i] >= 1) {
            cv::Mat temp = (labels == i);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(temp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<cv::Point> largestContour = contours[0];
            // using the largest contour we're getting the features for oriented bounding box - center-x, center-y, width, height, angle of rotation
            cv::RotatedRect rotated_rect = cv::minAreaRect(largestContour);
            cv::Point2f rect_vertices[4];
            rotated_rect.points(rect_vertices);
            ht_wt_ratio = (float)rotated_rect.size.width/(float)rotated_rect.size.height;
            cv::Point2f edge1 = rect_vertices[1] - rect_vertices[0];
            cv::Point2f edge2 = rect_vertices[2] - rect_vertices[1];
            area = std::abs(edge1.x * edge2.y - edge1.y * edge2.x);
            for(int j{0}; j < 4; j++) {
                cv::line(dst, rect_vertices[j], rect_vertices[(j+1)%4], cv::Scalar(255, 255, 255), 4);
            }
            cv::Mat roi_image = temp.clone();
            cv::Moments moments = cv::moments(roi_image);
            double m00 = moments.m00;
            double m10 = moments.m10;
            double m01 = moments.m01;
            double m11 = moments.mu11;
            double m02 = moments.mu02;
            double m20 = moments.mu20;
            // here m means raw_moment, mu - central_moment
            cv::Mat cov_mat = (cv::Mat_<double>(2, 2) << m20 / m00, m11 / m00, m11 / m00, m02 / m00);
            // calculates eigen-vectors, values so as to get the axis of least central moment
            cv::Mat eigenvalues, eigenvectors;
            cv::eigen(cov_mat, eigenvalues, eigenvectors);
            int centroidX = (float)m10/(float)m00;
            int centroidY = (float)m01/(float)m00;
            cv::arrowedLine(dst, cv::Point(centroidX, centroidY),
            cv::Point(centroidX + 72*eigenvectors.at<double>(0, 0), centroidY + 72*eigenvectors.at<double>(0, 1)), cv::Scalar(0, 255, 0), 2);
        }
    }
    if(name == "") { // if name is empty then task_4 will break here
        return;
    }
    // now the task_5 will be executed, where we'll be save the unknown object in the DB (which is txt file in our case)
    std::string data = name + ";" + std::to_string((float)stats.at<int>(1, cv::CC_STAT_AREA)/area) + ";" + std::to_string(ht_wt_ratio) + "\n";
    std::string PATH = "/home/arun/Documents/data.txt";
    std::ofstream file;
    file.open(PATH, std::ios::app);
    if(file.is_open()) {
        file << data;
        file.close();
        std::cout<<"The object has been saved successfully..."<<std::endl;
    }
    else {
        std::cout<<"Error in opening the file..."<<std::endl;
    }
    return;
}

// in this function it'll be testing phase and object detection takes place
void testing(cv::Mat &src, cv::Mat &dst, int thresh) {
    // read the data from DB
    std::string PATH = "/home/arun/Documents/data.txt";
    std::ifstream file(PATH);
    std::vector<std::string> object_name;
    std::vector<float> object_af;
    std::vector<float> object_ht_wt;
    if(file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            std::istringstream iss(line);
            std::string str2;
            int count = 0;
            while(std::getline(iss, str2, ';')){
                if(count == 0) {
                    object_name.push_back(str2);
                }
                else if(count == 1) {
                    object_af.push_back(std::stof(str2));
                }
                else {
                    object_ht_wt.push_back(std::stof(str2));
                }
                count += 1;
            }
        }
    }
    cv::Mat labels, centroid, stats;
    int numlabels = cv::connectedComponentsWithStats(src, labels, stats, centroid);
    std::vector<cv::Vec3b> colors = {cv::Vec3b(255, 0, 0), cv::Vec3b(255, 0, 255), cv::Vec3b(0, 0, 255),
                                     cv::Vec3b(255, 255, 0), cv::Vec3b(0, 255, 255), cv::Vec3b(255, 0, 255)};
    std::vector<int> accepted_label(numlabels, 0);
    int maxval = 0;
    for(int i{1}; i < numlabels; i++) {
        if(stats.at<int>(i, cv::CC_STAT_LEFT) <= 0 or stats.at<int>(i, cv::CC_STAT_TOP) <= 0) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH) >= src.cols-50) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT) >= src.rows-50) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_AREA) < thresh) {
            continue;
        }
        maxval += 1;
        accepted_label[i] = maxval;
    }
    float area;
    float ht_wt_ratio;
    for(int i{1}; i < numlabels; i++) {
        if(accepted_label[i] >= 1) {
            cv::Mat temp = (labels == i);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(temp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<cv::Point> largestContour = contours[0];
            cv::RotatedRect rotated_rect = cv::minAreaRect(largestContour);
            cv::Point2f rect_vertices[4];
            rotated_rect.points(rect_vertices);
            ht_wt_ratio = (float)rotated_rect.size.width/(float)rotated_rect.size.height;
            cv::Point2f edge1 = rect_vertices[1] - rect_vertices[0];
            cv::Point2f edge2 = rect_vertices[2] - rect_vertices[1];
            area = std::abs(edge1.x * edge2.y - edge1.y * edge2.x);
            for(int j{0}; j < 4; j++) {
                cv::line(dst, rect_vertices[j], rect_vertices[(j+1)%4], cv::Scalar(255, 0, 0), 3);
            }
            cv::Mat roi_image = temp.clone();
            cv::Moments moments = cv::moments(roi_image);
            double m00 = moments.m00;
            double m10 = moments.m10;
            double m01 = moments.m01;
            double m11 = moments.mu11;
            double m02 = moments.mu02;
            double m20 = moments.mu20;
            cv::Mat cov_mat = (cv::Mat_<double>(2, 2) << m20 / m00, m11 / m00, m11 / m00, m02 / m00);
            cv::Mat eigenvalues, eigenvectors;
            cv::eigen(cov_mat, eigenvalues, eigenvectors);
            int centroidX = (float)m10/(float)m00;
            int centroidY = (float)m01/(float)m00;
            cv::arrowedLine(dst, cv::Point(centroidX, centroidY),
                            cv::Point(centroidX + 72*eigenvectors.at<double>(0, 0), centroidY + 72*eigenvectors.at<double>(0, 1)), cv::Scalar(0, 255, 0), 2);
            std::string name = "unknown";
            float value = 0.00001;
            for(int j{0}; j < (int)object_name.size(); j++) {
                float ar = (float)stats.at<int>(i, cv::CC_STAT_AREA)/area;
                float a = (ar - object_af[j])*(ar - object_af[j]);
                float h = (ht_wt_ratio - object_ht_wt[j])*(ht_wt_ratio - object_ht_wt[j]);
                if(a*h < value) {
                    name = object_name[j];
                    value = a*h;
                }
            }
            cv::putText(dst, name, cv::Point(centroidX, centroidY), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
    }
    return;
}
// this function executes knn with k value getting from user
int knn(cv::Mat &src, cv::Mat &dst, int k, int thresh) {
    // read the data from DB
    std::string PATH = "/home/arun/Documents/data.txt";
    std::ifstream file(PATH);
    std::vector<std::string> object_name;
    std::vector<float> object_af;
    std::vector<float> object_ht_wt;
    if(file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            std::istringstream iss(line);
            std::string str2;
            int count = 0;
            while(std::getline(iss, str2, ';')){
                if(count == 0) {
                    object_name.push_back(str2);
                }
                else if(count == 1) {
                    object_af.push_back(std::stof(str2));
                }
                else {
                    object_ht_wt.push_back(std::stof(str2));
                }
                count += 1;
            }
        }
    }
    cv::Mat labels, centroid, stats;
    int numlabels = cv::connectedComponentsWithStats(src, labels, stats, centroid);
    std::vector<cv::Vec3b> colors = {cv::Vec3b(255, 0, 0), cv::Vec3b(255, 0, 255), cv::Vec3b(0, 0, 255),
                                     cv::Vec3b(255, 255, 0), cv::Vec3b(0, 255, 255), cv::Vec3b(255, 0, 255)};
    std::vector<int> accepted_label(numlabels, 0);
    int maxval = 0;
    for(int i{1}; i < numlabels; i++) {
        if(stats.at<int>(i, cv::CC_STAT_LEFT) <= 0 or stats.at<int>(i, cv::CC_STAT_TOP) <= 0) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH) >= src.cols-50) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT) >= src.rows-50) {
            continue;
        }
        if(stats.at<int>(i, cv::CC_STAT_AREA) < thresh) {
            continue;
        }
        maxval += 1;
        accepted_label[i] = maxval;
    }
    float area;
    float ht_wt_ratio;
    for(int i{1}; i < numlabels; i++) {
        if(accepted_label[i] >= 1) {
            cv::Mat temp = (labels == i);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(temp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<cv::Point> largestContour = contours[0];
            cv::RotatedRect rotated_rect = cv::minAreaRect(largestContour);
            cv::Point2f rect_vertices[4];
            rotated_rect.points(rect_vertices);
            ht_wt_ratio = (float)rotated_rect.size.width/(float)rotated_rect.size.height;
            cv::Point2f edge1 = rect_vertices[1] - rect_vertices[0];
            cv::Point2f edge2 = rect_vertices[2] - rect_vertices[1];
            area = (float)stats.at<int>(i, cv::CC_STAT_AREA)/std::abs(edge1.x * edge2.y - edge1.y * edge2.x);
            std::vector<std::string> obj_name;
            obj_name = object_name;
            auto last = std::unique(obj_name.begin(), obj_name.end());
            obj_name.erase(last, obj_name.end());
            std::vector<std::vector<float>> sq_distance((int)obj_name.size(), std::vector<float>(0));
            for(int j{0}; j < (int)object_name.size(); j++) {
                for(int l{0}; l < (int)obj_name.size(); l++) {
                    if(object_name[j] == obj_name[l]) {
                        float a = (area - object_af[j])*(area - object_af[j]);
                        float h = (ht_wt_ratio - object_ht_wt[j])*(ht_wt_ratio - object_ht_wt[j]);
                        sq_distance[l].push_back(a*h);
                    }
                }
            }
            float storage = INT_MAX;
            std::string name = "";
            for(int j{0}; j < (int)obj_name.size(); j++) {
                std::sort(sq_distance[j].begin(), sq_distance[j].end());
                float tmp = 0;
                for(int l{0}; l < k and l < (int)sq_distance[j].size(); l++) {
                    tmp += sq_distance[j][l];
                }
                if(storage > tmp) {
                    storage = tmp;
                    name = obj_name[j];
                }
            }
            cv::putText(dst, name, cv::Point(72, 72), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
    }
    return 0;
}
