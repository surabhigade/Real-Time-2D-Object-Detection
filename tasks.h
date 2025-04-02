#include <iostream>
#include <opencv2/opencv.hpp>
#ifndef TASKS_H 
#define TASKS_H
void threshold(cv::Mat &src, cv::Mat &dst, int thresh); 
int threshold_kmeans(cv::Mat &src, cv::Mat &dst);
void image_segment(cv::Mat &src, cv::Mat &dst, std::string name, int thresh, bool task_3); 
void testing(cv::Mat &src, cv::Mat &dst, int thresh); 
int knn(cv::Mat &src, cv::Mat &dst, int k, int thresh); 
void dilate_built(cv::Mat &src, cv::Mat &dst, bool connect8, bool inverse, int iteration); 

#endif// TASKS_H
