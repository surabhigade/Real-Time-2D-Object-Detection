#include <iostream>
#include <opencv2/opencv.hpp>
#include "tasks.h"
#include "kmeans.h"

// this is a function to get i/p from the user
std::string user_inp() {

   std::string name;
   std::cout<<"!!! For kind information - Please save only one object at a time"<<std::endl;

   std::cout<<"Please enter the name of the object: ";
   std::cin>>name;

   return name;
}

int main()
{
    // these're typical tasks that we're doing for the past 2 projects so i don't need to explain most part
    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);

    if(not capdev->isOpened()) {
        std::cout<<"Unable to open the camera..."<<std::endl;
        return(0);
    }

    cv::Size refS((int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                        (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout<<"Frame height: "<<refS.height<<"   Frame width: "<<refS.width;

    cv::namedWindow("processed_video", 1);
    cv::namedWindow("original_video", 1);

    cv::Mat frame, frame_clone;
    int filter = 0;
    char key;

    while(true) {
        *capdev >> frame;
        if(frame.empty()) {
            std::cout<<"The frame is empty...";
            break;
        }
    
        frame_clone = frame.clone();
    
        if(filter == 1) { // just a threshold - part of task_1
            cv::Mat temp;
            cv::cvtColor(frame, temp, cv::COLOR_RGB2GRAY);
            threshold(temp, frame, 100);
        }
        else if(filter == 2) { // threshold using k-means - part of task_1
            // this is made possible by prof. code (k-means). thanks
            cv::Mat temp = frame.clone();
            if(threshold_kmeans(temp, frame)) {
                std::cout<<"There is some error in the kmeans..."<<std::endl;
                filter = 0;
            }
        }
        else if(filter >= 3 and filter < 7) { // this is basically task_2, 3, 4 and 5
            // first we perform morphological filters like dilation and filters
            cv::Mat temp1, temp2;
            cv::Mat temp = frame.clone();
    
            cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
            threshold(frame, frame, 100);
    
            dilate_built(frame, temp1, true, false, 3);
            dilate_built(temp1, temp2, false, true, 3);
    
            dilate_built(temp2, temp1, true, true, 3);
            dilate_built(temp1, frame, false, false, 3);
    
            if(filter >= 4) { // Segment the image into regions - task_3
                temp = frame.clone();
                if(filter == 4) {
                    image_segment(temp, frame, "", 300, true);
                }
                else if(filter == 5) { // Compute features for each major region - task_4
                    image_segment(temp, frame, "", 300, false);
                }
                else { // for saving an unknown object, training mode - task_5
                    std::string str = user_inp();
                    image_segment(temp, frame, str, 300, false);
    
                    if(str != "") {
                        filter = 5;
                    }
                }
            }
        }



        else if(filter == 7) { // testing mode - task_6
            cv::Mat temp1, temp2;
            cv::Mat temp = frame.clone();
            cv::Mat frame3 = frame.clone();

            cv::cvtColor(frame3, frame3, cv::COLOR_RGB2GRAY);
            threshold(frame3, frame3, 100);

            dilate_built(frame3, temp1, true, false, 3);
            dilate_built(temp1, temp2, false, true, 3);

            dilate_built(temp2, temp1, true, true, 3);
            dilate_built(temp1, frame3, false, false, 3);

            testing(frame3, frame, 300);
        }
        
        else if(filter == 8) { // knn - task_9_A
            cv::Mat temp1, temp2;
            cv::Mat temp = frame.clone();
            cv::Mat frame3 = frame.clone();

            cv::cvtColor(frame3, frame3, cv::COLOR_RGB2GRAY);
            threshold(frame3, frame3, 100);

            dilate_built(frame3, temp1, true, false, 3);
            dilate_built(temp1, temp2, false, true, 3);

            dilate_built(temp2, temp1, true, true, 3);
            dilate_built(temp1, frame3, false, false, 3);

            knn(frame3, frame, 3, 300);
        }

        cv::imshow("processed_video", frame);
        cv::imshow("original_video", frame_clone);

        key = cv::waitKey(10);

        if(key == 'q') { // to quit the video
            break;
        }
        else if(key == 't') { // threshold - part of task_1
            filter = 1;
        }
        else if(key == 'r') { // reset
            filter = 0;
        }
        else if(key == 'k') { // threshold using k-means - part of task_1

            filter = 2;
        }
        else if(key == 'm') { // clean up the binary image using morphological filters - task_2
            filter = 3;
        }
        else if(key == 'g') { // Segment the image into regions - task_3
            filter = 4;
        }
        else if(key == 'd') { // Compute features for each major region - task_4
            filter = 5;
        }
        else if(key == 'n' and filter == 5) { // for saving an unknown object, training mode - task_5
            filter = 6;
        }
        else if(key == 'e') { // testing mode - task_6
            filter = 7;
        }
        else if(key == 'b') { // knn - task_9_A
            filter = 8;
        }

    }

}
