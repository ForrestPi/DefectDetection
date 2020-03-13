#ifndef  _FIND_CIRCLE_H
#define  _FIND_CIRCLE_H

#include <opencv2/opencv.hpp>

using namespace cv;

typedef struct CircleInfo_{
    float center_x;
    float center_y;
    float radius;
    float possibility;
} CircleInfo;

/////////////////
// the function can only detection one circle with a fixed radius
// image: CV_8UC1 or CV_8UC3
// delta: resolution
/////////////////
CircleInfo findCircle(Mat image, float radius, const char * sResultImageFileName=0);

#endif
