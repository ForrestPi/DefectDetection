#ifndef  _FIND_RECTANGLE_H
#define  _FIND_RECTANGLE_H


#include <opencv2/opencv.hpp>

using namespace cv;

typedef struct RectInfo_{
    float center_x;
    float center_y;
    float width;
    float height;
    float angle;//in degree
    float possibility;
} RectInfo;

/////////////////
// image: CV_8UC1 or CV_8UC3
// fWidth: width of the rectangle
// fHeight: height of the rectangle
/////////////////
RectInfo findRectangle(Mat image, float fWidth, float fHeight, const char * sResultImageFileName=0);

#endif // ! _FIND_RECTANGLE_H