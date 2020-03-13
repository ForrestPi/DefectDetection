#ifndef _FIND_SCREEN_OBJECT_H
#define _FIND_SCREEN_OBJECT_H

#ifdef FINDRECTANGLELIB_EXPORTS
#define FINDRECTANGLEDLL_API __declspec(dllexport) 
#else
#define FINDRECTANGLEDLL_API
#endif

#include <opencv2/opencv.hpp>

using namespace cv;

typedef struct ScreenObjectInfo_{
    float center_x;
    float center_y;
    float hole_x;
    float hole_y;
    float angle;//in degree
    float possibility;
} ScreenObjectInfo;

/////////////////
// find the rectangle object with a hole on it
// image: CV_8UC1 or CV_8UC3
// fWidth: the width in pixels of the screen object 
// fHeight: the height in pixels of the screen object 
// fCircleRadius: the radius in pixels of the camera hole
// fCircleOffsetX: in offset of the camera hole related to the center of the screen
// fCircleOffsetY: in offset of the camera hole related to the center of the screen
// fHoleSearchRegionSize: the width and height of the searching region for the camera hole
/////////////////
FINDRECTANGLEDLL_API ScreenObjectInfo findScreenObject(Mat image, float fWidth, float fHeight,
                                  float fHoleRadius, float fHoleOffsetX, float fHoleOffsetY, float fHoleSearchRegionSize, 
                                  const char * sResultImageFileName=0);
#endif