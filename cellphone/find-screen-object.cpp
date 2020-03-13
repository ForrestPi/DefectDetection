#include "find-screen-object.hpp"
#include "find-rectangle.hpp"
#include "find-circle.hpp"

using namespace cv;
using namespace std;

ScreenObjectInfo findScreenObject(Mat image, float fWidth, float fHeight,
    float fHoleRadius, float fHoleOffsetX, float fHoleOffsetY, float fHoleSearchRegionSize,
    const char * sResultImageFileName) 
{
    ScreenObjectInfo soi;
    soi.center_x = -1;
    soi.center_y = -1;
    soi.possibility = 0.0f;

    if (image.empty())
    {
        cerr << __FILE__ << ": " << __LINE__ << ": The input image is empty." << endl;
        return soi;
    }
    if (fWidth <= 0 || fWidth > image.cols || fHeight <= 0 || fHeight > image.rows)
    {
        cerr << __FILE__ << ": " << __LINE__ << ": The width or height is not correct." << endl;
    }

    //find the rectangle surrounded the screen
    RectInfo ri = findRectangle(image, fWidth, fHeight);
    //if cannot find it
    if (ri.center_x < 0)
        return soi;
    
    soi.center_x = ri.center_x;
    soi.center_y = ri.center_y;
    soi.angle = ri.angle;
    soi.possibility = ri.possibility;

    //estimate the position of the camera hole
    Rect holeRegion1, holeRegion2;
    double angle = ri.angle / 180.0 * CV_PI;
    float fHoleX = (float)(cos(angle) * fHoleOffsetX - sin(angle) * fHoleOffsetY);
    float fHoleY = (float)(sin(angle) * fHoleOffsetX + cos(angle) * fHoleOffsetY);

    holeRegion1.x = cvRound(ri.center_x + fHoleX - fHoleSearchRegionSize / 2);
    holeRegion1.y = cvRound(ri.center_y + fHoleY - fHoleSearchRegionSize / 2);
    holeRegion1.width = holeRegion1.height = cvRound(fHoleSearchRegionSize);

    //the opposite position of the possile region
    holeRegion2.x = cvRound(ri.center_x - fHoleX - fHoleSearchRegionSize / 2);
    holeRegion2.y = cvRound(ri.center_y - fHoleY - fHoleSearchRegionSize / 2);
    holeRegion2.width = holeRegion2.height = cvRound(fHoleSearchRegionSize);

    //make sure that the regions are not out of the image region
    if (holeRegion1.x < 0)
    {
        holeRegion1.width += holeRegion1.x;
        holeRegion1.x = 0;
    }
    if (holeRegion1.y < 0)
    {
        holeRegion1.height += holeRegion1.y;
        holeRegion1.y = 0;
    }
    if (holeRegion2.x + holeRegion2.width >= image.cols)
        holeRegion2.width -= (image.cols - holeRegion2.x - holeRegion2.width);
    if (holeRegion2.y + holeRegion2.height >= image.rows)
        holeRegion2.height -= (image.rows - holeRegion2.y - holeRegion2.height);
    if (holeRegion2.x < 0)
    {
        holeRegion2.width += holeRegion2.x;
        holeRegion2.x = 0;
    }
    if (holeRegion2.y < 0)
    {
        holeRegion2.height += holeRegion2.y;
        holeRegion2.y = 0;
    }
    if (holeRegion2.x + holeRegion2.width >= image.cols)
        holeRegion2.width -= (image.cols - holeRegion2.x - holeRegion2.width);
    if (holeRegion2.y + holeRegion2.height >= image.rows)
        holeRegion2.height -= (image.rows - holeRegion2.y - holeRegion2.height);

    //find the circles in the two regions
    Mat region1 = image(holeRegion1);
    Mat region2 = image(holeRegion2);

    CircleInfo ci1 = findCircle(region1, fHoleRadius);
    CircleInfo ci2 = findCircle(region2, fHoleRadius);

    if (ci1.center_x < 0 || ci2.center_x < 0)
        return soi;

    ci1.center_x += holeRegion1.x;
    ci1.center_y += holeRegion1.y;
    ci2.center_x += holeRegion2.x;
    ci2.center_y += holeRegion2.y;


    if (ci1.possibility >= ci2.possibility)
    {
        soi.hole_x = ci1.center_x;
        soi.hole_y = ci1.center_y;
        //ajust the rectangle center according to the hole's position
        soi.center_x -= (ri.center_x + fHoleX - ci1.center_x); 
        soi.center_y -= (ri.center_y + fHoleY - ci1.center_y);
    }
    else
    {
        soi.angle += 180;
        soi.hole_x = ci2.center_x;
        soi.hole_y = ci2.center_y;
        //ajust the rectangle center according to the hole's position
        soi.center_x -= (ri.center_x - fHoleX - ci2.center_x);
        soi.center_y -= (ri.center_y - fHoleY - ci2.center_y);
    }

#if 0
    {
        Mat draw = image.clone();
        circle(draw, Point(cvRound(ri.center_x), cvRound(ri.center_y)), 25, CV_RGB(255, 0, 0), 1, 8, 0);

        circle(draw, Point(cvRound(soi.hole_x), cvRound(soi.hole_y)), 2, CV_RGB(0, 255, 0), 3, 8, 0);
        circle(draw, Point(cvRound(soi.center_x), cvRound(soi.center_y)), 25, CV_RGB(0, 255, 0), 1, 8, 0);
        ellipse(draw, Point(cvRound(soi.center_x), cvRound(soi.center_y)), Size(cvRound(fWidth/2), cvRound(fHeight/2)), soi.angle, 160, 360+150, CV_RGB(0,255,0),1);

        //draw possibility
        char sP[1024];
        sprintf_s(sP, "score=%.2f", soi.possibility);
        putText(draw, sP, Point(cvRound(soi.center_x), cvRound(soi.center_y - 20)), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);


        if(sResultImageFileName)
            imwrite(sResultImageFileName, draw);
        imshow("result", draw);
    }
#endif
    return soi;
}

