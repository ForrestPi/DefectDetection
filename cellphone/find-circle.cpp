#include <iostream>
#include "find-circle.hpp"

using namespace std;

//////////////////////////////////////////////////////////////////////
// img: CV_8UC1 or CV_8UC3
// delta: resolution
//////////////////////////////////////////////////////////////////////
Mat calCircleHoughMat(const Mat img, float delta, float radius)
{
    Mat accum;
    Mat gray;
    Mat dX, dY;

    if (img.channels() == 1)
        gray = img.clone();
    else if (img.channels() == 3)
        cv::cvtColor(img, gray, CV_BGR2GRAY);
    else
    {
        cerr << "Unsupported image format. The image must be 1- or 3-channel." << endl;
        return accum;
    }
    GaussianBlur(gray, gray, Size(5, 5), 2);
    //cal dx & dy
    Sobel(gray, dX, CV_16SC1, 1, 0, 3);
    Sobel(gray, dY, CV_16SC1, 0, 1, 3);

    //clear memory
    accum.create(gray.size(), CV_32SC1);
    memset(accum.ptr(0), 0, accum.step * accum.rows);

    for (int r = 1; r < gray.rows - 1; r++)
    {
        short * pDx = dX.ptr<short>(r);
        short * pDy = dY.ptr<short>(r);
        for (int c = 1; c < gray.cols - 1; c++)
        {
            float fdx = pDx[c];
            float fdy = pDy[c];
            float gradient = sqrt(fdx*fdx + fdy*fdy);
            float ratio = radius / gradient;
            fdx *= ratio;
            fdy *= ratio;

            int x1 = cvRound(c - fdx);
            int y1 = cvRound(r - fdy);
            int x2 = cvRound(c + fdx);
            int y2 = cvRound(r + fdy);

            if (x1 >= 0 && x1 < accum.cols && y1 >= 0 && y1 < accum.rows)
                accum.at<int>(y1, x1) += 1;// cvRound(gradient);
            if (x2 >= 0 && x2 < accum.cols && y2 >= 0 && y2 < accum.rows)
                accum.at<int>(y2, x2) += 1;// cvRound(gradient);
        }
    }


    return accum;
}

CircleInfo findCircle(Mat image, float radius, const char * sResultImageFileName)
{
    CircleInfo ci;
    ci.center_x = -1;
    ci.center_y = -1;
    ci.radius = -1;
    ci.possibility = 0;

    if(image.empty())
    {
        cerr << __FILE__ << ": " << __LINE__ <<  ": The input image is empty" << endl;
        return ci;
    }


    Mat accum = calCircleHoughMat(image, 1.0, radius);

    double minVal, maxVal;
    Point pt; //strongest circle

    //find the strongest cicle
    minMaxLoc(accum, &minVal, &maxVal, NULL, &pt);

    ci.center_x = (float)(pt.x);
    ci.center_y = (float)(pt.y);
    ci.radius = radius;
    ci.possibility = (float)(maxVal / CV_PI / 2.f / radius / 100.f);

    //draw the circle
#if 0
    {
        // draw the circle
        circle(image, pt, 2, CV_RGB(0, 255, 0), 3, 8, 0);
        cout << "circle center: " << pt.x << ", " << pt.y << endl;

        imshow("circle", image);
    }
#endif
    return ci;
}