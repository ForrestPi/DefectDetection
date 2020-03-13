#include <iostream>
#include "find-rectangle.hpp"

using namespace std;

//////////////////////////////////////////////////////////////////////
// img: CV_8UC1 or CV_8UC3
// deltaRho: resolution for rho
// deltaTheta: resolution for Theta
// bHorizontal: true for horizontal lines, the grident at Y direction is used for horizontal line detection
//              false for verticle lines, the grident at X direction is used for verticle line detection
// fVerticleLineAngle: only valid when bHorizontal=true
//////////////////////////////////////////////////////////////////////
Mat calHoughMat(const Mat img, float deltaRho, float deltaTheta, bool bHorizontal=true, float fVerticleLineAngle=0.0f)
{
    cv::AutoBuffer<float> _tabSin, _tabCos;
    Mat _accum;

    const uchar* image_data;
    int step, width, height;
    int numangle, numrho;
    int total = 0;
    int i, j;
    float irho = 1 / deltaRho;

    Mat gray;
    Mat edge;
    if(img.channels() == 1)
        gray = img.clone();
    else if(img.channels() == 3)
        cvtColor(img, gray, CV_BGR2GRAY);
    else
    {
        cerr << "Unsupported image format. The image must be 1- or 3-channel." <<endl;
        return _accum;
    }
    GaussianBlur( gray, gray, Size(5,5), 2 );
    Mat gradient;
    
    if(bHorizontal)
        Sobel(gray, gradient, CV_16S, 0, 1, 3);
    else
        Sobel(gray, gradient, CV_16S, 1, 0, 3);
    convertScaleAbs(gradient, edge, 0.25);

    step = (int)edge.step;
    width = edge.cols;
    height = edge.rows;

    numangle = cvRound(CV_PI / deltaTheta);
    numrho = cvRound(((width + height) * 2 + 1) / deltaRho);

    _accum.create((numangle+2), (numrho+2), CV_32SC1);
    _tabSin.allocate(numangle);
    _tabCos.allocate(numangle);
    int *accum = _accum.ptr<int>(0);
    float *tabSin = _tabSin, *tabCos = _tabCos;

    memset( _accum.ptr<int>(0), 0, sizeof(accum[0]) * _accum.cols * _accum.rows );

    float ang = 0;
    for(int n = 0; n < numangle; ang += deltaTheta, n++ )
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }

    // fill accumulator
    image_data = edge.ptr<unsigned char>(0);
    //for horizontal lines
    if(bHorizontal)
    {
        for( i = 0; i < height; i++ )
        {
            for( j = 0; j < width; j++ )
            {
                for(int n = numangle*7/18; n < numangle*11/18; n++ ) //only lines near horizontal
                {
                    int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                    r += (numrho - 1) / 2;
                    accum[(n+1) * (numrho+2) + r+1] += image_data[i * step + j];
                }
            }
        }
    }
    else //for verticle lines
    {
         for( i = 0; i < height; i++ )
        {
            for( j = 0; j < width; j++ )
            {
                //only the lines perpendicular to the horizontal lines are considered
                int n = cvRound(fVerticleLineAngle/deltaTheta); 
                {
                    int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                    r += (numrho - 1) / 2;
                    accum[(n+1) * (numrho+2) + r+1] += image_data[i * step + j];
                }
            }
        }
   }

    return _accum;
}

//////////////////////////////////////////////////////////////////////
// find two parall horizontal lines
// accum: input
// deltaRho: input
// deltaTheta: input
// pRho1: output, the parameter for line1
// pTheta1: output, the parameter for line1
// pRho2: output, the parameter for line2
// pTheta2: output, the parameter for line2
//////////////////////////////////////////////////////////////////////
float findTwoStrongParallelLinesByDistance(const Mat accum, float deltaRho, float deltaTheta, float fDistance, int nImageHeight, float * pRho1, float * pTheta1, float * pRho2, float * pTheta2)
{
    Point pt1; //strongest line 
    Point pt2; //the parallel line
    int numrho = accum.cols - 2;
    double maxVal;
    /*
    double minVal;
    //find the strongest line
    minMaxLoc(accum, &minVal, &maxVal, NULL, &pt1);

    //find the parallel line
    maxVal = -1;
    pt2.x = 0;
    pt2.y = pt1.y;

    for(int rhoIdx =0; rhoIdx < accum.cols; rhoIdx++)
    {
        if(accum.at<int>(pt2.y, rhoIdx) > maxVal && 
            abs(pt1.x - rhoIdx) > accum.cols/10)
        {
            maxVal = accum.at<int>(pt2.y, rhoIdx);
            pt2.x = rhoIdx;
        }
    }
   */
    int nDistance = cvRound(fDistance / deltaRho);

    maxVal = -1.0f;
    for (int thetaIdx = 0; thetaIdx < accum.rows; thetaIdx++)
    {
        for (int rhoIdx = 0; rhoIdx < accum.cols ; rhoIdx++)
        {
            int rhoIdx2 = rhoIdx + nDistance;
            if (rhoIdx2 >= accum.cols)
                rhoIdx2 -= accum.cols;

            float f = (float)(accum.at<int>(thetaIdx, rhoIdx) + accum.at<int>(thetaIdx, rhoIdx2));
            if (f > maxVal)
            {
                maxVal = f;
                pt1.x = rhoIdx;
                pt1.y = thetaIdx;
                pt2.x = rhoIdx2;
                pt2.y = thetaIdx;
            }
        }
    }

    *pTheta1 = (pt1.y-1) * deltaTheta;
    *pRho1 = (pt1.x - (numrho - 1)/2) * deltaRho;
    *pTheta2 = (pt2.y-1) * deltaTheta;
    *pRho2 = (pt2.x - (numrho - 1)/2) * deltaRho;

    return (float)maxVal;
}

//////////////////////////////////////////////////////////////////////
// find two parallel horizontal lines
// accum: input
// deltaRho: input
// deltaTheta: input
// fAngle: input, the angle of the lines (only lines at this angle are considered)
// fDistance: input, the distance between two parallel lines
// pRho1: output, the parameter for line1
// pTheta1: output, the parameter for line1
// pRho2: output, the parameter for line2
// pTheta2: output, the parameter for line2
//////////////////////////////////////////////////////////////////////
float findTwoParallelLinesByAngleDistance(const Mat accum, float deltaRho, float deltaTheta, float fAngle, float fDistance, int nImageWidth, float * pRho1, float * pTheta1, float * pRho2, float * pTheta2)
{
    float maxVal = -1;
    int numrho = accum.cols - 2;
    int nDistance = cvRound(fDistance/deltaRho);

    Point pt1;
    Point pt2;
    pt1.y =  cvRound(fAngle/deltaTheta+1);
    pt2.y =  cvRound(fAngle/deltaTheta+1);

    //to reduce the search region
    int nRange = cvRound((nImageWidth - fDistance) / deltaRho);
    //int nStartIdx = (accum.cols / 2 - nRange);
    int nStartIdx = (accum.cols / 2 - 2);
    int nEndIdx = (accum.cols / 2+ nRange);
    nEndIdx = MIN(nEndIdx, (accum.cols - nDistance - 2));

    if (fAngle > CV_PI / 2)
    {
        nStartIdx = (accum.cols / 2 - nDistance) - nDistance;
        nEndIdx = (accum.cols / 2 + 2 ) - nDistance;
        if (nStartIdx < 0)
            nStartIdx = 0;
    }

    for(int rhoIdx = nStartIdx; rhoIdx < nEndIdx; rhoIdx++)
    //for (int rhoIdx = 0; rhoIdx < accum.cols - nDistance - 2; rhoIdx++)
    {
        int val = accum.at<int>(pt2.y, rhoIdx) + accum.at<int>(pt2.y, rhoIdx+nDistance);
        if( val > maxVal)
        {
            maxVal = (float)val;
            pt1.x = rhoIdx;
            pt2.x = rhoIdx + nDistance;
        }
    }

    *pTheta1 = (pt1.y-1) * deltaTheta;
    *pRho1 = (pt1.x - (numrho - 1)/2) * deltaRho;
    *pTheta2 = (pt2.y-1) * deltaTheta;
    *pRho2 = (pt2.x - (numrho - 1)/2) * deltaRho;

    return maxVal;
}


void drawLinebyRhoBeta(Mat image, float rho, float theta, Scalar color)
{
    Point pt1, pt2;
    if(fabs(sin(theta)) > 0.707)//horizontal line
    {
        pt1.x=0;
        pt1.y = cvRound(rho / sin(theta));
        pt2.x = image.cols - 1;
        pt2.y = cvRound( (rho - pt2.x * cos(theta)) / sin(theta) );
    }
    else //verticle line
    {
        pt1.y = 0;
        pt1.x = cvRound(rho / cos(theta));
        pt2.y = image.rows - 1;
        pt2.x = cvRound( (rho -  pt2.y * sin(theta)) / cos(theta));
    }

    cv::line(image, pt1, pt2, color, 1);
}

RectInfo findRectangle(Mat image, float fWidth, float fHeight, const char * sResultImageFileName)
{
    RectInfo ri;
    ri.center_x = -1;
    ri.center_y = -1;
    ri.width = -1;
    ri.height = -1;
    ri.angle = 0;

    if(image.empty())
    {
        cerr << __FILE__ << ": " << __LINE__ <<  ": The input image is empty" << endl;
        return ri;
    }

    //resolution
    float deltaRho = 1.0f;
    float deltaTheta = (float)CV_PI/180;
    //four edges of the rectangle
    float rho1, theta1;
    float rho2, theta2;
    float rho3, theta3;
    float rho4, theta4;

    //the rho-theta matrix for the horizontal edges
    Mat accum = calHoughMat(image, deltaRho, deltaTheta, true);

    //find two parallel horizontal lines
    ri.possibility = findTwoStrongParallelLinesByDistance(accum, deltaRho, deltaTheta, fHeight, image.rows, &rho1, &theta1, &rho2, &theta2);

    //the angle of the vertical edges
    float angle = theta1-(float)CV_PI/2;
    if(angle < 0)
        angle = theta1+(float)CV_PI/2;

    //the rho-theta matrix for the verticle edges
    accum = calHoughMat(image, deltaRho, deltaTheta, false, angle);
    //the two parallel verticle edges
    ri.possibility += findTwoParallelLinesByAngleDistance(accum, deltaRho, deltaTheta, angle, fWidth, image.cols, &rho3, &theta3, &rho4, &theta4);

    ri.possibility /= (30 * (image.cols + image.rows));
    if (ri.possibility > 1.0f)
        ri.possibility = 1.0f;

    //find the center of the rectangle
    {
        float deno = sin(theta1)*cos(theta3)-sin(theta3)*cos(theta1);
        float c1 = (rho1+rho2)/2.0f;
        float c2 = (rho3+rho4)/2.0f;
        ri.center_x = (sin(theta1)*c2 - sin(theta3)*c1)/deno;
        ri.center_y = (cos(theta3)*c1 - cos(theta1)*c2)/deno;
        
        ri.width = fabs(rho3-rho4);
        ri.height = fabs(rho1-rho2);
        ri.angle = (float)( theta1/CV_PI*180 - 90); //the horizontal line's angle is ZERO degree
    }

    //show the results
#if 0
    {
        Mat draw = image.clone();
        //draw the results
        drawLinebyRhoBeta(draw, rho1, theta1, CV_RGB(255,0,0));
        drawLinebyRhoBeta(draw, rho2, theta2, CV_RGB(0,255,0));

        drawLinebyRhoBeta(draw, rho3, theta3, CV_RGB(255,255,0));
        drawLinebyRhoBeta(draw, rho4, theta4, CV_RGB(0,0,255));

        //the two lines cross the center of the rectangle
        drawLinebyRhoBeta(draw, (rho1+rho2)/2.0f, theta1, CV_RGB(255,255,255));
        drawLinebyRhoBeta(draw, (rho3+rho4)/2.0f, theta3, CV_RGB(255,255,255));

        //draw the center
        cv::circle(draw, Point(cvRound(ri.center_x), cvRound(ri.center_y)), 25, CV_RGB(255,255,255));

        //draw possibility
        char sP[1024];
        sprintf(sP, "score=%.2f", ri.possibility);
        putText(draw, sP, Point(cvRound(ri.center_x), cvRound(ri.center_y - 20)), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255),2);

        //print the results
        cout << "Angle: " << ri.angle << endl;
        cout << "Rect size: [" << ri.width << ", " << ri.height << "]" << endl;
        cout << "Rect centor: " << ri.center_x << ", " << ri.center_y <<endl;
        cout << "Possibility: " << ri.possibility << endl;

        if (sResultImageFileName)
            imwrite(sResultImageFileName, draw);
        //show the result image in a window
        imshow("line", draw);
    }
#endif
    return ri;

}