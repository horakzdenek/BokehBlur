#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "MojeCV.hpp"

using namespace std;
using namespace cv;

int main(int, char**)
{
    Mat frame = imread("src.png");
    Mat k = imread("kernel_2.png",IMREAD_GRAYSCALE);



    Mat dst = MOJECV::bokeh(frame,k,25,0);



    imshow("x",dst);
    imwrite("result.png",dst);



    waitKey(0);
    return 0;


}
