// nadstavba pro OpenCV verzi 4.4.0
// update: 2020-12-27
// FIXME: port for 3.2.0
// update: 2021-05-30
#ifndef MOJECV_HPP_INCLUDED
#define MOJECV_HPP_INCLUDED

#include <iostream>
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;




namespace MOJECV
{
    // dopředná deklarace
    void CopyPixel(Mat src, Point a, Mat &dst, Point b);
    // End of dopředná deklarace


void bgr_to_bayer(Mat src, Mat &dst)
{
    int W,H;
    W = src.cols;
    H = src.rows;

    vector<Mat> bgr(3);
    split(src,bgr);
    Mat black(Size(W,H),CV_8UC1,Scalar(0));
    // green
    vector<Mat> dst_g(3);
    dst_g[0] = black;
    dst_g[1] = bgr[1];
    dst_g[2] = black;
    // blue
    vector<Mat> dst_b(3);
    dst_b[0] = bgr[0];
    dst_b[1] = black;
    dst_b[2] = black;
    //red
    vector<Mat> dst_r(3);
    dst_r[0] = black;
    dst_r[1] = black;
    dst_r[2] = bgr[2];

    // make separate bgr
    Mat cil_g,cil_b,cil_r;
    merge(dst_g,cil_g);
    merge(dst_b,cil_b);
    merge(dst_r,cil_r);
    Mat vystup(Size(2*W,2*H),CV_8UC3,Scalar(0,0,0));

    for(int y = 0; y < cil_b.rows; y++)
    {
        for(int x = 0; x < cil_b.cols; x++)
        {
            MOJECV::CopyPixel(cil_r,Point(x,y),vystup,Point(x*2,y*2));
            MOJECV::CopyPixel(cil_g,Point(x,y),vystup,Point((x*2)+1,y*2));
            MOJECV::CopyPixel(cil_g,Point(x,y),vystup,Point(x*2,(y*2)+1));
            MOJECV::CopyPixel(cil_b,Point(x,y),vystup,Point((x*2)+1,(y*2)+1));
        }
    }

    dst = vystup;

}

class ccd
{ // 3-2.0. compatible
public:
    ccd(void)
    {
        // konstruktor
    }
    void shutter(void)
    {
        snimac = Mat(resolution,CV_8UC3,Scalar(0,0,0));
    }
    void expose(Mat frame)
    {
        if(iterace == 0)
        {
            snimac = frame.clone();
            resolution.width = frame.cols;
            resolution.height = frame.rows;
        }else
        {
            snimac = ccd::lighten(snimac,frame);
        }
        iterace++;
    }
    Mat result(void)
    {
        return snimac;
    }


private:
    Size resolution;
    int iterace = 0;
    Mat snimac;
    Mat lighten(Mat a, Mat b)
    {
        Mat result = a.clone();
        Vec3b color_dest;
        for(int y = 0; y < a.rows; y++)
        {
            for (int x = 0; x < a.cols; x++)
            {
                // do it
                Vec3b intensity_a = a.at<Vec3b>(y, x);
                Vec3b intensity_b = b.at<Vec3b>(y, x);
                int blue_a = intensity_a.val[0];
                int green_a = intensity_a.val[1];
                int red_a = intensity_a.val[2];
                int blue_b = intensity_b.val[0];
                int green_b = intensity_b.val[1];
                int red_b = intensity_b.val[2];

                if((blue_a + green_a + red_a) > (blue_b + green_b + red_b))
                {
                    color_dest.val[0] = blue_a;
                    color_dest.val[1] = green_a;
                    color_dest.val[2] = red_a;
                }else
                {
                    color_dest.val[0] = blue_b;
                    color_dest.val[1] = green_b;
                    color_dest.val[2] = red_b;
                }

                result.at<Vec3b>(Point(x,y)) = color_dest;
            }
        }
        return result;
    }
};


class sensor // fixed for 3.2.0
{
    public:
        int iterace = 0;
        sensor(Mat frame) // konstruktor
        {
            chip = frame.clone();
            chip.convertTo(chip,CV_32FC3);
            chip = Scalar(0,0,0);
        }

        void expose(Mat frame)
        {
            Mat tmp = frame.clone();
            tmp.convertTo(tmp,CV_32FC3);
            chip += tmp;
            iterace++;
        }

        void shutter()
        {
            Mat tmp(Size(chip.cols,chip.rows),CV_32FC3,Scalar(0,0,0));
            chip = tmp.clone();
            iterace = 0;
        }

        Mat Output()
        {
            float alpha = (float)iterace;
            float beta = 1.0f / alpha;
            Mat x = chip * beta;
            Mat tmp;
            x.convertTo(tmp,CV_8UC3);
            return tmp;
        }
    private:
        Mat chip;

};

Mat translateImg(Mat img, int offsetx, int offsety)
{
    Mat t = img.clone();
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(t,t,trans_mat,t.size(),INTER_LINEAR,BORDER_REPLICATE);
    return t;
}

Mat bokeh(Mat img, Mat pattern, int s, int method)
{
    /** \brief
     *
     * \param img - source image
     * \param pattern - defocus kernel
     * \param s - blur size
     * \param method - 0/1
     * \return
     *
     */

    resize(pattern,pattern,Size(s,s),0,0,INTER_LANCZOS4);


    MOJECV::sensor snimac(img);
    MOJECV::ccd s2;
    int sx,sy;
    sx = (int)(pattern.cols/2.0);
    sy = (int)(pattern.rows/2.0);
    for(int y = 0; y < pattern.rows; y++)
    {
        for(int x = 0; x < pattern.cols; x++)
        {
            if(pattern.at<uchar>(y,x) == 255)
            {
                // do it
                if(method == 0)
                {
                    snimac.expose(translateImg(img,x-sx,y-sy));
                }
                if(method == 1)
                {
                    s2.expose(translateImg(img,x-sx,y-sy));
                }



            }
        }
    }

    if(method == 0)
    {
        return snimac.Output();
    }
    if(method == 1)
    {
        return s2.result();
    }


}




void show_histogram(std::string const& name, cv::Mat1b const& image)
{
    // Set histogram bins count
    int bins = 256;
    int histSize[] = {bins};
    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    // create matrix for histogram
    cv::Mat hist;
    int channels[] = {0};

    // create matrix for histogram visualization
    int const hist_height = 256;
    cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    double max_val=0;
    minMaxLoc(hist, 0, &max_val);

    // visualize each bin
    for(int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal*hist_height/max_val);
        cv::line
            ( hist_image
            , cv::Point(b, hist_height-height), cv::Point(b, hist_height)
            , cv::Scalar::all(255)
            );
    }
    cv::imshow(name, hist_image);
}


    void AutoCut(Mat &image, Mat &result, Rect vyrez, int iterace)
    {
        Mat res;
        Mat bgModel,fgModel;
        grabCut( image, res, vyrez, bgModel, fgModel, iterace, GC_INIT_WITH_RECT);
        Mat binMask;
        binMask = res & 1;
        Mat vystup = Mat(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        image.copyTo(vystup,binMask);
        result = vystup;
    }




    class preview
    {
        public:
            Mat frame;
            int iterace = 0;
            int frame_count = 1;
            string hint = "";

            Mat nahled()
            {
                Mat tmp;
                double pomer = (double)frame.cols/(double)frame.rows;
                resize(frame,tmp, Size((int)round(300*pomer),300));
                double pct = (double)iterace/(double)frame_count*100;
                int p = (int)round(pct);
                Point pt(0,12);
                Point pt1(0,0);
                Point pt2((int)round(0.01*p*tmp.cols),13);
                rectangle(tmp, pt2, pt1, Scalar(0, 0, 255), -1, 8, 0); /// Scalar definuje barvu -1 je výplň/tloušťka

                String text = to_string(p)+"%";
                if (iterace == 0)
                {
                    text = "";
                }
                int fontFace = FONT_HERSHEY_SIMPLEX;
                double fontScale = 0.4;
                int thickness = 1;
                putText(tmp, text, pt, fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
                if (!(hint == ""))
                {
                    rectangle(tmp, Point(tmp.cols,tmp.rows), Point(0,tmp.rows-15), Scalar(120, 120, 120), -1, 8, 0);
                    putText(tmp, hint, Point(0,tmp.rows-4), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
                }
                return tmp;
            }

    };

void sharpen2D(const cv::Mat &image, cv::Mat &result) {

	// Construct kernel (all entries initialized to 0)
	cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
	// assigns kernel values
	kernel.at<float>(1,1)= 5.0;
	kernel.at<float>(0,1)= -1.0;
	kernel.at<float>(2,1)= -1.0;
	kernel.at<float>(1,0)= -1.0;
	kernel.at<float>(1,2)= -1.0;

	//filter the image
	cv::filter2D(image,result,image.depth(),kernel);
}

void CopyPixel(Mat src, Point a, Mat &dst, Point b)
{
    Vec3b intensity_a = src.at<Vec3b>(a.y, a.x);
    dst.at<Vec3b>(b) = intensity_a;
}

Mat overlay(Mat a, Mat b)
{
        Mat tmp_hsv;
        cvtColor(a,tmp_hsv,COLOR_BGR2HSV);

        Mat img1 = tmp_hsv.clone();
        Mat img2 = b.clone();
        Mat result = a.clone();


        for(int i = 0; i < img1.size().height; ++i)
        {
            for(int j = 0; j < img1.size().width; ++j)
            {
                //float target = float(img1.at<uchar>(i, j)) / 255;
                Vec3b target_intensity = img1.at<Vec3b>(i, j);
                float target_b = float(target_intensity.val[0])/255;
                float target_g = float(target_intensity.val[1])/255;
                float target_r = float(target_intensity.val[2])/255;


                //float blend = float(img2.at<uchar>(i, j)) / 255;
                Vec3b blend_intensity = img2.at<Vec3b>(i, j);
                float blend_r = float(blend_intensity.val[2])/255;

                // overlay
                if(target_r > 0.5)
                {

                    //result.at<float>(i, j) = (1 - (1-2*(target-0.5)) * (1-blend));
                    Vec3b color;
                    color.val[0] = target_b*255;
                    color.val[1] = target_g*255;
                    color.val[2] = (1 - (1-2*(target_r-0.5)) * (1-blend_r))*255;
                    result.at<Vec3b>(i, j) = color;

                }
                else
                {
                    //result.at<float>(i, j) = ((2*target) * blend);
                    Vec3b color;
                    color.val[0] = target_b*255;
                    color.val[1] = target_g*255;
                    color.val[2] = ((2*target_r) * blend_r)*255;
                    result.at<Vec3b>(i, j) = color;
                }
            }
        }

        // zpětný převod na RGB model
        cvtColor(result,result,COLOR_HSV2BGR);

        return result;

}
void hdr(const Mat &image, Mat &result)
{
        int s = 225;
        Mat frame = image.clone();
        Mat gray = frame.clone();
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        cvtColor(gray, gray, COLOR_GRAY2BGR);
        bitwise_not(gray,gray);
        GaussianBlur(gray,gray,Size(s,s),0); // 175

        result = overlay(frame,gray);
}
Mat blend_multiply(const Mat& level1, const Mat& level2, uchar opacity)
{
    CV_Assert(level1.size() == level2.size());
    CV_Assert(level1.type() == level2.type());
    CV_Assert(level1.channels() == level2.channels());

    // Get 4 channel float images
    Mat4f src1, src2;

    if (level1.channels() == 3)
    {
        Mat4b tmp1, tmp2;
        cvtColor(level1, tmp1, COLOR_BGR2BGRA);
        cvtColor(level2, tmp2, COLOR_BGR2BGRA);
        tmp1.convertTo(src1, CV_32F, 1. / 255.);
        tmp2.convertTo(src2, CV_32F, 1. / 255.);
    }
    else
    {
        level1.convertTo(src1, CV_32F, 1. / 255.);
        level2.convertTo(src2, CV_32F, 1. / 255.);
    }

    Mat4f dst(src1.rows, src1.cols, Vec4f(0., 0., 0., 0.));

    // Loop on every pixel

    float fopacity = opacity / 255.f;
    float comp_alpha, new_alpha;

    for (int r = 0; r < src1.rows; ++r)
    {
        for (int c = 0; c < src2.cols; ++c)
        {
            const Vec4f& v1 = src1(r, c);
            const Vec4f& v2 = src2(r, c);
            Vec4f& out = dst(r, c);

            comp_alpha = min(v1[3], v2[3]) * fopacity;
            new_alpha = v1[3] + (1.f - v1[3]) * comp_alpha;

            if ((comp_alpha > 0.) && (new_alpha > 0.))
            {
                float ratio = comp_alpha / new_alpha;

                out[0] = max(0.f, min(v1[0] * v2[0], 1.f)) * ratio + (v1[0] * (1.f - ratio));
                out[1] = max(0.f, min(v1[1] * v2[1], 1.f)) * ratio + (v1[1] * (1.f - ratio));
                out[2] = max(0.f, min(v1[2] * v2[2], 1.f)) * ratio + (v1[2] * (1.f - ratio));
            }
            else
            {
                out[0] = v1[0];
                out[1] = v1[1];
                out[2] = v1[2];
            }

            out[3] = v1[3];

        }
    }

    Mat3b dst3b;
    Mat4b dst4b;
    dst.convertTo(dst4b, CV_8U, 255.);
    cvtColor(dst4b, dst3b, COLOR_BGRA2BGR);

    return dst3b;
}

    Mat Mix(Mat A, Mat B, double mix)
    {
        Mat dst;
        Mat a = A.clone();
        Mat b = B.clone();
        double beta = ( 1.0 - mix );
        addWeighted( A, mix, B, beta, 0.0, dst);
        return dst;
    }

    Mat Wave(Mat src, double period, double phase, int amp)
    {
        Mat result = src.clone();
        Vec3b color_dest;
        Point position = Point(0,0);

        for(int y = 0; y < src.rows; y++)
            {
                for (int x = 0; x < src.cols; x++)
                {
                    position.x = (int)(x + sin(y*period+phase)*amp);
                    position.y = (int)(y + sin(x*period+phase)*amp);

                    Vec3b intensity = src.at<Vec3b>(position);
                    int blue = intensity.val[0];
                    int green = intensity.val[1];
                    int red = intensity.val[2];

                    color_dest.val[0] = blue;
                    color_dest.val[1] = green;
                    color_dest.val[2] = red;
                    result.at<Vec3b>(Point(x,y)) = color_dest;

                }
            }
            return result;
    }

    Mat lighten(Mat a, Mat b)
    {
        Mat result = a.clone();
        Vec3b color_dest;
        for(int y = 0; y < a.rows; y++)
        {
            for (int x = 0; x < a.cols; x++)
            {
                // do it
                Vec3b intensity_a = a.at<Vec3b>(y, x);
                Vec3b intensity_b = b.at<Vec3b>(y, x);
                int blue_a = intensity_a.val[0];
                int green_a = intensity_a.val[1];
                int red_a = intensity_a.val[2];

                int blue_b = intensity_b.val[0];
                int green_b = intensity_b.val[1];
                int red_b = intensity_b.val[2];

                if((blue_a + green_a + red_a) > (blue_b + green_b + red_b))
                {
                    color_dest.val[0] = blue_a;
                    color_dest.val[1] = green_a;
                    color_dest.val[2] = red_a;
                }else
                {
                    color_dest.val[0] = blue_b;
                    color_dest.val[1] = green_b;
                    color_dest.val[2] = red_b;
                }

                result.at<Vec3b>(Point(x,y)) = color_dest;

            }
        }

        return result;
    }

    Mat AlphaBlend(Mat foreground, Mat background, Mat mask, Point position)
    {
        Mat a = foreground.clone();
        Mat b = background.clone();

        Mat img(background.size(), CV_8UC3, Scalar(0,0,0));  // foreground
        Mat maska(background.size(), CV_8UC3, Scalar(0,0,0)); // mask

        // insert a pozicování alfa obrázku
        mask.copyTo(maska(Rect(position.x,position.y,mask.cols,mask.rows)));
        a.copyTo(img(Rect(position.x,position.y,a.cols,a.rows)));

        // datové konverze
        img.convertTo(img,CV_32FC3);
        b.convertTo(b,CV_32FC3);
        maska.convertTo(maska,CV_32FC3,1.0/255);

        Mat ouImage = Mat::zeros(img.size(), img.type());
        multiply(maska, img, img);
        multiply(Scalar::all(1.0)-maska, b, b);
        add(img, b, ouImage);

        Mat x = ouImage;
        x.convertTo(x,CV_8UC3);
        return x;
}

    Mat mapping(Mat src, vector<Point2f> inputQuad, vector<Point2f> outputQuad )
    {
        Mat input = src.clone();
        Mat output;
        Mat lambda( 2, 4, CV_32FC1 );
        lambda = Mat::zeros( input.rows, input.cols, input.type() );
        lambda = getPerspectiveTransform( inputQuad, outputQuad );
        warpPerspective(input,output,lambda,output.size() );
        return output;
    }

    Mat AutoAlign(Mat src, Mat sablona, int number_of_iterations)
    {
        Mat im1 = sablona.clone();
        Mat im2 = src.clone();

        Mat im1_gray, im2_gray;
        cvtColor(im1, im1_gray, COLOR_BGR2GRAY);
        cvtColor(im2, im2_gray, COLOR_BGR2GRAY);

        const int warp_mode = MOTION_EUCLIDEAN;
        Mat warp_matrix;

        if ( warp_mode == MOTION_HOMOGRAPHY )
            warp_matrix = Mat::eye(3, 3, CV_32F);
        else
            warp_matrix = Mat::eye(2, 3, CV_32F);

        // Specify the threshold of the increment
        // in the correlation coefficient between two iterations
        double termination_eps = 1e-10;

        // Define termination criteria
        TermCriteria criteria (TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps);

        // Run the ECC algorithm. The results are stored in warp_matrix.
        findTransformECC(
                        im1_gray,
                        im2_gray,
                        warp_matrix,
                        warp_mode,
                        criteria
                    );

        Mat im2_aligned;
        if (warp_mode != MOTION_HOMOGRAPHY)
            // Use warpAffine for Translation, Euclidean and Affine
            warpAffine(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
        else
            // Use warpPerspective for Homography
            warpPerspective (im2, im2_aligned, warp_matrix, im1.size(),INTER_LINEAR + WARP_INVERSE_MAP);

        return im2_aligned;
    }


    string type2str(int type)
    {
        string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth )
        {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }

}






#endif // MOJECV_HPP_INCLUDED
