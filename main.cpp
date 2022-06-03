#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//`pkg-config opencv4 --cflags --libs`

int main( int argc, char** argv ) 
{   
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    string image1_path = samples::findFile("set3overlap/1.png");
    string image2_path = samples::findFile("set3overlap/2.png");

    Mat query = imread(image2_path, IMREAD_COLOR);
    Mat train = imread(image1_path, IMREAD_COLOR);

    double xRender = train.cols + (2 * query.cols) + 50;
    double yRender = train.rows + (query.rows / 2);

    // Calculate center matrix
    double midX = (xRender - train.cols) / 2;
    double midY = (yRender - train.rows) / 2;

    Mat C = (Mat_<double>(2,3) << 1, 0, midX, 0, 1, midY);

    // Push train to center
    warpAffine(train, train, C, Size(xRender, yRender));
    warpAffine(query, query, C, Size(xRender, yRender));


    std::vector<KeyPoint> kpsA, kpsB;
    Mat descriptorsA, descriptorsB;

    const int nfeatures = 500000;
    const int edgeThreshold = 10;

    Ptr<FeatureDetector> detector = ORB::create(nfeatures);
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> BFmatcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(query, kpsA);
    detector->detect(train, kpsB);

    descriptor->compute(query, kpsA, descriptorsA);
    descriptor->compute(train, kpsB, descriptorsB);

    Mat result;
    drawKeypoints(query, kpsA, result, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    vector<DMatch> matches;
    BFmatcher->match (descriptorsA, descriptorsB, matches);

    double min_dist=10000, max_dist=0;

    for (int i = 0; i < descriptorsA.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    std:vector<DMatch> good_matches;
    for (int i=0; i<descriptorsA.rows; i++){
        if (matches[i].distance <= max(2*min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }
    
    Mat img_match;
    Mat img_goodmatch;

    //drawMatches(query, kpsA, train, kpsB, matches, img_match);
    drawMatches(query, kpsA, train, kpsB, good_matches, img_goodmatch);

    std::vector<Point2f> queryVector;
    std::vector<Point2f> trainVector;

    for( size_t i = 0; i < good_matches.size(); i++ ){
        queryVector.push_back(kpsA[good_matches[i].queryIdx].pt);
        trainVector.push_back(kpsB[good_matches[i].trainIdx].pt);
    }

    Mat H = findHomography(queryVector, trainVector, RANSAC);

    /*
   [1.008416200554807, 0.0001126243523051494, -630.2860380694477;
    0.003563377880238407, 1.009041866561184, -2.453838492536256;
    1.265658829422426e-05, 4.221225619312872e-07, 1]
    */

    /*
    // Calculate total offset
    double xOffset = H.at<double>(0,2);
    double yOffset = H.at<double>(1,2);

    double xRender = train.cols + query.cols + abs(xOffset) + 50;
    double yRender = train.rows + query.rows + abs(yOffset) + 50;

    // Calculate center matrix
    double midX = (xRender - train.cols) / 2;
    double midY = (yRender - train.rows) / 2;

    Mat C = (Mat_<double>(2,3) << 1, 0, midX, 0, 1, midY);

    // Push train to center
    warpAffine(train, train, C, Size(xRender, yRender));
    warpAffine(query, query, C, Size(xRender, yRender));
    
    // Multiply H matrix with the same center matrix
    //H.at<double>(0,2) = H.at<double>(0,2) + midX;
    //H.at<double>(1,2) = H.at<double>(1,2) + midY;
    */

    Mat final;
    warpPerspective(query, final, H, Size(xRender, yRender));

    // Interpolation
    int fR, fG, fB, tR, tG, tB;
    for(int y=0; y<train.rows; y++){
        for(int x=0; x<train.cols; x++){

            Vec3b & finalValue = final.at<Vec3b>(y,x);
            Vec3b & trainValue = train.at<Vec3b>(y,x);
            
            fR = finalValue[0]; fG = finalValue[1]; fB = finalValue[2]; 
            tR = trainValue[0]; tG = trainValue[1]; tB = trainValue[2]; 
            
            int finalSum = fR + fG + fB;
            int trainSum = tR + tG + tB;
            
            if( finalSum != 0 && trainSum != 0 ){
                finalValue[0] = (fR + tR) / 2;
                finalValue[1] = (fG + tG) / 2;
                finalValue[2] = (fB + tB) / 2;
            }
            else{
                finalValue[0] = (fR + tR);
                finalValue[1] = (fG + tG);
                finalValue[2] = (fB + tB);
            }

            final.at<Vec3b>(Point(x,y)) = finalValue;
        }
    }

    //train.copyTo(final);   # No interpolation, simple stacking

    //imshow("Good Matches & Object detection", img_goodmatch);
    imshow("Final", final);

    int matchSize;
    matchSize = sizeof(img_match);

    int goodMatchSize;
    goodMatchSize = sizeof(img_goodmatch);

    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    ofstream logfile;
    logfile.open("results.csv");
    logfile << "Test Index,Keypoint Match Count,Good Match Count,Result(second)\n";
    logfile << "0," << matchSize << "," << goodMatchSize << "," << chrono::duration_cast<chrono::microseconds>(end - begin).count() << endl;
    logfile.close();

    waitKey(0);
    return 0;
}

