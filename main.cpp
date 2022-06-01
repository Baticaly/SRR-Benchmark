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

int keypointDetector () {

    string image1_path = samples::findFile("set3overlap/1.png");
    string image2_path = samples::findFile("set3overlap/2.png");

    Mat query = imread( image1_path, IMREAD_COLOR );
    Mat train = imread( image2_path, IMREAD_COLOR );

    std::vector<KeyPoint> kpsA, kpsB;
    Mat descriptorsA, descriptorsB;

    const int nfeatures = 500;
    const int edgeThreshold = 10;

    Ptr<FeatureDetector> detector = ORB::create(nfeatures, edgeThreshold);
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> BFmatcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect ( query,kpsA );
    detector->detect ( train,kpsB );

    descriptor->compute ( query, kpsA, descriptorsA );
    descriptor->compute ( train, kpsB, descriptorsB );

    Mat result;
    drawKeypoints( query, kpsA, result, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    //imshow("result1",result);

    vector<DMatch> matches;
    BFmatcher->match ( descriptorsA, descriptorsB, matches );

    // Distance Evaluation
    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < descriptorsA.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    std:vector< DMatch > good_matches;
    for (int i=0; i<descriptorsA.rows; i++)
    {
        if (matches[i].distance <= max (2*min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_match;
    Mat img_goodmatch;

    drawMatches(query, kpsA, train, kpsB, matches, img_match);
    drawMatches(query, kpsA, train, kpsB, good_matches, img_goodmatch);
    //imshow("Matches", img_goodmatch);

    // Jens Gustedt / Modern C
    // NOT TESTED

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( kpsA[ good_matches[i].queryIdx ].pt );
        scene.push_back( kpsB[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );

    //-- Get the corners from the image_1
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)query.cols, 0 );
    obj_corners[2] = Point2f( (float)query.cols, (float)query.rows );
    obj_corners[3] = Point2f( 0, (float)query.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners
    line( img_match, scene_corners[0] + Point2f((float)query.cols, 0), scene_corners[1] + Point2f((float)query.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_match, scene_corners[1] + Point2f((float)query.cols, 0), scene_corners[2] + Point2f((float)query.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_match, scene_corners[2] + Point2f((float)query.cols, 0), scene_corners[3] + Point2f((float)query.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_match, scene_corners[3] + Point2f((float)query.cols, 0), scene_corners[0] + Point2f((float)query.cols, 0), Scalar( 0, 255, 0), 4 );
    
    imshow("Good Matches & Object detection", img_goodmatch );
    imshow("img_match", img_match );

    return sizeof(img_goodmatch);

}

Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);

    int channels = I.channels();

    int nRows = I.rows;
    int nCols = I.cols * channels;

    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];
        }
    }
    return I;
}


int main( int argc, char** argv ) 
{   
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    //int matchSize, goodMatchSize = keypointDetector();

    string image1_path = samples::findFile("set3overlap/2.png");
    string image2_path = samples::findFile("set3overlap/1.png");

    Mat query = imread( image1_path, IMREAD_COLOR );
    Mat train = imread( image2_path, IMREAD_COLOR );

    std::vector<KeyPoint> kpsA, kpsB;
    Mat descriptorsA, descriptorsB;

    const int nfeatures = 500000;
    const int edgeThreshold = 10;

    Ptr<FeatureDetector> detector = ORB::create(nfeatures);
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> BFmatcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect ( query,kpsA );
    detector->detect ( train,kpsB );

    descriptor->compute ( query, kpsA, descriptorsA );
    descriptor->compute ( train, kpsB, descriptorsB );

    Mat result;
    drawKeypoints( query, kpsA, result, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    //imshow("result1",result);

    vector<DMatch> matches;
    BFmatcher->match ( descriptorsA, descriptorsB, matches );

    // Distance Evaluation
    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < descriptorsA.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    std:vector< DMatch > good_matches;
    for (int i=0; i<descriptorsA.rows; i++)
    {
        if (matches[i].distance <= max (2*min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }
    
    Mat img_match;
    Mat img_goodmatch;

    drawMatches(query, kpsA, train, kpsB, matches, img_match);
    drawMatches(query, kpsA, train, kpsB, good_matches, img_goodmatch);
    //imshow("Matches", img_goodmatch);

    // Jens Gustedt / Modern C
    // NOT TESTED

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( kpsA[ good_matches[i].queryIdx ].pt );
        scene.push_back( kpsB[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );

    cv::Mat final;
    warpPerspective(query,final,H,Size(query.cols+train.cols,query.rows));
    cv::Mat half(final,cv::Rect(0,0,train.cols,train.rows));
    train.copyTo(half);
    imshow( "Resultemp", final );

    /*
    width, height, _ = query.shape
        for h in range(0, height):
            for w in range(0, width):
                resultValue = result[h][w][0] + result[h][w][1] + result[h][w][2]
                queryValue = query_float[h][w][0] + query_float[h][w][1] + query_float[h][w][2]
                if queryValue != 0 and resultValue != 0:
                    result[h][w] = ( result[h][w] + query_float[h][w] ) / 2
                else:
                    result[h][w] = result[h][w] + query_float[h][w]

        result = result.astype(np.uint8)
    */

    //imshow("Good Matches & Object detection", img_goodmatch );
    imshow("train", train );

    int matchSize;
    matchSize = sizeof(img_match);

    int goodMatchSize;
    goodMatchSize = sizeof(img_goodmatch);

    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    ofstream logfile;
    logfile.open ("results.csv");
    logfile << "Test Index,Keypoint Match Count,Good Match Count,Result(second)\n";
    logfile << "0," << matchSize << "," << goodMatchSize << "," << chrono::duration_cast<chrono::microseconds>(end - begin).count() << endl;
    logfile.close();

    waitKey(0);
    return 0;
}

