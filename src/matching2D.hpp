#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#define YELLOW  "\033[33m"
#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis=false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis=false);;
void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis=false);
void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis=false);
void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis=false);
void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis=false);
void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis=false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis=false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);
void drawKeyPts(cv::Mat img, std::vector<cv::KeyPoint> kpts_, std::string keypoint_type_);
void descriptorDistanceRatio(std::vector<std::vector<cv::DMatch> > knn_matches_,  std::vector<cv::DMatch> &matches, float ratio);
#endif /* matching2D_hpp */
