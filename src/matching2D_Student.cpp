#include <numeric>
#include "matching2D.hpp"

using namespace std;

void descriptorDistanceRatio(vector<vector<cv::DMatch> > knn_matches_,  vector<cv::DMatch> &matches, float ratio)
{
    for (auto it = knn_matches_.begin(); it != knn_matches_.end(); ++it)
        {

            if ((*it)[0].distance < ratio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << GREEN << "# keypoints removed = " << knn_matches_.size() - matches.size() << RESET << endl;
}
// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        // int normType = (descriptorType.compare("SIFT") == 0)? cv::NORM_L2 : cv::NORM_HAMMING;
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
     if (descSource.type() != CV_32F || descRef.type() != CV_32F )
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        cout <<"Before source type: " << descSource.type() << " reference: " <<  descRef.type() << endl;

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
        descriptorDistanceRatio(knn_matches, matches, 0.8);
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        int descLength_ = 32;
        bool use_orientation_ = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(descLength_, use_orientation_);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int maxKeypoints_ = 500;
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {        
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {        
        extractor = cv::SIFT::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {        
        extractor =  cv::xfeatures2d::FREAK::create();
    }
    else
    {
        cout << descriptorType << ": NO Such desciptor is available."<< endl;
        return;
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << GREEN << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << RESET << endl;
    time_ = (1000 * t / 1.0);
    if (bVis)
        drawKeyPts(img, keypoints, "Shi-Tomasi");
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    //std::vector<cv::KeyPoint> keypoints;
    int threshold = 100;
    double t = (double)cv::getTickCount();

    for (int i = 0; i < dst_norm.rows; ++i)
    {
        for (int j = 0; j < dst_norm.cols; ++j)
        {
            bool new_keypoint_processed{false};
            if (static_cast<int>(dst_norm.at<float>(i, j) > threshold))
            {
                cv::KeyPoint new_keypoint{};
                new_keypoint.pt = cv::Point2f(j, i );
                new_keypoint.size = blockSize;
                new_keypoint.response = dst_norm.at<float>(i, j);
                
                for (auto& keypoint : keypoints)
                {
                    if (0.0 < cv::KeyPoint::overlap(new_keypoint, keypoint))
                    {
                        new_keypoint_processed = true;
                        if (new_keypoint.response > keypoint.response)
                        {
                            keypoint = new_keypoint;
                        }
                    }
                }
                if (!new_keypoint_processed)
                {
                    keypoints.push_back(new_keypoint);
                }
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time_ = (1000 * t / 1.0);
    cout << GREEN << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << RESET << endl;
    
    if (bVis)
        drawKeyPts(img, keypoints, "HARRIS");

}

void detKeypointsBRISK(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis)
{
    auto brisk_detector = cv::BRISK::create();
    
    double brisk_time = (double)cv::getTickCount();
    brisk_detector->detect(img, keypoints);
    brisk_time = ((double)cv::getTickCount() - brisk_time) / cv::getTickFrequency();
    time_ = (1000 * brisk_time / 1.0);
    cout << GREEN << "BRISK detector with n= " << keypoints.size() << " keypoints in " << 1000 * brisk_time / 1.0 << " ms" << RESET << endl;

    if (bVis)
        drawKeyPts(img, keypoints, "BRISK");
}

void detKeypointsFAST(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis)
{
    auto fast_detector = cv::FastFeatureDetector::create(75);

    double fast_time = (double)cv::getTickCount();
    fast_detector->detect(img, keypoints);
    fast_time = ((double)cv::getTickCount() - fast_time) / cv::getTickFrequency();
    time_ = (1000 * fast_time / 1.0);
    cout << GREEN << "FAST detector with n= " << keypoints.size() << " keypoints in " << 1000 * fast_time / 1.0 << " ms" << RESET << endl;

    if (bVis)
        drawKeyPts(img, keypoints, "FAST");
}

void detKeypointsORB(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis)
{
    int maxKeypoints = 500;

    auto ORB_detector = cv::ORB::create(maxKeypoints);

    double orb_time = (double)cv::getTickCount();
    ORB_detector->detect(img, keypoints);
    orb_time = ((double)cv::getTickCount() - orb_time) / cv::getTickFrequency();
    time_ = (1000 * orb_time / 1.0);
    cout << GREEN << "ORB detector with n= " << keypoints.size() << " keypoints in " << 1000 * orb_time / 1.0 << " ms" << RESET << endl;

    if (bVis)
        drawKeyPts(img, keypoints, "ORB");
}

void detKeypointsSIFT(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis)
{
    int maxKeypoints = 500;

    auto SIFT_detector = cv::SIFT::create(maxKeypoints);

    double sift_time = (double)cv::getTickCount();
    SIFT_detector->detect(img, keypoints);
    sift_time = ((double)cv::getTickCount() - sift_time) / cv::getTickFrequency();
    time_ = (1000 * sift_time / 1.0);
    cout << GREEN << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * sift_time / 1.0 << " ms" << RESET << endl;

    if (bVis)
        drawKeyPts(img, keypoints, "SIFT");
}

void detKeypointsAKAZE(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time_, bool bVis)
{
    auto AKAZE_detector = cv::AKAZE::create();

    double akaze_time = (double)cv::getTickCount();
    AKAZE_detector->detect(img, keypoints);
    akaze_time = ((double)cv::getTickCount() - akaze_time) / cv::getTickFrequency();
    time_ = (1000 * akaze_time / 1.0);
    cout << GREEN << "AKAZE detector with n= " << keypoints.size() << " keypoints in " << 1000 * akaze_time / 1.0 << " ms" << RESET << endl;

    if (bVis)
        drawKeyPts(img, keypoints, "AKAZE");
}

void drawKeyPts(cv::Mat img, std::vector<cv::KeyPoint> kpts_, std::string keypoint_type_)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kpts_, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = keypoint_type_+" Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}