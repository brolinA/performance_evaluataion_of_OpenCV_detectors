/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
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
#include <boost/circular_buffer.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
   if(argc < 3)
   {
    cout << RED <<"Invalid number of arguments \n Usage: /2D_feature_tracking <detectorType> <descriptorType>" << endl;
    return 0;
   }

    string dataPath = "../";
    int avg_matches=0;
    float total_time=0.0;
    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize);
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        cout << "IMAGE Number: " << imgIndex << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = argv[1];
        double kp_time;

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, kp_time, false);
        }
        else if(detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, kp_time, false);
        }
        else if(detectorType.compare("BRISK") == 0)
        {
            detKeypointsBRISK(keypoints, imgGray, kp_time, false);
        }
        else if(detectorType.compare("FAST") == 0)
        {
            detKeypointsFAST(keypoints, imgGray, kp_time, false);
        }
        else if(detectorType.compare("ORB") == 0)
        {
            detKeypointsORB(keypoints, imgGray, kp_time, false);
        }
        else if(detectorType.compare("AKAZE") == 0)
        {
            detKeypointsAKAZE(keypoints, imgGray, kp_time, false);
        }
        else if(detectorType.compare("SIFT") == 0)
        {
            detKeypointsSIFT(keypoints, imgGray, kp_time, false);
        }
       total_time += kp_time;

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        cv::Rect myROI(550, 190, 165, 150);
        if (bFocusOnVehicle)
        {
            std::vector<cv::KeyPoint> temp_kp = keypoints; //copy values to a temproary variable
            keypoints.clear(); //clear the keypoint vector
            float kp_size = 0.0;

            //check and add use only keypoints in frontvehicle
            for (std::vector<cv::KeyPoint>::iterator i = temp_kp.begin(); i != temp_kp.end(); ++i)
            {
                /*if(vehicleRect.contains(i->pt))
                {
                    keypoints.push_back(*i);
                }*/

                //implementing own version of cv::Rect::contain method
                float col_ = i->pt.x;
                float row_ = i->pt.y;

                if( col_ >= vehicleRect.x && col_ < vehicleRect.x+vehicleRect.width &&
                    row_ >= vehicleRect.y && row_ < vehicleRect.y+vehicleRect.height)
                {

                    keypoints.push_back(*i);
                    kp_size += i->size;
                }
            }
            double mean = kp_size/keypoints.size();

            cout << detectorType <<" keypoints have been resticted " << keypoints.size() << endl << "Average keypoint size: " << mean <<endl;

            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::Mat crop = visImage(myROI);
            string windowName = "cropped image";
            cv::namedWindow(windowName, 7);
            cv::moveWindow(windowName, 100, 100);
            cv::imshow(windowName, crop);
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        // cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        
        cv::Mat descriptors;
        string descriptorType = argv[2]; // BRIEF, ORB, FREAK, AKAZE, SIFT

        //Sanity check to make sure that the keypoints detected is compatible with the desriptor requested for.
        if(detectorType.compare("ORB") == 0 && descriptorType.compare("AKAZE") == 0)
        {
            cout << YELLOW << descriptorType << " is not compatible with " << detectorType << ". So changing descriptor to ORB" << RESET << endl;
            descriptorType = "ORB";
        }
         if(detectorType.compare("AKAZE") == 0 && descriptorType.compare("AKAZE") != 0)
        {
            cout << YELLOW << descriptorType << " is not compatible with " << detectorType << ". So changing descriptor to AKAZE" <<  RESET << endl;
            descriptorType = "AKAZE";
        }
        else if(detectorType.compare("HARRIS") == 0 && descriptorType.compare("SIFT") == 0)
        {
            cout << YELLOW << descriptorType << " is not compatible with " << detectorType << ". So changing descriptor to BRIEF" <<  RESET << endl;
            descriptorType = "BRIEF";
        }
        else if(detectorType.compare("FAST") == 0 && descriptorType.compare("AKAZE") == 0)
        {
            cout << YELLOW << descriptorType << " is not compatible with " << detectorType << ". So changing descriptor to BRIEF" <<  RESET << endl;
            descriptorType = "BRIEF";
        }
        else if(detectorType.compare("SIFT") == 0 && (descriptorType.compare("ORB") == 0 || descriptorType.compare("AKAZE") == 0) )
        {
            cout << YELLOW << descriptorType << " is not compatible with " << detectorType << ". So changing descriptor to SIFT" <<  RESET << endl;
            descriptorType = "SIFT";
        }

        double desc_time = (double)cv::getTickCount();
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        desc_time = ((double)cv::getTickCount() - desc_time) / cv::getTickFrequency();
        total_time += (1000 * desc_time / 1.0);
        
        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        //cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
            string descriptorType_ = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            if(descriptorType.compare("SIFT") == 0)
                descriptorType_ = "DES_HOG";

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType_, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
            cout << "Number of matches: "<< matches.size() << endl;
            avg_matches += matches.size();
            //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    cout << "------------------------------------- " << endl;

    } // eof loop over all images

    cout << GREEN << "Avergae matches: " << (int)(avg_matches/10) << RESET << endl;
    cout << GREEN << "Avergae Times: " << (total_time/20) << RESET << endl;

    return 0;
}