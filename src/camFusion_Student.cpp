
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, int imgIdx, bool save)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    if (save){
        cv::imwrite("../Lidar_top_view_images/"+to_string(imgIdx)+".png", topviewImg);
    }
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
// current bounding box, prevFrame keypoints, currFrame keypoints, key point matches
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    vector<double> kptMatchesDistance;
    for (cv::DMatch &match: kptMatches){
        kptMatchesDistance.push_back(cv::norm(kptsPrev[match.queryIdx].pt - kptsCurr[match.trainIdx].pt));
    }
    sort(kptMatchesDistance.begin(), kptMatchesDistance.end());
    size_t size = kptMatchesDistance.size();
    double medianKptsDist;
    if (size % 2 == 0)
    {
      medianKptsDist = (kptMatchesDistance[size / 2 - 1] + kptMatchesDistance[size / 2]) / 2;
    }
    else 
    {
      medianKptsDist = kptMatchesDistance[size / 2];
    }
    // querIdx  -> prevFrame
    // trainIdx -> currFrame
    for (cv::DMatch &match: kptMatches){
        // check if the keypoint if within the current bounding box and if the distance between the matched keypoints is not too big
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt) && (cv::norm(kptsPrev[match.queryIdx].pt - kptsCurr[match.trainIdx].pt) <= medianKptsDist)){
            boundingBox.keypoints.push_back(kptsCurr[match.trainIdx]);
            boundingBox.kptMatches.push_back(match);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrevPrev, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> currKptMatches, std::vector<cv::DMatch> prevKptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    double minDist = 10.0; // min. required distance
    // in order to establish matches between three images, we need to check if there exists a match between prevFrame and prevprevFrame
    // since this search needs to be done repeatedly, we can instead create a look up table to speed up the process
    unordered_set<int> prevFrameKptsMatchIds;
    for (auto it = currKptMatches.begin(); it != currKptMatches.end() - 1; ++it)
    {
        prevFrameKptsMatchIds.insert(it->queryIdx);
    }

    map<int, int> indexForPrevAndPrevPrevFrame;
    int position(0);
    for (auto it = prevKptMatches.begin(); it != prevKptMatches.end() - 1; ++it)
    {
        if (prevFrameKptsMatchIds.find(it->trainIdx) != prevFrameKptsMatchIds.end()){
            indexForPrevAndPrevPrevFrame[it->trainIdx] = position;
        }
        position++;
    }

    for (auto it1 = currKptMatches.begin(); it1 != currKptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr     = kptsCurr[it1->trainIdx];
        cv::KeyPoint kpOuterPrev     = kptsPrev[it1->queryIdx];
        cv::KeyPoint kpOuterPrevPrev;
        if (indexForPrevAndPrevPrevFrame.count(it1->queryIdx) > 0)
        {
            kpOuterPrevPrev = kptsPrevPrev[(prevKptMatches.begin() + indexForPrevAndPrevPrevFrame[it1->queryIdx])->queryIdx];
        }
        else
        {
            continue;
        }
        for (auto it2 = currKptMatches.begin() + 1; it2 != currKptMatches.end(); ++it2)
        { // inner keypoint loop
            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr[it2->trainIdx];
            cv::KeyPoint kpInnerPrev = kptsPrev[it2->queryIdx];
            cv::KeyPoint kpInnerPrevPrev;
            if (indexForPrevAndPrevPrevFrame.count(it2->queryIdx) > 0)
            {
                kpInnerPrevPrev = kptsPrevPrev[(prevKptMatches.begin() + indexForPrevAndPrevPrevFrame[it2->queryIdx])->queryIdx];
            }
            else
            {
                continue;
            }

            // compute distances and distance ratios
            double h2 = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double h1 = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            double h0 = cv::norm(kpOuterPrevPrev.pt - kpInnerPrevPrev.pt);
            
            if (h0 > std::numeric_limits<double>::epsilon() && h1 > std::numeric_limits<double>::epsilon() && h0 != h2)
            { // avoid division by zero

                double distRatio = (h1 * (h0 - h2))/(h0 * h2);
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    // double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    size_t size = distRatios.size();
    double medianDistRatio;

    sort(distRatios.begin(), distRatios.end());
    if (size % 2 == 0)
    {
      medianDistRatio = (distRatios[size / 2 - 1] + distRatios[size / 2]) / 2;
    }
    else 
    {
      medianDistRatio = distRatios[size / 2];
    }
    double dT = 1 / frameRate;
    // when medianDistRatio > 1 ====> that the preceding vehicle has moved closer to the ego vehicle
    // when medianDistRatio < 1 ====> that the preceding vehicle has moved farther from the ego vehicle
    // hence we can take the absolute value of TTC 
    TTC = abs(2 * dT / medianDistRatio);
}

// clustering helper functions
void proximity(const vector<LidarPoint> &pts, vector<int> &cluster, vector<bool> &Pstatus, KdTree<LidarPoint> *tree, uint id, float *tol)
{
	Pstatus[id] = true;
	cluster.push_back(id);
	vector<int> nearbyPoints = tree->search(pts[id], *tol);
	for (int &nearbyPtId: nearbyPoints){
		if (!Pstatus[nearbyPtId]){
			proximity(pts, cluster, Pstatus, tree, nearbyPtId, tol);
		}
	}
}

vector<vector<int>> euclideanCluster(const vector<LidarPoint> &points, KdTree<LidarPoint>  *tree, float distanceTol)
{
    int minClusterSize = 10;
	vector<vector<int>> clusters;
	vector<bool> processedStatus(points.size(), false);
	for (uint id = 0; id < points.size(); id++)
	{
		if (!processedStatus[id])
		{
			vector<int> cluster;
			proximity(points, cluster, processedStatus, tree, id, &distanceTol);
			// reject small clusters
            if (cluster.size() > minClusterSize){
                clusters.emplace_back(cluster);
            }
		}
	}
	return clusters;
}

void computeTTCLidar(vector<LidarPoint> &lidarPointsPrev,
                     vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, bool log)
{
    LidarPoint minPrev, minCurr;
    float distanceTol = 1.5;
    {
        KdTree<LidarPoint>* tree = new KdTree<LidarPoint>(3); // pass the dimension information
    
        for (uint i=0; i<lidarPointsPrev.size(); i++) 
            tree->insert(lidarPointsPrev[i],i); 

        std::vector<std::vector<int>> clusters = euclideanCluster(lidarPointsPrev, tree, distanceTol);
        // make a new vector of lidarpoints only if some outliers have been found and the clusters returned exclude them
        if (clusters[0].size() != lidarPointsPrev.size()){
            cout << "######### " <<  clusters[0].size() - lidarPointsPrev.size() << " Outliers removed #########" << endl;
            vector<LidarPoint>  tmpLidarPoints;
            for (int idx: clusters[0]){
                tmpLidarPoints.push_back(lidarPointsPrev[idx]);
            }
            minPrev = *std::min_element(tmpLidarPoints.begin(), tmpLidarPoints.end(), [](LidarPoint &pt1, LidarPoint &pt2){return pt1.x < pt2.x;});    
        }
        else{
            minPrev = *std::min_element(lidarPointsPrev.begin(), lidarPointsPrev.end(), [](LidarPoint &pt1, LidarPoint &pt2){return pt1.x < pt2.x;});    
        }
    }
    {
        KdTree<LidarPoint>* tree = new KdTree<LidarPoint>(3); // pass the dimension information
    
        for (uint i=0; i<lidarPointsCurr.size(); i++) 
            tree->insert(lidarPointsCurr[i],i); 

        std::vector<std::vector<int>> clusters = euclideanCluster(lidarPointsCurr, tree, distanceTol);
        // make a new vector of lidarpoints only if some outliers have been found and the clusters returned exclude them
        if (clusters[0].size() != lidarPointsCurr.size()){
            cout << "######### " <<  clusters[0].size() - lidarPointsCurr.size() << " Outliers removed #########" << endl;
            vector<LidarPoint>  tmpLidarPoints;
            for (int idx: clusters[0]){
                tmpLidarPoints.push_back(lidarPointsCurr[idx]);
            }
            minCurr = *std::min_element(tmpLidarPoints.begin(), tmpLidarPoints.end(), [](LidarPoint &pt1, LidarPoint &pt2){return pt1.x < pt2.x;});    
        }
        else{
            minCurr = *std::min_element(lidarPointsCurr.begin(), lidarPointsCurr.end(), [](LidarPoint &pt1, LidarPoint &pt2){return pt1.x < pt2.x;});
        }
    }
    // compute the TTC
    if (log){
        cout << "#################################" << endl;
        cout << "##### prevFrame X-min: " << minPrev.x << endl;
        cout << "##### currFrame X-min: " << minCurr.x << endl;
        cout << "#################################" << endl;
    }
    TTC = minCurr.x / (abs(minCurr.x - minPrev.x) * frameRate);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    vector<int> prevIds, currIds;
    for (BoundingBox &bxp: prevFrame.boundingBoxes){
        prevIds.push_back(bxp.boxID);
    }
    for (BoundingBox &bxc: currFrame.boundingBoxes){
        currIds.push_back(bxc.boxID);
    }
    cv::Mat votingMatrix(prevIds.size(), currIds.size(), CV_8UC1, 0.0);

    // loop over matches
    for (cv::DMatch &match: matches){
        // for every match, find the box in prevFrame to which it belongs
        for (BoundingBox &bxp: prevFrame.boundingBoxes){
            if (bxp.roi.contains(prevFrame.keypoints[match.queryIdx].pt)){
                // for every match, find the box in currFrame to which it belongs
                for (BoundingBox &bxc: currFrame.boundingBoxes){
                    if (bxc.roi.contains(currFrame.keypoints[match.trainIdx].pt)){
                        if (bbBestMatches.count(bxp.boxID) == 0){
                            votingMatrix.at<u_char>(bxp.boxID, bxc.boxID)++; 
                            // bbBestMatches[bxp.boxID] = bxc.boxID;
                            // continue;
                        }
                    }
                }
                // continue;
            }
        }
    }
    // Example of voting matrix
    // [ 86,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0;
    //    1,  26,   0,   0,  63,   0,   0,   0,   0,   0,   0;
    //    0,   0,   0,   0,   0,  22,   5,   0,  13,   0,   0;
    //   29,   0,   0,   0,   0,   0,   0,   1,   0,   3,   0;
    //    0,  68,   3,   0,   0,   0,   0,   0,   0,   0,   0;
    //    0,   0,   0,   0,   0,   0,   0,   0,   3,   0,   4]
    // cout << votingMatrix << endl;

    // Now we need to pick the pair i.e. for a row i.e. for a particular bounding box in prevFrame
    // a column with the highest votes which denotes the bounding box in the currFrame
    for (size_t r = 0; r < votingMatrix.rows; r++){
        int colWithMaxVotes = 0;
        int maxVotes = 0;
        for (size_t c = 0; c < votingMatrix.cols; c++){
            if (maxVotes < votingMatrix.at<u_char>(r, c)){
                maxVotes = votingMatrix.at<u_char>(r, c);
                colWithMaxVotes = c;
            }
        }
        if (maxVotes != 0){
            bbBestMatches[r] = colWithMaxVotes; // keys are previous frame bounding box ids and values are current frame bounding box ids
        }
    }
}
