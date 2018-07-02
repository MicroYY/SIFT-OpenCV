#include <opencv2/opencv.hpp>
//#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>






//#include<opencv2/features2d/features2d.hpp>
//#include<opencv2/xfeatures2d/nonfree.hpp>
//#include<opencv2/highgui/highgui.hpp>
//#include<iostream>

/**********
	命令行参数： 第一张输入图片
				第二张输入图片
				普通匹配结果 (距离小于最小距离*2即认为匹配）
				1st/2nd<0.8结果
				反向匹配
				1st/2nd<0.8+双向匹配结果
				1st/2nd<0.8+双向匹配+ransac
				设置sifi特征点数量


**********/



int main(int argc, char** argv)
{
	cv::Ptr<cv::DescriptorExtractor> sift = cv::xfeatures2d::SIFT::create(atoi(argv[8]));
	//cv::Feature2D = cv::DescriptorExtractor

	cv::Mat img1 = cv::imread(argv[1]);
	std::vector<cv::KeyPoint> keypoints1;
	cv::Mat descriptor1;
	sift->detectAndCompute(img1, cv::Mat(), keypoints1, descriptor1);


	cv::Mat img2 = cv::imread(argv[2]);
	std::vector<cv::KeyPoint> keypoints2;
	cv::Mat descriptor2;
	sift->detectAndCompute(img2, cv::Mat(), keypoints2, descriptor2);


	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	//暴力搜索 cv::BFMatcher



	/********************************************************************************/
	//最近邻
	//普通匹配 (距离小于最小距离*2即认为匹配）
	std::vector<cv::DMatch> match1;
	matcher->match(descriptor1, descriptor2, match1);

	double max = match1[0].distance;
	double min = match1[0].distance;
	for (int i = 1; i < descriptor1.rows; i++)
	{
		double dist = match1[i].distance;
		if (dist > max)
			max = dist;
		if (dist < min)
			min = dist;
	}

	std::vector<cv::DMatch> good_matches1;
	for (int i = 0; i < descriptor1.rows; i++)
	{
		if (match1[i].distance < 2 * min)
			good_matches1.push_back(match1[i]);
	}

#ifdef OUTPUT
	std::cout << "********** 1NN **********" << std::endl;
	for (int i = 0; i < good_matches1.size(); i++)
	{
		std::cout << "distance:" << good_matches1[i].distance << std::endl;
		std::cout << "image index:" << good_matches1[i].imgIdx << std::endl;
		std::cout << "query index:" << good_matches1[i].queryIdx << std::endl;
		std::cout << "train index:" << good_matches1[i].trainIdx << std::endl;
		std::cout << std::endl;
	}
#endif // OUTPUT


	cv::Mat match_img1;
	cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches1, match_img1);
	cv::imshow("", match_img1);
	cv::imwrite(argv[3], match_img1);



	/********************************************************************************/
	//knn
	//1st/2nd<0.8
	std::vector<std::vector<cv::DMatch>> match2;
	matcher->knnMatch(descriptor1, descriptor2, match2, 2);

	std::vector<cv::DMatch> good_matches2;
	for (int i = 0; i < match2.size(); i++)
	{
		if (match2[i][0].distance / match2[i][1].distance < 0.8)
			good_matches2.push_back(match2[i][0]);
	}
#ifdef OUTPUT
	std::cout << "********** KNN **********" << std::endl;
	for (int i = 0; i < match2.size(); i++)
	{
		for (int j = 0; j < match2[i].size(); j++)
		{
			std::cout << "第" << j << "个最近邻点" << std::endl;
			std::cout << "distance:" << match2[i][j].distance << std::endl;
			std::cout << "image index:" << match2[i][j].imgIdx << std::endl;
			std::cout << "query index:" << match2[i][j].queryIdx << std::endl;
			std::cout << "train index:" << match2[i][j].trainIdx << std::endl;
		}
		std::cout << std::endl;
	}
#endif // OUTPUT

	cv::Mat match_img2;
	cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches2, match_img2);
	cv::imshow("", match_img2);
	cv::imwrite(argv[4], match_img2);

	/********************************************************************************/
	//反向匹配
	std::vector<std::vector<cv::DMatch>> match3;
	matcher->knnMatch(descriptor2, descriptor1, match3, 2);

	std::vector<cv::DMatch> good_matches3;
	for (int i = 0; i < match3.size(); i++)
	{
		if (match3[i][0].distance / match3[i][1].distance < 0.8)
			good_matches3.push_back(match3[i][0]);
	}

	cv::Mat match_img3;
	cv::drawMatches(img2, keypoints2, img1, keypoints1, good_matches3, match_img3);
	cv::imshow("", match_img3);
	cv::imwrite(argv[5], match_img3);

	/********************************************************************************/
	//1st/2nd<0.8+双向匹配
	std::vector<cv::DMatch> two_way_matches_1_2;
	std::vector<cv::DMatch> two_way_matches_2_1;
	std::vector<int> matches_1_2(match2.size(), -1);
	std::vector<int> matches_2_1(match3.size(), -1);
	for (int i = 0; i < good_matches2.size(); i++)
	{
		int query_index = good_matches2[i].queryIdx;
		int train_index = good_matches2[i].trainIdx;
		matches_1_2[query_index] = train_index;
	}
	for (int i = 0; i < good_matches3.size(); i++)
	{
		int query_index = good_matches3[i].queryIdx;
		int train_index = good_matches3[i].trainIdx;
		matches_2_1[query_index] = train_index;
	}

	for (int i = 0; i < matches_1_2.size(); i++)
	{
		if (matches_1_2[i] < 0)
			continue;
		if (matches_2_1[matches_1_2[i]] != i)
			matches_1_2[i] = -1;
	}
	for (int i = 0; i < matches_2_1.size(); i++)
	{
		if (matches_2_1[i] < 0)
			continue;
		if (matches_1_2[matches_2_1[i]] != i)
			matches_2_1[i] = -1;
	}

	for (int i = 0; i < good_matches2.size(); i++)
	{
		int query_index = good_matches2[i].queryIdx;
		if (matches_1_2[query_index] >= 0)
			two_way_matches_1_2.push_back(good_matches2[i]);
	}
	for (int i = 0; i < good_matches3.size(); i++)
	{
		int query_index = good_matches3[i].queryIdx;
		if (matches_2_1[query_index] >= 0)
			two_way_matches_2_1.push_back(good_matches3[i]);
	}

	cv::Mat match_img4;
	cv::drawMatches(img1, keypoints1, img2, keypoints2, two_way_matches_1_2, match_img4);
	cv::imshow("", match_img4);
	cv::imwrite(argv[6], match_img4);

	/********************************************************************************/
	//1st/2nd<0.8+双向匹配+ransac
	std::vector<cv::DMatch> ransac_matches;
	std::vector<cv::Point2f> points1;
	std::vector<cv::Point2f> points2;
	for (int i = 0; i < two_way_matches_1_2.size(); i++)
	{
		points1.push_back(keypoints1[two_way_matches_1_2[i].queryIdx].pt);
		points2.push_back(keypoints2[two_way_matches_1_2[i].trainIdx].pt);
	}
	std::vector<uchar> inliersMask(points1.size());
	cv::findFundamentalMat(points1,points2, inliersMask,CV_FM_RANSAC);
	for (int i = 0; i < inliersMask.size(); i++)
	{
		if (inliersMask[i])
			ransac_matches.push_back(two_way_matches_1_2[i]);
	}

	cv::Mat match_img5;
	cv::drawMatches(img1, keypoints1, img2, keypoints2, ransac_matches, match_img5);
	cv::imshow("", match_img5);
	cv::imwrite(argv[7], match_img5);

	cv::waitKey();

}