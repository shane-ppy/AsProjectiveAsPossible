#include "CVUtility.h"
// #include "opencv2/xfeatures2d.hpp"
#include <vector>

using namespace cv;
using namespace Eigen;
using namespace std;

void getColorSubpixelRGB(const Mat &image, float x, float y, int width, int height, uint8_t &r, uint8_t &g, uint8_t &b)
{
  int x_int = (int)x;
  int y_int = (int)y;

  int x0 = x_int < 0 ? 0 : (x_int >= width ? width - 1 : x_int);
  int x1 = x_int + 1 < 0 ? 0 : (x_int + 1 >= width ? width - 1 : x_int + 1);
  int y0 = y_int < 0 ? 0 : (y_int >= height ? height - 1 : y_int);
  int y1 = y_int + 1 < 0 ? 0 : (y_int + 1 >= height ? height - 1 : y_int + 1);

  float dx = x - (float)x_int;
  float dy = y - (float)y_int;

  r = (1.f - dy) * (image.at<Vec3b>(y0, x0)[0] * (1.f - dx) + image.at<Vec3b>(y0, x1)[0] * dx) + dy * (image.at<Vec3b>(y1, x0)[0] * (1.f - dx) + image.at<Vec3b>(y1, x1)[0] * dx);
  g = (1.f - dy) * (image.at<Vec3b>(y0, x0)[1] * (1.f - dx) + image.at<Vec3b>(y0, x1)[1] * dx) + dy * (image.at<Vec3b>(y1, x0)[1] * (1.f - dx) + image.at<Vec3b>(y1, x1)[1] * dx);
  b = (1.f - dy) * (image.at<Vec3b>(y0, x0)[2] * (1.f - dx) + image.at<Vec3b>(y0, x1)[2] * dx) + dy * (image.at<Vec3b>(y1, x0)[2] * (1.f - dx) + image.at<Vec3b>(y1, x1)[2] * dx);
}

void combineMat(Mat &out, const Mat &left, const Mat &right)
{
  int height = left.size[0];
  int width = left.size[1];
  out = Mat(height, 2 * width, CV_8UC3);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width * 2; j++)
      if (j < width)
        out.at<Vec3b>(i, j) = left.at<Vec3b>(i, j);
      else
        out.at<Vec3b>(i, j) = right.at<Vec3b>(i, j - width);
}

void displayMat(const Mat &display)
{
  int height = display.size[0];
  int width = display.size[1];
  int longEdge = max(height, width);
  float resize_ratio = 1000.f / longEdge;
  Size size((int)width * resize_ratio, (int)height * resize_ratio);
  Mat resized;
  resize(display, resized, size);
  imshow("img", resized);
  waitKey(0);
}

float aa = 280 * 280, bb = 245 * 245;

bool filterkp(KeyPoint kp)
{
  float x = kp.pt.x - 430;
  float y_shift = 430 - kp.pt.y - 160;
  return x * x / aa + y_shift * y_shift / bb < 1;
}

bool filterxy(float xc, float yc)
{
  float x = xc - 430;
  float y_shift = 430 - yc - 160;
  return x * x / aa + y_shift * y_shift / bb < 1;
}

// void detectSiftMatchWithOpenCV(Mat &img1, Mat &img2, MatrixXf &match)
// {

//   cv::Ptr<cv::SIFT> detector = cv::SIFT::create(0, 3, 0.03, 10, 0.5);
//   vector<KeyPoint> key1;
//   vector<KeyPoint> key2;
//   Mat desc1, desc2;
//   Mat output;
  
//   detector->detectAndCompute(img1, noArray(), key1, desc1);
//   detector->detectAndCompute(img2, noArray(), key2, desc2);

//   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
//   std::vector<std::vector<DMatch>> knn_matches;
//   matcher->knnMatch(desc1, desc2, knn_matches, 2);
//   //-- Filter matches using the Lowe's ratio test
//   const float ratio_thresh = 0.7f;

//   std::vector<DMatch> good_matches;
//   for (size_t i = 0; i < knn_matches.size(); i++)
//   {
//     // good_matches.push_back(knn_matches[i][0]);
//     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
//     {
//       KeyPoint kp = key1[knn_matches[i][0].queryIdx];
//       if (true || filterkp(kp))
//       {
//         good_matches.push_back(knn_matches[i][0]);
//       }
//     }
//   }
//   // drawMatches(img1, key1, img2, key2, good_matches, output);
//   // imshow("img", output);
//   // waitKey(0);

//   // drawKeypoints(img1, key1, output, Scalar_<double>::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//   // imshow("img", output);
//   // waitKey(0);

//   match.resize(good_matches.size(), 6);
//   cout << "match count: " << good_matches.size() << endl;
//   for (int i = 0; i < good_matches.size(); i++)
//   {
//     match(i, 0) = key1[good_matches[i].queryIdx].pt.x;
//     match(i, 1) = key1[good_matches[i].queryIdx].pt.y;
//     match(i, 2) = 1;
//     match(i, 3) = key2[good_matches[i].trainIdx].pt.x;
//     match(i, 4) = key2[good_matches[i].trainIdx].pt.y;
//     match(i, 5) = 1;
//   }
// }
