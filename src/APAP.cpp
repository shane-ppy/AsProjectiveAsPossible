#include "APAP.h"

using namespace cv;
using namespace Eigen;
using namespace std;

bool saveData;
bool displayResult;

void drawMatch(Mat &img, const MatrixXf &match, const Matrix3f &inv_T1, const Matrix3f &inv_T2) {
  int height = img.size[0];
  int width = img.size[1];
  int reference = height;
  if (width < height)
    reference = width;
  float circleRadius = reference/200.f;
  float strokeSize = reference/600.f;
  MatrixXf pts1 = (inv_T1*match.block(0, 0, match.rows(), 3).transpose()).transpose();
  MatrixXf pts2 = (inv_T2*match.block(0, 3, match.rows(), 3).transpose()).transpose();

  for (int i = 0 ; i < match.rows(); i++) {
    Point p1 = Point(pts1.row(i)[0], pts1.row(i)[1]);
    Point p2 = Point(pts2.row(i)[0]+img.size[1]/2, pts2.row(i)[1]);
    line(img, p1, p2, Scalar(0, 255, 0), strokeSize);
  }

  for (int i = 0 ; i < match.rows(); i++) {
    Point p1 = Point(pts1.row(i)[0], pts1.row(i)[1]);
    Point p2 = Point(pts2.row(i)[0]+img.size[1]/2, pts2.row(i)[1]);
    circle(img, p1, circleRadius, Scalar(0, 0, 255), strokeSize);
    circle(img, p2, circleRadius, Scalar(0, 0, 255), strokeSize);
  }
}

void warpAndFuseImage(const Mat &img1, const Mat &img2, const Matrix3f &H, int &offX, int &offY, int &cw, int &ch, Mat &linearFusion) {

  int height = img1.size[0];
  int width = img1.size[1];

  // Vector3f TL = H.inverse()*(VectorXf(3) << 0, 0, 1).finished();
  // Vector3f TR = H.inverse()*(VectorXf(3) << width-1, 0, 1).finished();
  // Vector3f BL = H.inverse()*(VectorXf(3) << 0, height-1, 1).finished();
  // Vector3f BR = H.inverse()*(VectorXf(3) << width-1, height-1, 1).finished();
  // noHomogeneous(TL);
  // noHomogeneous(TR);
  // noHomogeneous(BL);
  // noHomogeneous(BR);
  // vector<int> xSet(6);
  // vector<int> ySet(6);
  // xSet[0] = static_cast<int>(TL[0]);
  // xSet[1] = static_cast<int>(TR[0]);
  // xSet[2] = static_cast<int>(BL[0]);
  // xSet[3] = static_cast<int>(BR[0]);
  // xSet[4] = 0;
  // xSet[5] = width;
  // ySet[0] = static_cast<int>(TL[1]);
  // ySet[1] = static_cast<int>(TR[1]);
  // ySet[2] = static_cast<int>(BL[1]);
  // ySet[3] = static_cast<int>(BR[1]);
  // ySet[4] = 0;
  // ySet[5] = height;
  // int minX = minOfSet(xSet);
  // int maxX = maxOfSet(xSet);
  // int minY = minOfSet(ySet);
  // int maxY = maxOfSet(ySet);

  int minX = 0;
  int minY = 0;

  // offX = -minX;
  // offY = -minY;

  // cw = maxX - minX;
  // ch = maxY - minY;

  // cw = width;
  // ch = height;
  Mat warp_img1 = Mat::zeros(ch, cw, CV_8UC3);
  Mat warp_img2 = Mat::zeros(ch, cw, CV_8UC3);
  linearFusion = Mat::zeros(ch, cw, CV_8UC3); img1.copyTo(warp_img1(Rect_<int>(-minX, -minY, width, height))); uint8_t r, g, b;
  float h00 = H(0, 0);
  float h01 = H(0, 1);
  float h02 = H(0, 2);
  float h10 = H(1, 0);
  float h11 = H(1, 1);
  float h12 = H(1, 2);
  float h20 = H(2, 0);
  float h21 = H(2, 1);
  float h22 = H(2, 2);
  for (int i = 0; i < ch; i++)
    for (int j = 0; j < cw; j++) {
      // Eigen too slow
      float x = ((j+minX)*h00 + (i+minY)*h01+h02)/((j+minX)*h20+(i+minY)*h21+h22);
      float y = ((j+minX)*h10 + (i+minY)*h11+h12)/((j+minX)*h20+(i+minY)*h21+h22);
      if (x > 0 && x < width && y > 0 && y < height) {
        getColorSubpixelRGB(img2, x, y, width, height, r, g, b);
        warp_img2.at<Vec3b>(i, j) = Vec3b(r, g, b);
      }
    }
  for (int i = 0; i < ch; i++)
    for (int j = 0; j < cw; j++) {
      Vec3b img1Color = warp_img1.at<Vec3b>(i, j);
      Vec3b img2Color = warp_img2.at<Vec3b>(i, j);
      int leftWeight = 1;
      int rightWeight = 1;
      if (img1Color[0] < 10 && img1Color[1] < 10 && img1Color[2] < 10)
        leftWeight = 0;
      if (img2Color[0] < 10 && img2Color[1] < 10 && img2Color[2] < 10)
        rightWeight = 0;
      if (leftWeight + rightWeight != 0) {
        r = static_cast<uint8_t>(img1Color[0]*leftWeight/(leftWeight+rightWeight)+img2Color[0]*rightWeight/(leftWeight+rightWeight));
        g = static_cast<uint8_t>(img1Color[1]*leftWeight/(leftWeight+rightWeight)+img2Color[1]*rightWeight/(leftWeight+rightWeight));
        b = static_cast<uint8_t>(img1Color[2]*leftWeight/(leftWeight+rightWeight)+img2Color[2]*rightWeight/(leftWeight+rightWeight));
        linearFusion.at<Vec3b>(i, j) = Vec3b(r, g, b);
      }
    }
  if (displayResult)
    displayMat(linearFusion);
  if (saveData)
    imwrite("/home/shane/stitching/AsProjectiveAsPossible/build/Global.jpg", linearFusion);
}

void warpAndFuseImageAPAP(const Mat &img1, const Mat &img2, const MatrixXf &H, int offX, int offY, int cw, int ch, const ArrayXf &X, const ArrayXf &Y, Mat &linearFusion) {

  int height = img1.size[0];
  int width = img1.size[1];

  Mat warp_img1 = Mat::zeros(ch, cw, CV_8UC3);
  Mat warp_img2 = Mat::zeros(ch, cw, CV_8UC3);
  linearFusion = Mat::zeros(ch, cw, CV_8UC3); 
  img1.copyTo(warp_img1(Rect_<int>(offX, offY, width, height))); 
  uint8_t r, g, b;
  for (int i = 0; i < ch; i++)
    for (int j = 0; j < cw; j++) {

      // find grid coord for current pix value
      int xindex;
      int yindex;
      for (xindex = 0; (xindex+1) < X.rows() && j>= X[xindex]; xindex++);
      for (yindex = 0; (yindex+1) < Y.rows() && i>= Y[yindex]; yindex++);

      int Hindex = yindex*Y.rows()+xindex;
      // int Hindex = 0;
      float div = ((j-offX)*H(Hindex, 6)+(i-offY)*H(Hindex, 7)+H(Hindex, 8));
      float x = ((j-offX)*H(Hindex, 0) + (i-offY)*H(Hindex, 1)+H(Hindex, 2))/div;
      float y = ((j-offX)*H(Hindex, 3) + (i-offY)*H(Hindex, 4)+H(Hindex, 5))/div;
      if (x > 0 && x < width && y > 0 && y < height) {
        getColorSubpixelRGB(img2, x, y, width, height, r, g, b);
        warp_img2.at<Vec3b>(i, j) = Vec3b(r, g, b);
      }
    }
  for (int i = 0; i < ch; i++)
    for (int j = 0; j < cw; j++) {
      Vec3b img1Color = warp_img1.at<Vec3b>(i, j);
      Vec3b img2Color = warp_img2.at<Vec3b>(i, j);
      int leftWeight = 1;
      int rightWeight = 1;
      if (img1Color[0] < 10 && img1Color[1] < 10 && img1Color[2] < 10)
        leftWeight = 0;
      if (img2Color[0] < 10 && img2Color[1] < 10 && img2Color[2] < 10)
        rightWeight = 0;
      if (leftWeight + rightWeight != 0) {
        r = static_cast<uint8_t>(img1Color[0]*leftWeight/(leftWeight+rightWeight)+img2Color[0]*rightWeight/(leftWeight+rightWeight));
        g = static_cast<uint8_t>(img1Color[1]*leftWeight/(leftWeight+rightWeight)+img2Color[1]*rightWeight/(leftWeight+rightWeight));
        b = static_cast<uint8_t>(img1Color[2]*leftWeight/(leftWeight+rightWeight)+img2Color[2]*rightWeight/(leftWeight+rightWeight));
        linearFusion.at<Vec3b>(i, j) = Vec3b(r, g, b);
      }
    }
  if (displayResult)
    displayMat(linearFusion);
  if (saveData)
    imwrite("/home/shane/stitching/AsProjectiveAsPossible/build/APAP.jpg", linearFusion);
}

int GlobalHomography(MatrixXf &inlier, MatrixXf &A, Matrix3f &T1, Matrix3f &T2, int &offX, int &offY, int &cw, int &ch, Mat &img1, Mat &img2, Mat &img12H) {

  // int height = img1.size[0];
  // int width = img1.size[1]; 
  // MatrixXf match;
  
  // // detectSiftMatchWithOpenCV(img1, img2, match);
  // // detectSiftMatchWithSiftGPU(img1_path, img2_path, match);
  // // detectSiftMatchWithVLFeat(img1, img2, match);
  // // detectSiftMatchWithROCm(img1, img2, match);
  // detectORBMatchWithOpenCV(img1, img2, match);
  
  // normalizeMatch(match, T1, T2);

  // // cout << T1;

  // singleModelRANSAC(match, 2000, inlier);
  // // multiModelRANSAC(match, 500, inlier);
  // Matrix3f H;
  // fitHomography(inlier.block(0, 0, inlier.rows(), 3), inlier.block(0, 3, inlier.rows(), 3), H, A);

  // if (displayResult) {
  //   Mat display;
  //   combineMat(display, img1, img2);
  //   drawMatch(display, match, T1.inverse(), T2.inverse());
  //   displayMat(display);

  //   combineMat(display, img1, img2);
  //   drawMatch(display, inlier, T1.inverse(), T2.inverse());
  //   displayMat(display);
  // }

  // Matrix3f Hg = T2.inverse()*H*T1;
  Matrix3f Hg;
  Hg << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  warpAndFuseImage(img1, img2, Hg, offX, offY, cw, ch, img12H);
  return 0;
}

void APAP(const MatrixXf &inlier, const MatrixXf &A, const Matrix3f &T1, const Matrix3f &T2, int offX, int offY, int cw, int ch, const Mat &img1, const Mat &img2, Mat &img12) {

  // APAP params
  float gamma = 0.1;
  float sigma = 12.f;
  float sigmaSquared = sigma*sigma;
  // Grid parition
  int GW = 60;
  int GH = 60;
  ArrayXf X = ArrayXf::LinSpaced(GW, 0, cw);
  ArrayXf Y = ArrayXf::LinSpaced(GH, 0, ch);
  ArrayXf MvX = X - offX;
  ArrayXf MvY = Y - offY;

  MatrixXf Hmdlt = MatrixXf::Zero(GW*GH, 9);
  MatrixXf pts1 = (T1.inverse()*inlier.block(0, 0, inlier.rows(), 3).transpose()).transpose();
  // MatrixXf pts2 = (T2.inverse()*inlier.block(0, 3, inlier.rows(), 3).transpose()).transpose();
  Matrix3f inv_T2 = T2.inverse();

  #pragma omp parallel for shared(Hmdlt) num_threads(6)
  for (int i = 0; i < GW*GH - 1; i++) {
    MatrixXf Wi(pts1.rows()*2, pts1.rows()*2);
    Wi.setZero();
    float localX = MvX(i%GW);
    float localY = MvY(i/GW);
    
    for (int j = 0; j < pts1.rows(); j++) {
      float dist_weight = exp(-pdist2(localX, localY, pts1(j, 0), pts1(j, 1))/sigmaSquared);
      float weight = max(dist_weight, gamma);
      Wi(j*2, j*2) = weight;
      Wi(j*2+1, j*2+1) = weight;
    }

    // Solve for SVD
    // cout << (Wi * A).rows() << ", " << (Wi * A).cols() << endl;
    if (true) {
      JacobiSVD<MatrixXf, HouseholderQRPreconditioner> svd(Wi*A, ComputeFullV);
      MatrixXf V = svd.matrixV();
      VectorXf h = V.col(V.cols()-1);
      Hmdlt.row(i) = unrollMatrix3f(inv_T2*rollVector9f(h)*T1);
    } else {
      VectorXf Hg(9);
      Hg << 1, 0, 0, 0, 1, 0, 0, 0, 1;
      Hmdlt.row(i) = Hg;
    }
  }
  
  warpAndFuseImageAPAP(img1, img2, Hmdlt, offX, offY, cw, ch, X, Y, img12);
}

struct Data {
  float residue;
  size_t index;
  Data(float _residue, size_t _index):residue(_residue), index(_index){}
};

long* cnpy2ptr(std::string data_fname)
{
  cnpy::NpyArray npy_data = cnpy::npy_load(data_fname);
  // double* ptr = npy_data.data<double>();
  int data_row = npy_data.shape[0];
  int data_col = npy_data.shape[1];
  long* ptr = static_cast<long *>(malloc(data_row * data_col * sizeof(long)));
  memcpy(ptr, npy_data.data<long>(), data_row * data_col * sizeof(long));
  return ptr;
}

long *transform_i[3] = {
  cnpy2ptr("/home/shane/stitching/image_processing/transform_i_0.npy"),
  cnpy2ptr("/home/shane/stitching/image_processing/transform_i_2.npy"),
  cnpy2ptr("/home/shane/stitching/image_processing/transform_i_4.npy")
};

long *transform_j[3] = {
  cnpy2ptr("/home/shane/stitching/image_processing/transform_j_0.npy"),
  cnpy2ptr("/home/shane/stitching/image_processing/transform_j_2.npy"),
  cnpy2ptr("/home/shane/stitching/image_processing/transform_j_4.npy")
};

void transform(Mat &raw, Mat &img, int cam) {
  img = Mat(Size(860,860),CV_8UC3, Scalar(0, 0, 0));
  #pragma omp parallel for num_threads(6)
  for (int i = 0; i < 860; i++) {
    for (int j = 0; j < 860; j++) {
      int ind_i = (int) transform_i[cam][i * 860 + j];
      int ind_j = (int) transform_j[cam][i * 860 + j];
      Vec3b bgrPixel = raw.at<Vec3b>(ind_i, ind_j);
      img.at<Vec3b>(i, j) = bgrPixel;
    }
  }
}

int main() {
  MatrixXf inlier, A;
  Matrix3f T1, T2;
  Mat img1, img2, img3, img23H, img23A, img123H, img123A;
  Mat raw1, raw2, raw3;
  int offX = 0; 
  int offY = 0; 
  int cw = 860;
  int ch = 860;
  // int offX, offY, cw, ch;

  const char* img1_path = "/home/shane/0.jpg";
  const char* img2_path = "/home/shane/camera2_test.jpg";
  const char* img3_path = "/home/shane/camera4_test.jpg";
  displayResult = false;
  saveData = true;

  // raw1 = imread(img1_path);
  // raw2 = imread(img2_path);
  // raw3 = imread(img3_path);
  // transform(raw1, img1, 0);
  // transform(raw2, img2, 1);
  // transform(raw3, img3, 2);

  // GlobalHomography(inlier, A, T1, T2, offX, offY, cw, ch, img2, img3, img23H);
  // APAP(inlier, A, T1, T2, offX, offY, cw, ch, img2, img3, img23A);
  // GlobalHomography(inlier, A, T1, T2, offX, offY, cw, ch, img1, img23A, img123H);
  // APAP(inlier, A, T1, T2, offX, offY, cw, ch, img1, img23A, img123A);

  VideoCapture camera0("/home/shane/texture0.mp4"); 
  VideoCapture camera2("/home/shane/texture2.mp4"); 
  VideoCapture camera4("/home/shane/texture4.mp4"); 

  while (camera0.isOpened() && camera2.isOpened() && camera4.isOpened()) {
    if (!camera0.read(raw1)) break;
    if (!camera2.read(raw2)) break;
    if (!camera4.read(raw3)) break;

    transform(raw1, img1, 0);
    transform(raw2, img2, 1);
    transform(raw3, img3, 2);

    GlobalHomography(inlier, A, T1, T2, offX, offY, cw, ch, img2, img3, img23H);
    // APAP(inlier, A, T1, T2, offX, offY, cw, ch, img2, img3, img23A);
    GlobalHomography(inlier, A, T1, T2, offX, offY, cw, ch, img1, img23H, img123H);
    imshow("bev", img123H);
    char c = waitKey(1);
    if (c == 27) break;
  }

  // auto t1 = chrono::high_resolution_clock::now();
  
  // auto t2 = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  // cout << duration << endl;

  


  return 0;
}
