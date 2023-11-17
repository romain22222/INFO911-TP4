#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


struct ColorDistribution {
  float data[ 512 ]; // l'histogramme
  int nb;                     // le nombre d'échantillons
    
  ColorDistribution() { reset(); }
  ColorDistribution& operator=( const ColorDistribution& other ) = default;
  // Met à zéro l'histogramme    
  void reset();
  // Ajoute l'échantillon color à l'histogramme:
  // met +1 dans la bonne case de l'histogramme et augmente le nb d'échantillons
  void add( Vec3b color ); 
  // Indique qu'on a fini de mettre les échantillons:
  // divise chaque valeur du tableau par le nombre d'échantillons
  // pour que case représente la proportion des picels qui ont cette couleur.
  void finished();
  // Retourne la distance entre cet histogramme et l'histogramme other
  float distance( const ColorDistribution& other ) const;

  float at(int i, int j, int k) const;
  float& at(int i, int j, int k);
};

ColorDistribution
getColorDistribution( Mat input, Point pt1, Point pt2 )
{
  ColorDistribution cd;
  for ( int y = pt1.y; y < pt2.y; y++ )
    for ( int x = pt1.x; x < pt2.x; x++ )
      cd.add( input.at<Vec3b>( y, x ) );
  cd.finished();
  return cd;
}

float
ColorDistribution::at(int i, int j, int k) const{
  return this->data[i*64 + j*8 + k ];
}

float&
ColorDistribution::at(int i, int j, int k) {
  return this->data[i*64 + j*8 + k];
}

float
ColorDistribution::distance(const ColorDistribution& other) const{
  float dist = 0;
  for (int i = 0; i < 512; i++){
        float sub = this->data[i] + other.data[i];
        if (sub == 0 ) continue;
        dist+= (this->data[i] - other.data[i]) * (this->data[i] - other.data[i]) / sub;
  }
  return dist;
}

void
ColorDistribution::reset(){
  for(int i = 0; i < 512; i++){
        this->data[i] = 0;
  }
  this->nb = 0;
}

void
ColorDistribution::add(Vec3b color){
  int i = color[0] / 32;
  int j = color[1] / 32;
  int k = color[2] / 32;
  this->at(i,j,k) += 1;
  this->nb += 1;
}

void
ColorDistribution::finished(){
  for(int i = 0; i < 512; i++){
        this->data[i] /= this->nb;
  }
}

float
minDistance(const ColorDistribution& h, const std::vector<ColorDistribution>& hists){
  float min = h.distance(hists[0]);
  for (int i = 1; i < hists.size(); i++){
    float dist = h.distance(hists[i]);
    if (dist < min) min = dist;
  }
  return min;
}

float
moyenneDistance(const ColorDistribution& h, const std::vector<ColorDistribution>& hists){
  float moy = 0;
  for (int i = 0; i < hists.size(); i++){
    moy += h.distance(hists[i]);
  }
  return moy / hists.size();
}

Mat
recoObject(
  Mat input, 
  const std::vector<std::vector<ColorDistribution>>& all_col_hists,
  const::vector<Vec3b>& colors, 
  const int bloc){
  Mat output = input.clone();
  const int width = input.cols;
  const int height = input.rows;
  for (int y = 0; y < height; y+=bloc){
    for (int x = 0; x < width; x+=bloc){
      Point pt1(x,y);
      Point pt2(x+bloc,y+bloc);
      ColorDistribution cd = getColorDistribution(input, pt1, pt2);
      float min = moyenneDistance(cd, all_col_hists[0]);
      int index = 0;
      for (int i = 1; i < all_col_hists.size(); i++){
        float dist = moyenneDistance(cd, all_col_hists[i]);
        if (dist < min){
          min = dist;
          index = i;
        }
      }
      rectangle(output, pt1, pt2, colors[index], -1);
    }
  }
  Mat output2 = output.clone();
  for(int row = bloc; row < height; row += bloc){
    for(int col = bloc; col < width; col += bloc){
      Point pt1(col,row);
      Point pt2(col+bloc,row+bloc);
      std::vector<int> scores = {};
      std::vector<Vec3b> colors = {};
      for(int x = 0; x < 3; x++){
        for(int y = 0; y < 3; y++){
          Vec3b color = output2.at<Vec3b>(row + (y-1) * bloc, col + (x-1) * bloc);
          if(std::find(colors.begin(), colors.end(), color) == colors.end()){
            colors.push_back(color);
            scores.push_back(1);
          }
          else{
            int index = std::find(colors.begin(), colors.end(), color) - colors.begin();
            scores[index] += 1;
          }
        }
      }
      int max = scores[0];
      int index = 0;
      for(int i = 1; i < scores.size(); i++){
        if(scores[i] > max){
          max = scores[i];
          index = i;
        }
      }
      rectangle(output, pt1, pt2, colors[index], -1);
    }
  }
  return output;
} 

int main( int argc, char** argv )
{
  std::vector<ColorDistribution> col_hists;

  Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;
  VideoCapture* pCap = nullptr;
  const int width = 640;
  const int height= 480;
  const int size  = 50;
  // Ouvre la camera
  pCap = new VideoCapture( 0 );
  if( ! pCap->isOpened() ) {
    cout << "Couldn't open image / camera ";
    return 1;
  }
  // Force une camera 640x480 (pas trop grande).
  pCap->set( CAP_PROP_FRAME_WIDTH, 640 );
  pCap->set( CAP_PROP_FRAME_HEIGHT, 480 );
  (*pCap) >> img_input;
  if( img_input.empty() ) return 1; // probleme avec la camera
  Point pt1( width/2-size/2, height/2-size/2 );
  Point pt2( width/2+size/2, height/2+size/2 );
  namedWindow( "input", 1 );
  int seuil = 5;
  createTrackbar( "seuil", "input", &seuil, 100);
  imshow( "input", img_input );
  bool freeze = false;
  bool reco = false;
  std::vector<std::vector<ColorDistribution>> all_col_hists;
  std::vector<Vec3b> colors;
  ColorDistribution distrib;
  ColorDistribution distrib2;
  Point p1(0,0);
  Point p2(width/2,height);
  Point p3(width/2,0);
  Point p4(width,height);
  while ( true )
    {
      char c = (char)waitKey(50); // attend 50ms -> 20 images/s
      if ( pCap != nullptr && ! freeze )
        (*pCap) >> img_input;     // récupère l'image de la caméra
      if ( c == 27 || c == 'q' )  // permet de quitter l'application
        break;
      if ( c == 'f' ) // permet de geler l'image
        freeze = ! freeze;
      if ( c == 'v' ){
        distrib = getColorDistribution(img_input, p1, p2);
        distrib2 = getColorDistribution(img_input, p3, p4);
        cout << "Distance : " << distrib.distance(distrib2) << endl;
        distrib.reset();
        distrib2.reset();
      }
      if ( c == 'c'){
        ColorDistribution cd = getColorDistribution(img_input, p1, p2);
        for(int i = 0; i < col_hists.size(); i++){
          if (cd.distance(col_hists[i]) < seuil/100){
            col_hists.erase(col_hists.begin() + i);
          }
        }
        col_hists.push_back(cd);
        
      }
      if (c == 'r'){
        reco = !reco;
        colors.clear();
        for(int i = 0; i < all_col_hists.size(); i++){
          int r = (i) & 1;
          int g = (i) & 2;
          int b = (i) & 4;
          colors.push_back(Vec3b(255 * b, 255 * g, 255 * r));
        }
      }
      if (c == 's'){
        all_col_hists.push_back(col_hists);
        col_hists.clear();
      }
      Mat output = img_input;
      if ( reco ) 
      { // mode reconnaissance
        Mat gray;
        cvtColor(img_input, gray, COLOR_BGR2GRAY);
        Mat reco = recoObject( img_input, all_col_hists, colors, 8 );
        cvtColor(gray, img_input, COLOR_GRAY2BGR);
        output = 0.5 * reco + 0.5 * img_input; // mélange reco + caméra
      }
      else
        cv::rectangle( img_input, pt1, pt2, Scalar( { 255.0, 255.0, 255.0 } ), 1 );
      imshow( "input", output );
    }
  return 0;
}