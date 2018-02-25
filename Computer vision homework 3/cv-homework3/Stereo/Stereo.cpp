// StereoEvaluating.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
using namespace std;
using namespace cv;

IplImage * originImg1;
IplImage * originImg2;
IplImage * img1;
IplImage * img2;
void loadImage()
{
	originImg1=cvLoadImage("tsukuba_l.ppm");
	originImg2=cvLoadImage("tsukuba_r.ppm");
	if(!originImg1||!originImg2)
		cout<<"Failed to load Image!"<<endl;
	img1=cvCreateImage(cvGetSize(originImg1),8,1);
	img2=cvCreateImage(cvGetSize(originImg2),8,1);
	cvCvtColor(originImg1,img1,CV_RGB2GRAY);
	cvCvtColor(originImg2,img2,CV_RGB2GRAY);
}

void showhorizontal()
{
	CvMat* pair=cvCreateMat(img1->height,img1->width*2,CV_8UC3);
	CvMat part;//使用指针和对象在这里不一样，对象成员变量已分配只是没有分配数据区
	cvGetCols(pair,&part,0,img1->width);//相当于把pair左半部分内存给了part
	cvCvtColor(img1,&part,CV_GRAY2BGR);
	cvGetCols(pair,&part,img1->width,img1->width*2);
	cvCvtColor(img2,&part,CV_GRAY2BGR);
	for (int j=0;j<img1->height;j+=16)
		cvLine(pair,cvPoint(0,j),cvPoint(img1->width*2,j),CV_RGB(0,255,0));
	cvNamedWindow("rectified",0);
	cvShowImage("rectified",pair);
	cvWaitKey(0);
	cvReleaseMat(&pair);
	cvDestroyWindow("rectified");
}

void BM()
{
	CvStereoBMState* BMState=cvCreateStereoBMState();
	assert(BMState);
	BMState->preFilterSize=9;
	BMState->preFilterCap=31;
	BMState->SADWindowSize=15;
	BMState->minDisparity=0;
	BMState->numberOfDisparities=64;
	BMState->textureThreshold=10;
	BMState->uniquenessRatio=15;
	BMState->speckleWindowSize=100;
	BMState->speckleRange=32;
	BMState->disp12MaxDiff=1;

	CvMat* disp=cvCreateMat(img1->height,img1->width,CV_16S);
	CvMat* vdisp=cvCreateMat(img1->height,img1->width,CV_8U);
	int64 t=getTickCount();
	cvFindStereoCorrespondenceBM(img1,img2,disp,BMState);
	t=getTickCount()-t;
	cout<<"Time elapsed:"<<t*1000/getTickFrequency()<<endl;
	cvSave("disp.xml",disp);
	cvNormalize(disp,vdisp,0,255,CV_MINMAX);
	cvNamedWindow("BM_disparity");
	cvShowImage("BM_disparity",vdisp);
	cvWaitKey(0);
	//cvSaveImage("cones\\BM_disparity.png",vdisp);
	cvReleaseMat(&disp);
	cvReleaseMat(&vdisp);
	cvDestroyWindow("BM_disparity");
}

int _tmain(int argc, _TCHAR* argv[])
{

	loadImage();
	showhorizontal();
	int tag=0;
    cout<<"--------------using bm method-----------"<<endl;
    BM();
	waitKey(6000);
	cvDestroyAllWindows();
	return 0;
}

