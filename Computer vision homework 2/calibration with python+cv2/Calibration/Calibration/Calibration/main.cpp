#include <cv.h>
#include <highgui.h>
#include <iostream>
#include<opencv\cv.hpp>
#include<opencv2\opencv.hpp>

#include<stdlib.h>
#include<stdio.h>


using namespace std;
using namespace cv;
void PrintMat(CvMat*);
void FputMat(FILE *, CvMat *);
int main(int argc, char * argv[])
{
	/*读入图像*/
	CvSize image_size;
	int n_board ;//图像数目
	cout << "input the number of input_images: ";
	cin >> n_board;
	cout << endl;
	int sn_board = 0;//成功找到角点的图像数目
	int board_w = 6;
	int board_h = 9;
	int board_n = board_h*board_w;//每幅图像的角点数
	CvSize patter_size = cvSize(board_w, board_h);//每幅图像的角点数
	CvPoint2D32f * corners = new CvPoint2D32f[board_n];//一幅图像的角点数组
	CvMat * object_points = cvCreateMat(board_n*n_board, 3, CV_32FC1);
	CvMat * image_points = cvCreateMat(board_n*n_board, 2, CV_32FC1);
	CvMat * point_counts = cvCreateMat(n_board, 1, CV_32SC1);
	for (int i = 1; i <= n_board; i++)
	{
		/*读入图像*/
		char path[100] = "images\\";
		char num[30];
		_itoa(i, num, 10);
		strcat(num, ".jpg");
		IplImage *SourceImg = cvLoadImage(strcat(path, num), CV_LOAD_IMAGE_COLOR);
		image_size = cvGetSize(SourceImg);//图像的大小
		IplImage *SourceImgGray = cvCreateImage(image_size, IPL_DEPTH_8U, 1);

		cvCvtColor(SourceImg, SourceImgGray, CV_BGR2GRAY);
		cvNamedWindow("MyCalib", CV_WINDOW_AUTOSIZE);
		cvShowImage("MyCalib", SourceImg);
		if (i < 3)
		{
		cvWaitKey(NULL);
		}


		/*对没一幅图像  提取角点   并精确到亚像素*/

		int corner_count;

		if (0 == cvFindChessboardCorners(SourceImgGray, patter_size, corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS))

		{
			cout << "......无法找出第" << i << "幅图的角点" << endl;
			cvWaitKey(33);
			cvReleaseImage(&SourceImgGray);
			cvReleaseImage(&SourceImg);
			continue;
			//return -1;
		}
		else
		{
			cvFindCornerSubPix(SourceImgGray, corners, corner_count, cvSize(11, 11), cvSize(-1, -1)
				, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));							//CvSize win 和matlab标定工具箱类似
			cvDrawChessboardCorners(SourceImg, patter_size, corners, corner_count, 1);
			cvShowImage("MyCalib", SourceImg);
			//cvSaveImage("F:\\vs2008test\\mycalib_0\\11.jpg",SourceImg);
		}
		if (i < 3)
		{
			cvWaitKey(NULL);
		}
		else
		if (i < n_board)
			{
				cvWaitKey(200);
			}

		for (int j = 0; j<board_n; j++)
		{

			CV_MAT_ELEM(*image_points, float, sn_board*board_n + j, 0) = corners[j].x;
			CV_MAT_ELEM(*image_points, float, sn_board*board_n + j, 1) = corners[j].y;
			CV_MAT_ELEM(*object_points, float, sn_board*board_n + j, 0) = float(j / board_w);
			CV_MAT_ELEM(*object_points, float, sn_board*board_n + j, 1) = float(j%board_w);
			CV_MAT_ELEM(*object_points, float, sn_board*board_n + j, 2) = 0.0f;
			CV_MAT_ELEM(*point_counts, int, sn_board, 0) = board_n;
		}
		sn_board++;
		cvReleaseImage(&SourceImgGray);
		cvReleaseImage(&SourceImg);
		cout << "......成功找出第" << i << "幅图的角点" << endl;
	}
	cout << "......一共成功获得" << sn_board << "幅图像的角点" << "......无法获得" << n_board - sn_board << "幅图像的角点" << endl << endl << endl << endl;
	cvWaitKey(NULL);
	//重新赋值
	CvMat * object_points0 = cvCreateMat(board_n*sn_board, 3, CV_32FC1);
	CvMat * image_points0 = cvCreateMat(board_n*sn_board, 2, CV_32FC1);
	CvMat * point_counts0 = cvCreateMat(sn_board, 1, CV_32SC1);
	for (int i = 0; i<sn_board*board_n; i++)
	{

		CV_MAT_ELEM(*image_points0, float, i, 0) = CV_MAT_ELEM(*image_points, float, i, 0);
		CV_MAT_ELEM(*image_points0, float, i, 1) = CV_MAT_ELEM(*image_points, float, i, 1);
		CV_MAT_ELEM(*object_points0, float, i, 0) = CV_MAT_ELEM(*object_points, float, i, 0);
		CV_MAT_ELEM(*object_points0, float, i, 1) = CV_MAT_ELEM(*object_points, float, i, 1);
		CV_MAT_ELEM(*object_points0, float, i, 2) = 0.0f;
	}
	for (int i = 0; i<sn_board; i++)
	{
		CV_MAT_ELEM(*point_counts0, int, i, 0) = CV_MAT_ELEM(*point_counts, int, i, 0);
	}
	cvReleaseMat(&object_points);
	cvReleaseMat(&point_counts);
	cvReleaseMat(&image_points);
	/*摄像机标定并求得内部参数*/

	CvMat * camera_matrix = cvCreateMat(3, 3, CV_32FC1);
	CvMat * distortion_coeffs = cvCreateMat(1, 4, CV_32FC1);
	CvMat * rotation_vectors = cvCreateMat(sn_board, 3, CV_32FC1);
	CvMat * translation_vectors = cvCreateMat(sn_board, 3, CV_32FC1);
	int flags = 0;



	cvCalibrateCamera2(object_points0, image_points0, point_counts0, image_size, camera_matrix
		//,distortion_coeffs,NULL,NULL,flags);
		, distortion_coeffs, rotation_vectors, translation_vectors, 0);
	//输出结果
	printf("/*****摄像机内部参数*****/\n");
	PrintMat(camera_matrix);
	printf("/*****畸变参数k1,k2,p1,p2*****/\n");
	PrintMat(distortion_coeffs);
	//cvWaitKey(NULL);
	//保存数据
	cout << "......保存内部参数与畸变参数" << endl << endl << endl<<endl;
	cvSave("camera_matrix1111.xml", camera_matrix);
	cvSave("distortion_coeffs.xml", distortion_coeffs);


	cvWaitKey(NULL);
	FILE *fp = NULL;
	if ((fp = fopen("camera_matrix.txt", "w")) == NULL)
	{
		printf("无法打开文件！");
		exit(1);
	}
	FputMat(fp, camera_matrix);
	fclose(fp);
	if ((fp = fopen("distortion_coeffs.txt", "w")) == NULL)
	{
		printf("无法打开文件！");
		exit(1);
	}
	FputMat(fp, distortion_coeffs);
	fclose(fp);

	//误差分析
	CvMat * object_points2 = cvCreateMat(board_n, 3, CV_32FC1);
	CvMat * image_points2 = cvCreateMat(board_n, 2, CV_32FC1);
	CvMat * rotation_vectors2 = cvCreateMat(1, 3, CV_32FC1);
	CvMat * translation_vectors2 = cvCreateMat(1, 3, CV_32FC1);
	CvMat * Err = cvCreateMat(sn_board*board_n, 2, CV_32FC1);
	for (int k = 0; k<sn_board; k++)
	{
		for (int i = 0; i<board_n; i++)//取一幅图的数据
		{
			CV_MAT_ELEM(*object_points2, float, i, 0) = CV_MAT_ELEM(*object_points0, float, k*board_n + i, 0);
			CV_MAT_ELEM(*object_points2, float, i, 1) = CV_MAT_ELEM(*object_points0, float, k*board_n + i, 1);
			CV_MAT_ELEM(*object_points2, float, i, 2) = 0.0f;
		}
		for (int i = 0; i<3; i++)
		{
			CV_MAT_ELEM(*rotation_vectors2, float, 0, i) = CV_MAT_ELEM(*rotation_vectors, float, k, i);
			CV_MAT_ELEM(*translation_vectors2, float, 0, i) = CV_MAT_ELEM(*translation_vectors, float, k, i);
		}
		cvProjectPoints2(object_points2, rotation_vectors2, translation_vectors2, camera_matrix,
			distortion_coeffs, image_points2);
		for (int i = 0; i<board_n; i++)
		{
			CV_MAT_ELEM(*Err, float, k*board_n + i, 0) = CV_MAT_ELEM(*image_points0, float, k*board_n + i, 0) - CV_MAT_ELEM(*image_points2, float, i, 0);
			CV_MAT_ELEM(*Err, float, k*board_n + i, 1) = CV_MAT_ELEM(*image_points0, float, k*board_n + i, 1) - CV_MAT_ELEM(*image_points2, float, i, 1);
		}
	}
	//PrintMat(Err);
	CvMat * Err_abs = cvCreateMat(board_n*sn_board, 2, CV_32FC1);
	cvAbs(Err, Err_abs);
	CvScalar Dmean;
	CvScalar Ddev;
	cvAvgSdv(Err_abs, &Dmean, &Ddev);  // 计算平均值 和  标准差
	printf("/*****反投影误差分析*****/\n");
	cout << "......绝对值误差矩阵的平均值：" << endl;
	cout << Dmean.val[0] << endl;
	cout << "......绝对值误差矩阵的标准差：" << endl;
	cout << Ddev.val[0] << endl;
	cout << "......保存误差矩阵" << endl << endl;
	cvSave("err.xml", Err);
	if ((fp = fopen("err.txt", "w")) == NULL)
	{
		printf("无法打开文件！");
		exit(1);
	}
	FputMat(fp, Err);
	fclose(fp);

	cout << "put Enter to out" << endl;
	while (1)
	{
		char c = waitKey(33);
		if (c == 13)
			break;
	}

	cvReleaseMat(&object_points2);
	cvReleaseMat(&image_points2);
	cvReleaseMat(&rotation_vectors2);
	cvReleaseMat(&translation_vectors2);
	cvReleaseMat(&Err);
	cvReleaseMat(&Err_abs);

	//释放总变量
	//cvWaitKey(NULL);
	cvReleaseMat(&object_points0);
	cvReleaseMat(&point_counts0);
	cvReleaseMat(&image_points0);
	cvReleaseMat(&camera_matrix);
	cvReleaseMat(&distortion_coeffs);
	cvReleaseMat(&rotation_vectors);
	cvReleaseMat(&translation_vectors);
	delete[] corners;
	cvDestroyWindow("MyCalib");
}

void PrintMat(CvMat* arry)
{
	for (int i = 0; i<arry->rows; i++)
	{
		for (int j = 0; j<arry->cols; j++)
		{
			printf("%f", CV_MAT_ELEM(*arry, float, i, j));
			if (j<arry->cols - 1)
				printf(",");
		}
		printf("\n");
	}
}
void FputMat(FILE * fp, CvMat * arry)
{
	for (int i = 0; i<arry->rows; i++)
	{
		for (int j = 0; j<arry->cols; j++)
		{
			fprintf(fp, "%f", CV_MAT_ELEM(*arry, float, i, j));
			if (j<arry->cols - 1)
				fprintf(fp, ",");
		}
		fprintf(fp, "\n");
	}
}

