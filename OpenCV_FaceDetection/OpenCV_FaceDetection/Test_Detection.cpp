#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void FaceDetection_Image();
void HandDetection_Image();
void FaceDetection_Video();
void HandDetection_Video();

int main()
{
	FaceDetection_Image(); // Face Detection Based On Image
	//HandDetection_Image();

	//FaceDetection_Video();
	//HandDetection_Video();

	return 0;
}

void FaceDetection_Video()
{
	//동영상 파일로부터 부터 데이터 읽어오기 위해 준비  
	VideoCapture cap1("c://opencv_test/face2.mp4");
	if (!cap1.isOpened())
	{
		printf("동영상 파일을 열수 없습니다. \n");
	}

	Mat image;
	namedWindow("video", 1);

	// 추가
	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade;
	face_cascade.load("C:/OpenCV/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
	vector<Rect> faces;
	// 추가 끝
	int fpsNum = cap1.get((CV_CAP_PROP_FPS));

	while (1)
	{
		//웹캡으로부터 한 프레임을 읽어옴  
		cap1 >> image;

		face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		// Draw circles on the detected faces
		for (int i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(image, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}

		imshow("video", image);

		//30ms 정도 대기하도록 해야 동영상이 너무 빨리 재생되지 않음.  
		if (image.data == NULL)
			VideoCapture cap1("c://opencv_test/face2.mp4");

		if (waitKey(fpsNum) ==0) break; //ESC키 누르면 종료  
	}
}

void FaceDetection_Image()
{
	Mat image;
	image = imread("C://face_i.jpg", CV_LOAD_IMAGE_COLOR);

	if (image.data == NULL)
	{
		cout << "image file is empty" << endl;
	}

	namedWindow("window1", 1);   imshow("window1", image);

	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade;
	face_cascade.load("C:/OpenCV/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");

	// Detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	// Draw circles on the detected faces
	for (int i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(image, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}


	imshow("Detected Face", image);

	waitKey(0);
}

void HandDetection_Image()
{
	Mat image;
	image = imread("C://opencv_test/hand.png");

	Mat skinMat;
	cvtColor(image, skinMat, CV_BGR2YCrCb);
	inRange(skinMat, Scalar(0, 133, 77), Scalar(255, 173, 127), skinMat);

	// 처리 결과 표시
	imshow("Detected Face", image);
	imshow("Detected Face", skinMat);

	waitKey(0);
}

void HandDetection_Video()
{
	//동영상 파일로부터 부터 데이터 읽어오기 위해 준비  
	VideoCapture cap1("c://opencv_test/hand.mp4");
	if (!cap1.isOpened())
	{
		printf("동영상 파일을 열수 없습니다. \n");
	}

	Mat image;
	namedWindow("video", 1);
	Mat skinMat;
	int fps = cap1.get(CV_CAP_PROP_FPS); // 비디오를 초당 몇 프레임 읽어올 것인지 설정한다

	while (1)
	{
		//웹캡으로부터 한 프레임을 읽어옴  
		cap1 >> image;
		
		cvtColor(image, skinMat, CV_BGR2YCrCb);

		inRange(skinMat, Scalar(0, 133, 77), Scalar(255, 173, 127), skinMat);

		imshow("video", skinMat);

		//30ms 정도 대기하도록 해야 동영상이 너무 빨리 재생되지 않음.  
		if (image.data == NULL)
			VideoCapture cap1("c://opencv_test/hand.mp4");

		if (waitKey(fps) == 0) break; //ESC키 누르면 종료  
	}
}
