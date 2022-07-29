# 2022학년도 1학기 컴퓨터 비전 프로젝트
## 필기체 계산기

필기체 계산기 프로그램 실행 영상

https://user-images.githubusercontent.com/103232995/181704288-2cd9aff3-2597-4b78-9339-f9fd9b9296ef.mp4

유튜브 영상 : https://youtu.be/-ijsDiOtz5I

인식되는 문자들 

* 숫자 : 0 1 2 3 4 5 6 7 8 9
* 사칙연산 : + - x / 
* 괄호 : ( )
* 소수점 : .
* 루트 : √
* 원주율 : π
* 자연상수 : e 

## 필기체 계산기 알고리즘
![슬라이드1](https://user-images.githubusercontent.com/103232995/181683549-3ccc3bea-70e1-42ff-939a-9bb573fa464d.JPG)
![슬라이드2](https://user-images.githubusercontent.com/103232995/181683591-22b13e88-2230-4a4d-9a5c-b6835b0dc6e7.JPG)
![슬라이드3](https://user-images.githubusercontent.com/103232995/181683594-60ee789d-c39c-4a0d-b126-569b24f2f949.JPG)
![슬라이드4](https://user-images.githubusercontent.com/103232995/181683595-363545d9-45d0-44c7-8a4a-a603d84afdec.JPG)
![슬라이드5](https://user-images.githubusercontent.com/103232995/181683596-4bfcfcd0-df01-4ffb-9167-7a9b23a76177.JPG)
![슬라이드6](https://user-images.githubusercontent.com/103232995/181683598-f352de1b-b2bb-4d0c-8d2e-7c0fd2d307ef.JPG)
![슬라이드7](https://user-images.githubusercontent.com/103232995/181683599-737aa580-8719-4674-a1cb-f223ee3ccb6a.JPG)
![슬라이드8](https://user-images.githubusercontent.com/103232995/181683600-7e052d04-5633-47c5-9eb7-ebe459003472.JPG)
![슬라이드9](https://user-images.githubusercontent.com/103232995/181683602-0c9d2435-177e-4f5f-a7df-72e2eaaebea1.JPG)
![슬라이드10](https://user-images.githubusercontent.com/103232995/181683603-6e66ae44-707c-4485-a3d6-cbcf7f93984e.JPG)
![슬라이드11](https://user-images.githubusercontent.com/103232995/181683608-2a4ef2ca-9925-44d6-b153-7787bc0a4ba0.JPG)
![슬라이드12](https://user-images.githubusercontent.com/103232995/181683610-8f79779e-3ba6-4f22-8a41-d048073e6a6a.JPG)
![슬라이드13](https://user-images.githubusercontent.com/103232995/181683612-6d63234b-181d-4c1e-afa5-3a2874b613cc.JPG)
![슬라이드14](https://user-images.githubusercontent.com/103232995/181683614-5e043477-5507-4c74-a6d6-b2193ec0b3c9.JPG)
![슬라이드15](https://user-images.githubusercontent.com/103232995/181683616-e929536c-0433-465e-a0df-e2a666f3b51b.JPG)
![슬라이드16](https://user-images.githubusercontent.com/103232995/181683618-df141b1d-bcab-4613-9c13-f9f339bf6a50.JPG)
![슬라이드17](https://user-images.githubusercontent.com/103232995/181683620-220d142c-f028-47ed-ba4c-4015627c51d8.JPG)
![슬라이드18](https://user-images.githubusercontent.com/103232995/181683622-b17ee941-1e88-4fde-8878-41518ec10180.JPG)


## 필기체 계산기 소스코드

### 
``` c++


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <stack>
#define PLUS -2000000001
#define MINUS -2000000002
#define TIMES -2000000003
#define DIVIDE -2000000004
#define PARENTHESIS_LEFT -2000000005
#define PARENTHESIS_RIGHT -2000000006
#define PO -2000000007
#define ROUTE -2000000008
#define SQUARE -2000000009
// + - X / ( )
#define overflow -2000000000
using namespace cv;
using namespace std;
static Point pt; // 전역변수
static stack<double> ads;
void on_mouse(int event, int x, int y, int flags, void* userdata);
Mat y_cut(Mat m);
Mat x_cut(Mat m);
Mat c_cut(Mat m, int* p, double* c);
bool minus_(int* p);
bool one(int* p);
bool divide_(Mat m);
bool times_(Mat m, int* p, double* centroid);
bool plus_(Mat m, int* p, double* centroid);
bool po(int* p);
bool pi(Mat x, int x_avg, Mat y, int y_avg);
bool seven_route(Mat ym, int y_avg, int y9_y0);
bool route(Mat c);
bool par_left(Mat x, int x_avg, int l_right);
bool par_right(Mat x, int x_avg, int l_left);
bool three_five(Mat y, int y_avg);
bool two(Mat y, int y_avg);
bool zero(Mat c, int c_avg, Mat y, int y_avg, Mat x, int x_avg);
bool four(Mat x, int x_avg, Mat y, int y_avg);
bool nine(Mat x, int x_avg);
bool six(Mat x, int x_avg);
bool e_(Mat c);
vector<double> blank_space(vector<double>& space);
vector<double> calc_error_handling(vector<double>& v);
int pre(int i);
void a(int i);
vector<double> fix(vector<double>& v);
double calc(vector<double>& v);
vector<double> in_clac(vector<double>& v, int& letter_cnt_v, int& letter_po_cnt_v, int& par_left_v, int& par_right_v);
vector<double> route_result(vector<int>& route_in_cnt, vector<double> v);
vector<double> square_of(vector<double>& v);
```

### 메인 함수
``` c++
int main(void) {
	int k;

	Mat img(640, 1760, CV_8UC1, Scalar(0));
	line(img, Point(img.cols - 160, 0), Point(img.cols - 160, img.rows), Scalar(255), 2);
	line(img, Point(0, img.rows - 160), Point(img.cols, img.rows - 160), Scalar(255), 2);
	line(img, Point(img.cols - 120, img.rows - 100), Point(img.cols - 40, img.rows - 100), Scalar(255), 5);
	line(img, Point(img.cols - 120, img.rows - 60), Point(img.cols - 40, img.rows - 60), Scalar(255), 5);
	line(img, Point(img.cols - 160, img.rows - 320), Point(img.cols, img.rows - 320), Scalar(255), 2);
	line(img, Point(img.cols - 160, img.rows - 480), Point(img.cols, img.rows - 480), Scalar(255), 2);
	putText(img, "Ans", Point(img.cols - 135, img.rows - 380), FONT_HERSHEY_DUPLEX, 2, Scalar(255), 2);
	putText(img, "AC", Point(img.cols - 125, img.rows - 220), FONT_HERSHEY_DUPLEX, 2, Scalar(255), 2);
	putText(img, "integer", Point(img.cols - 140, 70), FONT_HERSHEY_DUPLEX, 1, Scalar(255), 1);
	putText(img, "part", Point(img.cols - 140, 105), FONT_HERSHEY_DUPLEX, 1, Scalar(255), 1);
	putText(img, "Calculation expression", Point(30, img.rows - 136), FONT_HERSHEY_DUPLEX, 1, Scalar(255), 1);

	namedWindow("img");
	setMouseCallback("img", on_mouse, (void*)&img);

	imshow("img", img);
	k = waitKey();
}
```

### 마우스 이벤트 

``` c++
void on_mouse(int event, int x, int y, int flags, void* userdata) {
	Mat img = *(Mat*)userdata;
	Mat c = img(Rect(0, 0, img.cols - 160 - 1, img.rows - 160 - 1)); // 숫자 영역
	switch (event) {
	case EVENT_LBUTTONDOWN:
		pt = Point(x, y);

		if (x > img.cols - 160 && y < img.rows - 160 && y > img.rows - 320) {
			c = Scalar(0);
		}
		else if (x > img.cols - 160 && y < img.rows - 320 && y > img.rows - 480) {
			c = Scalar(0);
			if (ads.size() == 0) {
				ads.push(0);
			}
			double a = ads.top();
			if (a - int(a) == 0) {
				int ads_result = a;
				putText(img, to_string(ads_result), Point(img.cols - 1600, img.rows - 320), FONT_HERSHEY_DUPLEX, 4, Scalar(255), 4);
			}
			else {
				double ads_result = a;
				putText(img, to_string(ads_result), Point(img.cols - 1600, img.rows - 320), FONT_HERSHEY_DUPLEX, 4, Scalar(255), 4);
			}
		}
		else if (x > img.cols - 160 && y < img.rows - 480) {
			c = Scalar(0);
			if (ads.size() == 0) {
				ads.push(0);
			}
			int a = (int)ads.top();
			putText(img, to_string(a), Point(img.cols - 1600, img.rows - 320), FONT_HERSHEY_DUPLEX, 4, Scalar(255), 4);
		}
		else if (x > img.cols - 160 && y > img.rows - 160) {
			cout << "--------------------------------------" << endl << endl;
			img(Rect(0, img.rows - 135, img.cols - 160 - 10, 135)) = Scalar(0);

			morphologyEx(c, c, MORPH_CLOSE, Mat(), Point(-1, -1), 3); // 닫기 연산으로 연결

			Mat labels, stats, centroids; // 레이블링
			int cnt = connectedComponentsWithStats(c, labels, stats, centroids);

			if (cnt == 1) {
				break;
			}

			vector<int> v, aaa; // v : x좌표 값 작음 ~ x좌표 값 큼, aaa : v벡터의 값에 대응되는 레이블링 번호
			for (int num = 1; num < cnt; num++) {
				int* p = stats.ptr<int>(num);
				v.push_back(p[0]);
			}
			sort(v.begin(), v.end());
			for (int i = 0; i < cnt - 1; i++) {
				for (int num = 1; num < cnt; num++) {
					int* p = stats.ptr<int>(num);
					if (v.at(i) == p[0]) {
						aaa.push_back(num);
					}
				}
			}

			vector<double> letter;

			int route_len = 0, route_len_stat = 0, route_in_cnt = 0, route_cnt = -1;
			vector<int> v_route_in_cnt;

			int before_p1 = 0;

			String Calculation = "";

			for (int& num : aaa) {
				cout << num << "번째 문자" << endl;

				int* p = stats.ptr<int>(num);
				double* centroid = centroids.ptr<double>(num);
				//	Mat n = img(Rect(p[0], p[1], p[2], p[3])); // 객체 영역



				Mat n, n_1;
				//	img(Rect(p[0], p[1], p[2], p[3])).copyTo(n);
				img.copyTo(n_1);
				n = n_1(Rect(p[0], p[1], p[2], p[3]));

				Mat n_labels = labels(Rect(p[0], p[1], p[2], p[3]));

				cout << n.at<uchar>(0, 0) << endl;

				for (int j = 0; j < n.rows; j++) { // 객체 안에 다른 객체가 있는 경우
					for (int i = 0; i < n.cols; i++) {
						if (n_labels.at<int>(j, i) != num) {
							n.at<uchar>(j, i) = 0;
						}
					}
				}

			//	cout << n.type() << ", " << n_labels.type() << endl;
			//	cout << "x : " << n.cols << "y ; " << n.rows << endl;

				vector<vector<Point>> contours;
				findContours(n, contours, RETR_LIST, CHAIN_APPROX_SIMPLE); // 외각선 검출

				Mat ym = y_cut(n);
				double y_avg = ym.at<int>(10, 0) / 10;

				// x age
				Mat xm = x_cut(n);
				double x_avg = xm.at<int>(0, 10) / 10.0;

				Mat cm = c_cut(n_1, p, centroid); // 
				double cm_avg = (cm.at<int>(0, 0) + cm.at<int>(1, 0) + cm.at<int>(0, 1) + cm.at<int>(1, 1)) / 4.0;
				cout << "인식된 숫자 : ";

				cout << "p2 = " << p[2] << ", p[3] = " << p[3] << endl;
```

### 숫자, 기호 구분 알고리즘

``` c++
if (contours.size() == 1) { // 1,2,3,5,7,+,-,x,/,(,)
		if (minus_(p)) {
			cout << "-" << endl;
			letter.push_back(MINUS);
			Calculation += "-";
		}
		else { // 1,2,3,5,7,+,x,/,(,)
			if (one(p)) {
				cout << "1" << endl;
				letter.push_back(1);
				Calculation += "1";
			}
			else if (divide_(cm)) {
				cout << "/" << endl;
				letter.push_back(DIVIDE);
				Calculation += "/";
			}
			else {
				if (times_(img, p, centroid)) { // x썻는데 5뜸
					cout << "x" << endl;
					letter.push_back(TIMES);
					Calculation += "x";
				}
				else {
					if (plus_(img, p, centroid)) {
						cout << "+" << endl;
						letter.push_back(PLUS);
						Calculation += "+";
					}
					else {
						if (p[2] < 30 && p[3] < 30) {
							cout << "." << endl;
							letter.push_back(PO);
							Calculation += ".";
						}
						else {
							if (pi(xm, x_avg, ym, y_avg)) {
								cout << "파이 임" << endl;
								if (letter.size() != 0 and letter.back() > overflow) {
									letter.push_back(TIMES);
								}
								letter.push_back(CV_PI);
								Calculation += "pi";
							}
							else {
								double y9_y0 = abs(ym.at<int>(0, 0) - ym.at<int>(9, 0)) / y_avg;
								if (seven_route(ym, y_avg, y9_y0)) {  // 인식 안됨 가끔
									cout << cm.at<int>(1, 1) << endl;
									if (route(cm)) {
										cout << "route" << endl;
										letter.push_back(ROUTE);
										route_cnt = letter.size();

										v_route_in_cnt.push_back(0);

										route_len_stat = p[0];
										route_len = p[0] + p[2];
										Calculation += "r";
									}
									else {
										cout << "7" << endl;
										letter.push_back(7);
										Calculation += "7";
									}
								}
								else {
									int l_right = 0, l_left = 0;
									for (int j = p[1] + 0.1 * p[3]; j < p[1] + 0.9 * p[3]; j++) {
										if (img.at<uchar>(j, p[0] + 0.9 * p[2]) > 0) {
											l_right++;
											break;
										}
									}
									for (int j = p[1] + 0.1 * p[3]; j < p[1] + 0.9 * p[3]; j++) {
										if (img.at<uchar>(j, p[0] + 0.1 * p[2]) > 0) {
											l_left++;
											break;
										}
									}
									if (par_left(xm, x_avg, l_right)) {
										cout << "(" << endl;
										letter.push_back(PARENTHESIS_LEFT);
										Calculation += "(";
									}
									else if (par_right(xm, x_avg, l_left)) {
										cout << ")" << endl;
										letter.push_back(PARENTHESIS_RIGHT);
										Calculation += ")";
									}
									else {
										if (three_five(ym, y_avg)) {
											Mat n_cut = img(Rect(p[0], p[1], centroid[0] - p[0], p[3]));
											vector<vector<Point>> contours_cut;
											findContours(n_cut, contours_cut, RETR_LIST, CHAIN_APPROX_SIMPLE); // 외각선 검출
											if (contours_cut.size() == 2) {
												cout << "5" << endl;
												letter.push_back(5);
												Calculation += "5";
											}
											else if (contours_cut.size() == 3) {
												cout << "3" << endl;
												letter.push_back(3);
												Calculation += "3";
											}
											else {
												cout << "인식안되서 overflow으로 처리 " << endl;
												letter.push_back(overflow);
											}
										}
										else if (two(ym, y_avg)) {
											cout << "2" << endl;
											letter.push_back(2);
											Calculation += "2";
										}
										else {
											cout << "인식안되서 overflow으로 처리 " << endl;
											cout << ".." << endl;
											letter.push_back(overflow);
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	else if (contours.size() == 2) {

		if (zero(cm, cm_avg, ym, y_avg, xm, x_avg)) {
			cout << "0" << endl;
			letter.push_back(0);
			Calculation += "0";
		}
		else if (four(xm, x_avg, ym, y_avg)) {
			cout << "4" << endl;
			letter.push_back(4);
			Calculation += "4";
		}
		else if (nine(xm, x_avg)) {
			cout << "9" << endl;
			//	cout << "cm(0,1) : " << cm.at<int>(0, 1) << ", cm(1,0) : " << cm.at<int>(1, 0) << endl;
			letter.push_back(9);
			Calculation += "9";
		}
		else if (six(xm, x_avg)) {
			if (e_(cm)) {
				cout << "e" << endl;
				if (letter.size() != 0 and letter.back() != SQUARE and letter.back() > overflow) {
					letter.push_back(TIMES);
				}
				letter.push_back(2.718281828459);
				Calculation += "e";
			}
			else {
				cout << "6" << endl;
				//	cout << "cm(0,1) : " << cm.at<int>(0, 1) << ", cm(1,0) : " << cm.at<int>(1, 0) << endl;
				letter.push_back(6);
				Calculation += "6";

				cout << " 9 dkf" << endl;
				cout << xm.at<int>(0, 9) + xm.at<int>(0, 8) << " " << 2.5 * x_avg << endl;
				cout << xm.at<int>(0, 7) + xm.at<int>(0, 8) << " " << 2.5 * x_avg << endl;
			}
		}
		else {
			cout << "인식안되서 overflow으로 처리 " << endl;
			letter.push_back(overflow);
			xm.at<int>(0, 0) + xm.at<int>(0, 1) + xm.at<int>(0, 2) > 3 * x_avg;
			cout << "xm(0,0~3) :" << xm.at<int>(0, 0) << ", " << xm.at<int>(0, 1) << ", " << xm.at<int>(0, 2) << endl;
			cout << "x_avg = " << x_avg;
			cout << "cm(0,1) : " << cm.at<int>(0, 1) << ", cm(1,0) : " << cm.at<int>(1, 0) << endl;
		}
	}
	else if (contours.size() == 3) {
		cout << "8" << endl;
		letter.push_back(8);
		Calculation += "8";
	}
	else {
		cout << "인식안되서 overflow으로 처리 " << endl;
		letter.push_back(overflow);
	}
	if (p[0] > route_len_stat and p[0] + p[2] < route_len) {
		route_in_cnt++;
		v_route_in_cnt.back()++;
	}

	if (before_p1 > p[1] + p[3]) {
		letter.emplace(letter.end() - 1, SQUARE);
		Calculation.insert(Calculation.size() - 1, "^");
	}

	if (letter.back() == PO) {
		before_p1 = 0;
	}
	else {
		before_p1 = p[1];
	}

	/*
	// 바운딩 박스
	rectangle(img, Rect(p[0], p[1], p[2], p[3]), Scalar(255), 1);

	// 무게중심
	line(img, Point(p[0], centroid[1]), Point(p[0] + p[2], centroid[1]), Scalar(255), 1);
	line(img, Point(centroid[0], p[1]), Point(centroid[0], p[1] + p[3]), Scalar(255), 1);

	// xm
	for (int i = 1; i < 10; i++) {
		line(img, Point(p[0] + 0.1 * i * p[2], p[1]), Point(p[0] + 0.1 * i * p[2], p[1] + p[3]), Scalar(255), 1);
	}

	// ym
	for (int i = 1; i < 10; i++) {
		line(img, Point(p[0], p[1] + 0.1 * i * p[3]), Point(p[0] + p[2], p[1] + 0.1 * i * p[3]), Scalar(255), 1);
	}
	*/
}
'''

### 숫자, 기호 구분 알고리즘

''' 
// 루트에 괄호 추가
			cout << Calculation << endl;
			int v_k = 0;
			if (v_route_in_cnt.size() > 0) {
				for (int i = 1; i < letter.size(); i++) {
					if (letter.at(i) == ROUTE) {
						if (letter.at(i - 1) > overflow) {
							letter.emplace(letter.begin() + i, TIMES);
						}
					}
				}
				for (int i = 0; i < Calculation.size(); i++) {
					if (Calculation[i] == 'r') {
						if (i + 2 < Calculation.size() and Calculation[i + 1] == 'p') {
							Calculation.insert(i + 2 + v_route_in_cnt.at(v_k), ")");
						}
						else {
							Calculation.insert(i + 1 + v_route_in_cnt.at(v_k), ")");
						}
						Calculation.insert(i + 1, "(");
						v_k++;
					}
				}
			}
			

			// 루트랑 지수 그림
			cout << "실행전 : " << Calculation << endl;
			vector<double> text_calc = letter;
		//	line(img, Point(30, 400), Point(30, 480), Scalar(255));
			int x_line = 30, line_po_cnt = 0;
			for (int i = 0; i < Calculation.size(); i++) {
				char ca = Calculation[i];

				if (Calculation.at(i) == '+' || Calculation.at(i) == '-') {
					x_line += 75;
				}
				else if (Calculation.at(i) == 'x') {
					x_line += 55;
				}
				else if (Calculation.at(i) == '/') {
					x_line += 70;
				}
				else if (Calculation.at(i) == '(' || Calculation.at(i) == ')') {
					x_line += 42;
				}
				else if (Calculation.at(i) == '.') {
					x_line += 32;
					if (line_po_cnt++ % 2 == 0) {
						x_line++;
					}
				}
				else if (Calculation.at(i) == 'r') {
					line(img, Point(10 + x_line, img.rows - 60), Point(20 + x_line, img.rows - 60), Scalar(255), 3);
					line(img, Point(20 + x_line, img.rows - 60), Point(30 + x_line, img.rows - 30), Scalar(255), 3);
					line(img, Point(30 + x_line, img.rows - 30), Point(45 + x_line, img.rows - 90), Scalar(255), 3);
					x_line += 50;
					Calculation.at(i) = ' ';
				}
				else if (Calculation.at(i) == '^') {
					
				//	cout << "^ 확인" << endl;
					String sq = "";
					for (int j = i + 1; j < Calculation.size();j++) {
						sq += Calculation[j];
					//	cout << "지수 : " << sq << endl;
						if (Calculation.at(j) == '+' || Calculation.at(j) == '-' || Calculation.at(j) == 'x' || Calculation.at(j) == '/' || Calculation.at(j) == ' '
							|| Calculation.at(j) == '(' || Calculation.at(j) == ')' || Calculation.at(j) == 'r' || Calculation.size() - 1 == j) {
							if (Calculation.size() - 1 != j) {
							//	cout << "sq 삭제 전 : " << sq << endl;
								sq.erase(sq.size() - 1, 1);
							}
							else {
								Calculation.erase(Calculation.size() - 1, 1);
							}
							putText(img, sq, Point(x_line - 10, img.rows - 100), FONT_HERSHEY_DUPLEX, 1, Scalar(255), 1);
							Calculation.erase(i, j - i);
							i--;
							break;
						}
					}
				}
				else {
					x_line += 60;
				}
				
			//	cout << "이번 숫자 " << Calculation.at(i) << endl;
			//	line(img, Point(x_line, 340), Point(x_line, 480), Scalar(255));
			}
			cout << "실행후 : " << Calculation << endl;




			cout << "처음 " << endl;
			for (int i = 0; i < letter.size(); i++) {
				cout << letter.at(i) << " ";
			}
			cout << endl;

			
			letter = route_result(v_route_in_cnt, letter); // 루트 값 먼저 계산
			letter = blank_space(letter); // 계산하고 남은 칸 지움

			cout << "루트 후 " << endl;
			for (int i = 0; i < letter.size(); i++) {
				cout << letter.at(i) << " ";
			}
			cout << endl;

			int letter_cnt = 0;
			int letter_po_cnt = 0;
			int par_left = 0, par_right = 0;


			letter = in_clac(letter, letter_cnt, letter_po_cnt, par_left, par_right); // 1 0 0 -> 100으로 만듦
		
			
			cout << "에러처리 시작 " << endl;
			letter = calc_error_handling(letter);
			cout << "calc_error_handling 실행 후 값 : ";


			for (int i = 0; i < letter.size(); i++) {
				cout << letter.at(i) << " ";
			}
			cout << endl;
			// 빈칸(overflow) 제거
			letter = blank_space(letter);
			cout << "in_clac 실행 후 : ";
			for (int i = 0; i < letter.size(); i++) {
				cout << letter.at(i) << " ";
			}
			cout << endl;

			letter = square_of(letter); // 지수 계산

			


			// 빈칸(overflow) 제거
			letter = blank_space(letter);
			
			//  + x -> + 0 x 
			int a = 0;
			for (int i = 1; i < letter.size(); i++) {
				if (letter.at(i) == PLUS || letter.at(i) == MINUS) {
					a = 1;
				}
				else if (a == 1 && (letter.at(i) == TIMES || letter.at(i) == DIVIDE)) {
					letter.emplace(letter.begin() + i, 0);
					a = 0;
				}
				else {
					a = 0;
				}

			}
			double res;
			if (letter.size() > 0) {
				letter = fix(letter);
				res = calc(letter); // 소수 7째 자리에서 반올림

				Calculation += "=";

				if (round(res * 1000000) / 1000000 - (int)res == 0) {
					int result = (int)res;
					Calculation += to_string(result);
					ads.push(result);
				}
				else {
					double result = res;
					Calculation += to_string(result);
					ads.push(result);
				}
			}
			cout << Calculation << endl;
			putText(img, Calculation, Point(30, img.rows - 30), FONT_HERSHEY_DUPLEX, 3, Scalar(255), 2);
		}
		cout << ads.size();
		imshow("img", img);
		break;

	case EVENT_MOUSEMOVE:
		if (flags & EVENT_FLAG_LBUTTON) {
			if (x < img.cols - 160 - 1 && y < img.rows - 160 - 1)
				line(img, pt, Point(x, y), Scalar(255), 5);
			imshow("img", img);
			pt = Point(x, y);
		}
		break;
	}
}

```

### 함수 정의

''' c++
Mat y_cut(Mat m) { // 0~9 : 각 번호의 합, 10 : 전체합
	Mat img = m;
	Mat ym = Mat::zeros(11, 1, CV_32SC1);
	int y_sum = 0;
	for (int y_num = 0; y_num < 10; y_num++) {
		for (int i = 0; i < img.cols; i++) {
			for (int j = 0.1 * y_num * img.rows; j < 0.1 * (y_num + 1) * img.rows; j++) {
				if (img.at<uchar>(j, i) > 0) {
					ym.at<int>(y_num, 0)++;
				}
			}
		}
		//			cout << y_num << "번째 : " << ym.at<int>(y_num, 0) << endl;
		y_sum += ym.at<int>(y_num, 0);
	}
	ym.at<int>(10, 0) = y_sum;
	return ym;
}
Mat x_cut(Mat m) {
	Mat img = m;
	Mat xm = Mat::zeros(1, 11, CV_32SC1);
	int x_sum = 0;
	for (int x_num = 0; x_num < 10; x_num++) {
		for (int j = 0; j < img.rows; j++) {
			for (int i = 0.1 * x_num * img.cols; i < 0.1 * (x_num + 1) * img.cols; i++) {
				if (img.at<uchar>(j, i) > 0) {
					xm.at<int>(0, x_num)++;
				}
			}
		}
		//			cout << x_num << "번째 : " << xm.at<int>(0, x_num) << endl;
		x_sum += xm.at<int>(0, x_num);
	}
	xm.at<int>(0, 10) = x_sum;
	return xm;
}
Mat c_cut(Mat m, int* p, double* centroid) {
	Mat img = m;
	Mat cm = Mat::zeros(2, 2, CV_32SC1);
	for (int i = p[0]; i < centroid[0]; i++) {
		for (int j = p[1]; j < centroid[1]; j++) {
			if (img.at<uchar>(j, i) > 0) {
				cm.at<int>(0, 0)++;
			}
		}
		for (int j = centroid[1]; j < p[1] + p[3]; j++) {
			if (img.at<uchar>(j, i) > 0) {
				cm.at<int>(1, 0)++;
			}
		}
	}
	for (int i = centroid[0]; i < p[0] + p[2]; i++) {
		for (int j = p[1]; j < centroid[1]; j++) {
			if (img.at<uchar>(j, i) > 0) {
				cm.at<int>(0, 1)++;
			}
		}
		for (int j = centroid[1]; j < p[1] + p[3]; j++) {
			if (img.at<uchar>(j, i) > 0) {
				cm.at<int>(1, 1)++;
			}
		}
	}
	return cm;
}
bool minus_(int* p) {
	if (p[2] / p[3] >= 4 && p[3] <= 30) {
		cout << "p2 : " << p[2] << "p3 : " << p[3] << endl;
		return true;
	}
	else {
		return false;
	}
}
bool one(int* p) {
	if (p[3] / p[2] >= 3) {
		cout << "p2 : " << p[2] << "p3 : " << p[3] << endl;
		return true;
	}
	else {
		return false;
	}
}
bool divide_(Mat m) {
	Mat cm = m;
	if (cm.at<int>(0, 1) > 4 * cm.at<int>(0, 0) && cm.at<int>(1, 0) > 4 * cm.at<int>(1, 1)) {
		return true;
	}
	else {
		return false;
	}
}
bool times_(Mat m, int* p, double* centroid) {
	Mat img = m;
	int x = 0;
	for (int j = centroid[1] - 0.1 * p[3]; j < centroid[1] + 0.1 * p[3]; j++) {
		for (int i = p[0]; i < p[0] + 0.1 * p[2]; i++) {
			if (img.at<uchar>(j, i) > 0) {
				x++;
				break;
			}
		}
		for (int i = p[0] + 0.9 * p[2]; i < p[0] + p[2]; i++) {
			if (img.at<uchar>(j, i) > 0) {
				x++;
				break;
			}
		}
	}
	for (int i = centroid[0] - 0.1 * p[2]; i < centroid[0] + 0.1 * p[2]; i++) {
		for (int j = p[1]; j < p[1] + 0.1 * p[3]; j++) {
			if (img.at<uchar>(j, i) > 0) {
				x++;
				break;
			}
		}
		for (int j = p[1] + 0.9 * p[3]; j < p[1] + p[3]; j++) {
			if (img.at<uchar>(j, i) > 0) {
				x++;
				break;
			}
		}
	}
	if (x == 0) {
		return true;
	}
	else {
		return false;
	}
}
bool plus_(Mat m, int* p, double* centroid) {
	Mat img = m;
	int plus = 0;
	for (int j = p[1]; j < p[1] + 0.2 * p[3]; j++) {
		for (int i = p[0]; i < p[0] + 0.2 * p[2]; i++) {
			if (img.at<uchar>(j, i) > 0) {
				plus++;
				break;
			}
		}
		for (int i = p[0] + 0.8 * p[2]; i < p[0] + p[2]; i++) {
			if (img.at<uchar>(j, i) > 0) {
				plus++;
				break;
			}
		}
	}
	for (int j = p[1] + 0.8 * p[3]; j < p[1] + p[3]; j++) {
		for (int i = p[0]; i < p[0] + 0.2 * p[2]; i++) {
			if (img.at<uchar>(j, i) > 0) {
				plus++;
				break;
			}
		}
		for (int i = p[0] + 0.8 * p[2]; i < p[0] + p[2]; i++) {
			if (img.at<uchar>(j, i) > 0) {
				plus++;
				break;
			}
		}
	}
	if (plus == 0) {
		return true;
	}
	else {
		return false;
	}
}
bool po(int* p) {
	if (p[2] < 30 && p[3] < 30) {
		return true;
	}
	else {
		return false;
	}
}
bool pi(Mat x, int x_avg, Mat y, int y_avg) {
	Mat xm = x, ym = y;
	if (ym.at<int>(0, 0) > 3 * y_avg && (ym.at<int>(8, 0) > y_avg || ym.at<int>(9, 0) > y_avg) &&
		(xm.at<int>(0, 1) > x_avg || xm.at<int>(0, 2) > x_avg || xm.at<int>(0, 3) > x_avg) &&
		(xm.at<int>(0, 5) > x_avg || xm.at<int>(0, 6) > x_avg)) {
		return true;
	}
	else {
		return false;
	}
}
bool seven_route(Mat y, int y_avg, int y9_y0) {
	Mat ym = y;
	if (ym.at<int>(9, 0) < y_avg && y9_y0 > 1.5) {
		return true;
	}
	else {
		return false;
	}
}
bool route(Mat c) {
	Mat cm = c;
	if (cm.at<int>(1, 1) == 0) {
		return true;
	}
	else {
		return false;
	}
}

bool par_left(Mat x, int x_avg, int l_right) {
	Mat xm = x;
	if (xm.at<int>(0, 0) > 2. * x_avg && l_right == 0) {
		return true;
	}
	else {
		return false;
	}
}
bool par_right(Mat x, int x_avg, int l_left) {
	Mat xm = x;
	if (xm.at<int>(0, 9) > 2. * x_avg && l_left == 0) {
		return true;
	}
	else {
		return false;
	}
}
bool three_five(Mat y, int y_avg) {
	Mat ym = y;
	if (ym.at<int>(3, 0) > y_avg || ym.at<int>(4, 0) > y_avg || ym.at<int>(5, 0) > y_avg || ym.at<int>(6, 0) > y_avg) {
		return true;
	}
	else {
		return false;
	}
}
bool two(Mat y, int y_avg) {
	Mat ym = y;
	if ((ym.at<int>(7, 0) + ym.at<int>(8, 0) + ym.at<int>(9, 0)) > 4 * y_avg) {
		return true;
	}
	else {
		return false;
	}
}
bool zero(Mat c, int cm_avg, Mat y, int y_avg, Mat x, int x_avg) {
	Mat cm = c, ym = y, xm = x;
	if (abs(cm.at<int>(0, 0) - cm_avg) < 0.4 * cm_avg && abs(cm.at<int>(0, 1) - cm_avg) < 0.4 * cm_avg &&
		abs(cm.at<int>(1, 0) - cm_avg) < 0.4 * cm_avg && abs(cm.at<int>(1, 1) - cm_avg) < 0.4 * cm_avg &&
		ym.at<int>(3, 0) < y_avg && ym.at<int>(4, 0) < y_avg && ym.at<int>(5, 0) < y_avg &&
		ym.at<int>(6, 0) < y_avg && ym.at<int>(7, 0) < y_avg &&
		xm.at<int>(0, 0) > x_avg && xm.at<int>(0, 9) > x_avg) {
		return true;
	}
	else {
		return false;
	}
}
bool four(Mat x, int x_avg, Mat y, int y_avg) {
	Mat xm = x, ym = y;
	if (xm.at<int>(0, 3) + xm.at<int>(0, 4) + xm.at<int>(0, 5) + xm.at<int>(0, 6) > 5 * x_avg ||
		ym.at<int>(3, 0) + ym.at<int>(4, 0) + ym.at<int>(5, 0) + ym.at<int>(6, 0) > 5 * x_avg) {
		return true;
	}
	else {
		return false;
	}
}
bool nine(Mat x, int x_avg) {
	Mat xm = x;
	if ((xm.at<int>(0, 9) + xm.at<int>(0, 8) > 2.5 * x_avg || xm.at<int>(0, 8) + xm.at<int>(0, 7) > 2.5 * x_avg)) {
		return true;
	}
	else {
		return false;
	}
}

bool six(Mat x, int x_avg) {
	Mat xm = x;
	if ((xm.at<int>(0, 0) + xm.at<int>(0, 1) + xm.at<int>(0, 2) > 2.8 * x_avg)) {
		return true;
	}
	else {
		return false;
	}
}
bool e_(Mat c) {
	Mat cm = c;
	if (cm.at<int>(0, 1) > cm.at<int>(1, 1)) {
		return true;
	}
	else {
		return false;
	}
}
// 빈칸(overflow) 제거
vector<double> blank_space(vector<double>& space) {
	vector<double> letter = space;
	for (vector<double>::iterator it = letter.begin(); it != letter.end();) {
		if (*it == overflow) {
			it = letter.erase(it);
		}
		else {
			it++;
		}
	}
	return letter;
}
vector<double> calc_error_handling(vector<double>& v) {
	vector<double> letter = v;

	int calc_cnt = 0;
	for (int i = 0; i < letter.size(); i++) {
		if (letter.at(i) <= PLUS && letter.at(i) >= DIVIDE) {
			if (calc_cnt > 0) {
				letter.at(i) = overflow;
			}
			calc_cnt++;
		}
		else {
			calc_cnt = 0;
		}
	}
	blank_space(letter);
	cout << "e" << endl;
	int cnt_num = 0;
	for (int i = 0; i < letter.size(); i++) {
		if (letter.at(i) > overflow) {
			cnt_num++;
			break;
		}
	}
	cout << "cnt_num : " << cnt_num << endl;
	if (cnt_num == 0) {
		letter.clear();
		letter.push_back(0);
	}

	for (int i = 0; i < letter.size(); i++) {
		cout << letter.at(i) << " ";
	}

	// 1. 기호 1개만 입력된 경우 : ( , ) , x ...
	if (letter.size() == 1 && letter.at(0) <= overflow) { // ) ㅊ
		letter.at(0) = 0; // 그 값을 0으로 변경.
	}
	// 2. 괄호 안에 아무 값도 없는 경우 : ()
	else {
		for (int i = 0; i < letter.size(); i++) {
			if (letter.at(i) == PARENTHESIS_LEFT) {
				// ( 로 식이 끝난 경우.
				// 2 + 3 - 2 ( 

				if (letter.size() - 1 == i) {
					letter.at(i) = overflow;
					// 2 + 3 - 2 x ( 
					
					if (i != 0 && letter.at(i - 1) < overflow && letter.at(i - 1) > PARENTHESIS_LEFT) {
						letter.at(i - 1) = overflow;
					}
					blank_space(letter);
				}
				cout << "l ";
				for (int i = 0; i < letter.size(); i++) {
					cout << letter.at(i) << " ";
				}
				cout << endl;

				// () -> (0)
				if ((i + 1) < letter.size()) {
					if (letter.at(i + 1) == PARENTHESIS_RIGHT) {
						letter.emplace(letter.begin() + i + 1, 0);
					}

					// ( + 2 ) -> ( 0 + 2 )
					else if (letter.at(i + 1) == PLUS || letter.at(i + 1) == MINUS) {
						letter.emplace(letter.begin() + i + 1, 0);
					}
				}

				cout << "e ";
				for (int i = 0; i < letter.size(); i++) {
					cout << letter.at(i) << " ";
				}
				cout << endl;

				if (i > 0 and letter.at(i - 1) > overflow) {
					if (letter.size() - 1 == i) {
						letter.at(i) = overflow;
						blank_space(letter);
					}
					else {
						letter.emplace(letter.begin() + i, TIMES);
					}
				}
			}
		}
	}
	blank_space(letter);
	cout << "a" << endl;
	
	for (int i = 0; i < letter.size(); i++) {
		cout << letter.at(i) << " ";
	}
	cout << endl;
	

	int z = 0;
	int z_before = 0;
	while (z < letter.size()) {
		double z_num = letter.at(z);
		if (z_num == PLUS || z_num == MINUS) {
			if (z == 0) { // + a -> a, - a -> -a
				letter.emplace(letter.begin(), 0);
			}
			else if (z > 0 and (letter.at(z_before) == PLUS || letter.at(z_before) == MINUS ||
				letter.at(z_before) == TIMES || letter.at(z_before) == DIVIDE)) {
				if (z_num == PLUS) { // a x + 2 -> a x 2
					letter.at(z) = letter.at(z + 1);
					letter.erase(letter.begin() + z + 1);
				}
				else { // a / - 3 -> a / -3
					letter.at(z) = -1 * letter.at(z + 1);
					letter.erase(letter.begin() + z + 1);
				}
			}
		}
		z_before = z;
		z++;
	}
	blank_space(letter);
	cout << "b" << endl;
	if (letter.size() > 0) {
		if (letter.at(0) == TIMES || letter.at(0) == DIVIDE) { // x 2 -> 0 x 2
			letter.emplace(letter.begin(), 0);
		}

		if (letter.back() == PLUS || letter.back() == MINUS || letter.back() == TIMES || letter.back() == DIVIDE) {
			letter.push_back(0); // 2 + -> 2 + 0
		}
	}
	cout << "c" << endl;
	int par_left = 0, par_right = 0;
	for (int i = 0; i < letter.size(); i++) {
		if (letter.at(i) == PARENTHESIS_LEFT) {
			par_left++;
		}
		else if (letter.at(i) == PARENTHESIS_RIGHT) {
			par_right++;
		}
	}

	if (par_left > par_right) { // (( ) -> (( ))
		for (int i = par_right; i < par_left; i++) {
			letter.push_back(PARENTHESIS_RIGHT);
		}
	}
	else if (par_left < par_right) { // ( )) -> (( ))
		for (int i = par_left; i < par_right; i++) {
			letter.emplace(letter.begin(), PARENTHESIS_LEFT);
		}
	}
	cout << "d" << endl;
	
	return letter;
}
int pre(int i) {
	switch (i) {
	case PARENTHESIS_LEFT: case PARENTHESIS_RIGHT: return 0;

	case PLUS: case MINUS: return 1;

	case TIMES: case DIVIDE: return 2;
	}
	return -1;
}
void a(int i) {
	switch (i) {
	case PLUS: cout << "+";
		break;
	case MINUS: cout << "-";
		break;
	case TIMES: cout << 'x';
		break;
	case DIVIDE: cout << "/";
		break;
	case PARENTHESIS_LEFT: cout << "(";
		break;
	case PARENTHESIS_RIGHT: cout << ")";
		break;
	case PO :cout << ".";
		break;
	case ROUTE: cout << "route";
		break;
	case SQUARE: cout << "^";
		break;
	}
	cout << " ";
}
vector<double> fix(vector<double>& v) {
	double c, op;
	stack<int> st;
	int k = 0;

	vector<double> result;

	while (k < v.size()) {
		c = v.at(k);
		if (c == PARENTHESIS_LEFT) st.push(c);
		else if (c == PARENTHESIS_RIGHT) {
			while (!st.empty()) {
				op = st.top();
				st.pop();
				if (op == PARENTHESIS_LEFT) break;
				else {
					a(op);
					result.push_back(op);
				}
			}
		}
		else if (c == PLUS || c == MINUS || c == TIMES || c == DIVIDE) {
			while (!st.empty()) {
				op = st.top();
				if (pre(c) <= pre(op)) {
					a(op);
					result.push_back(op);
					st.pop();
				}
				else break;
			}
			st.push(c);
		}
		else {
			cout << c << " ";
			result.push_back(c);
		}
		k++;
	}
	while (!st.empty()) {
		a(st.top());
		result.push_back(st.top());
		st.pop();
	}
	return result;
}
double calc(vector<double>& v) {
	stack<double> st;
	double c;
	int k = 0;
	while (k < v.size()) {
		c = v.at(k);
		if (c == PLUS || c == MINUS || c == TIMES || c == DIVIDE) {
			double val2 = st.top();
			st.pop();
			double val1 = st.top();
			st.pop();
			switch ((int)c) {
			case PLUS: st.push(val1 + val2); break;
			case MINUS: st.push(val1 - val2); break;
			case TIMES: st.push(val1 * val2); break;
			case DIVIDE: st.push(val1 / val2); break;
			}
		}
		else {
			st.push(c);
		}
		k++;
	}
	c = st.top();
	return c;
}
vector<double> in_clac(vector<double>& v, int& letter_cnt_v, int& letter_po_cnt_v, int& par_left_v, int& par_right_v) {
	vector<double> letter = v;
	int letter_cnt = 0, letter_po_cnt = 0, par_left = 0, par_right = 0;

	cout << "in_clac에 들어온 값 " << endl;
	for (int i = 0; i < letter.size(); i++) {
		cout << letter.at(i) << " ";
	}
	cout << endl;



	for (int i = 0; i < letter.size(); i++) {
		if (letter.at(i) > overflow) {
			int j = i - letter_cnt;
			if (letter_cnt > 0 && letter_po_cnt == 0) {
				letter.at(j) = letter.at(j) * 10 + letter.at(i);
				letter.at(i) = overflow;
			}
			else if (letter_po_cnt > 0) {
				if (letter_cnt == 1) {
					letter.at(j) = 0;
				}
				letter.at(j) = letter.at(j) + pow(0.1, letter_po_cnt) * letter.at(i);
				letter.at(i) = overflow;
				letter_po_cnt++;
			}
			letter_cnt++;
		}
		else {
			if (letter.at(i) == PO) {
				letter_po_cnt++;
				letter_cnt++;
				letter.at(i) = overflow;
			}
			else if (letter.at(i) == PARENTHESIS_LEFT) {
				par_left++;
				letter_cnt = 0;
				letter_po_cnt = 0;
			}
			else if (letter.at(i) == PARENTHESIS_RIGHT) {
				par_right++;
				letter_cnt = 0;
				letter_po_cnt = 0;
			}
			else {
				letter_cnt = 0;
				letter_po_cnt = 0;
			}
		}
	}

	cout << "in_clac에서 나가는 값 " << endl;
	for (int i = 0; i < letter.size(); i++) {
		cout << letter.at(i) << " ";
	}
	cout << endl;


	letter_cnt_v = letter_cnt, letter_po_cnt_v = letter_po_cnt;
	par_left_v = par_left, par_right_v = par_right;
	return letter;
}
vector<double> route_result(vector<int>& route_in_cnt, vector<double> v) {
	vector<double> letter = v;
	vector<double> c;
	int a = 0, b = 0, e = 0, d = 0;
	double res;
	vector<int> i_route_stat;
	vector<double> d_calc;
	for (int i = 0; i < letter.size(); i++) {
		if (letter.at(i) == ROUTE) {
			i_route_stat.push_back(i);
		}
	}
	for (int i = 0; i < i_route_stat.size(); i++) {
		if (route_in_cnt.at(i) == 0) {
			d_calc.push_back(0);
		}
		for (int j = 0; j < route_in_cnt.at(i); j++) {
			d_calc.push_back(letter.at(i_route_stat.at(i) + j + 1));
			letter.at(i_route_stat.at(i) + j + 1) = overflow;
		}
		cout << "a";
		d_calc = in_clac(d_calc, a, b, e, d);
		cout << "b";
		d_calc = blank_space(d_calc);
		d_calc = calc_error_handling(d_calc);
		cout << "c";
		d_calc = fix(d_calc);
		cout << "d";
		res = sqrt(calc(d_calc));
		cout << "e";
		letter.at(i_route_stat.at(i)) = res;
		cout << "f";
		d_calc.clear();
		cout << "g";
	}

	cout << "결과 : ";
	for (int i = 0; i < letter.size(); i++) {
		cout << letter.at(i) << " ";
	}
	cout << endl;

	cout << "route_reust 함수 종료 " << endl;
	return letter;
}
vector<double> square_of(vector<double>& v) {
	vector<double> letter = v;
	cout << "지수 계산 실행 전" << endl;
	for (int i = 0; i < letter.size(); i++) {
		if (letter.at(i) < overflow) {
			a(letter.at(i));
		}
		else if (letter.at(i) == overflow) {
			cout << "overflow ";
		}
		else {
			cout << letter.at(i) << " ";
		}
	}
	cout << endl;
	int square_cnt = 0;
	for (int i = 0; i < letter.size(); i++) {
		if (letter.at(i) == SQUARE) {
			square_cnt++;
		}
	}
	if (square_cnt == 0) {
		return letter;
	}
	int k = 0;
	int num = -1;

	int par_left = -1;

	// 괄호 없는거
	for (int i = 0; i < letter.size(); i++) {
		if (k == 0 && letter.at(i) > overflow) {
			num = i;
		}
		else if (k == 0 && letter.at(i) == PARENTHESIS_RIGHT) {
			num = i;
		}
		else if (num > -1 and letter.at(i) == SQUARE) {
			k = i;
			letter.at(i) = overflow;
		}
		else if (num > -1 and k) {
			if (letter.at(num) != PARENTHESIS_RIGHT) {
				k = 0;
				letter.at(num) = pow(letter.at(num), letter.at(i));
				letter.at(i) = overflow;
				cout << "3";
			}
			else { // 안됨
				cout << " zz" << endl;
				int par_left;
				vector<double> par_square;
				int sq_res = letter.at(i);

				letter.at(i) = overflow;
				for (int j = num; j >= 0; j--) {
					if (letter.at(j) == PARENTHESIS_LEFT) {
						par_left = j;
					}
				}

				cout << endl;
				cout << "par_left : " << par_left << ", num : " << num << endl;
				for (int j = par_left + 1; j < num; j++) {
					par_square.push_back(letter.at(j));
					letter.at(j) = overflow;
				}

				cout << "k3 ";
				for (int j = 0; j < letter.size(); j++) {
					if (letter.at(j) < overflow) {
						a(letter.at(j));
					}
					else if (letter.at(j) == overflow) {
						cout << "overflow ";
					}
					else {
						cout << letter.at(j) << " ";
					}
				}
				cout << endl;

				cout << "par : ";
				for (int j = 0; j < par_square.size(); j++) {
					if (par_square.at(j) < overflow) {
						a(par_square.at(j));
					}
					else if (par_square.at(j) == overflow) {
						cout << "overflow ";
					}
					else {
						cout << par_square.at(j) << " ";
					}
				}
				cout << endl << "letter : ";
				for (int j = 0; j < letter.size(); j++) {
					if (letter.at(j) < overflow) {
						a(letter.at(j));
					}
					else if (letter.at(j) == overflow) {
						cout << "overflow ";
					}
					else {
						cout << letter.at(j) << " ";
					}
				}
				cout << endl;
				letter.at(num) = overflow;
				cout << endl;
				par_square = fix(par_square);
				
				letter.at(par_left) = pow(calc(par_square), sq_res);
				letter = blank_space(letter);
				
			}
		}
	}

	letter = blank_space(letter);
	cout << "지수 함수 실행 결과" << endl;
	for (int i = 0; i < letter.size(); i++) {
		if (letter.at(i) < overflow) {
			a((int)letter.at(i));
		}
		else if (letter.at(i) == overflow) {
			cout << "overflow ";
		}
		else {
			cout << letter.at(i) << " ";
		}
	}
	
	cout << endl;

	return letter;
}
```
