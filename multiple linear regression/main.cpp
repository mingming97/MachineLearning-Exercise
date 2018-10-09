#include <bits/stdc++.h>
#include "matrix.h"
#define M 15

using namespace std;

//��ױƷ����������۲����Ϊ��������y(��ƿ)�������˿�x1(����)���˾�������x2(ǧԪ)�� 

//�����˿�x1 
long double xx1[M] = {27.4, 18.0, 37.5, 20.5, 8.60,
					  26.5, 9.8, 33.0, 19.5, 5.3,
					  43.0, 37.2, 23.6, 15.7, 37.0};

//�˾�����x2
long double xx2[M] = {2.450, 3.254, 3.802, 2.838, 2.347,
					   3.782, 3.008, 2.450, 2.137, 2.560,
					   4.020, 4.427, 2.660, 2.088, 2.605};

//��������y
long double yy[M] = {1.62, 1.20, 2.23, 1.31, 0.67,
					 1.69, 0.81, 1.92, 1.16, 0.55,
					 2.52, 2.32, 1.44, 1.03, 2.12};

//������Ԫ���Իع�ģ�� y = theta0 + theta1 * x1 + theta2 * x2 
int main(int argc, char** argv) {
	
	//feed the data
	vector<vector<long double> > tmp;
	tmp.push_back(vector<long double>(M, 1));
	tmp.push_back(vector<long double>(xx1, xx1 + M));
	tmp.push_back(vector<long double>(xx2, xx2 + M));
	Matrix x(3, M, tmp);
	x = x.transpose();
	// x.print();
	tmp.clear();
	tmp.push_back(vector<long double>(yy, yy + M));
	Matrix y(1, M, tmp);
	y = y.transpose();
	// y.print();
	Matrix theta(3, 1);
	
	//compute theta
	
	theta = (x.transpose() * x).inverse() * x.transpose() * y;
	theta.print();
	printf("Our model is y = %Lf + %Lf * x1 + %Lf * x2", theta(1, 1), theta(2, 1), theta(3, 1));
	return 0;
}
