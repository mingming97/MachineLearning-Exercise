#include <bits/stdc++.h>
#include "matrix.h" 
#define M 20

using namespace std;

//xx代表城镇居民家庭人均可支配收入 
long double xx[M] = {343.4, 477.6, 739.1, 1373.9, 1510.2,
					 1700.6, 2026.6, 2577.4, 3496.2, 4283.0,
					 4838.9, 5160.3, 5425.1, 5854.0, 6280.0,
					 6859.6, 7702.8, 8472.2, 9421.6, 10493.0};

//yy代表城市人均住宅面积					 
long double yy[M] = {6.7, 7.2, 10.0, 13.5, 13.7, 
					 14.2, 14.8, 15.2, 15.7, 16.3, 
					 17.0, 17.8, 18.7, 19.4, 20.3,
					 20.8, 22.8, 23.7, 25.0, 26.1};

//建立一元线性回归模型 y = theta0 + theta1 * x;
int main(int argc, char** argv) {
	
	//feed the data
	vector<vector<long double> > tmp;
	tmp.push_back(vector<long double>(M, 1));
	tmp.push_back(vector<long double>(xx, xx + M)); 
	Matrix x(2, M, tmp);
	x = x.transpose();
	tmp.clear();
	tmp.push_back(vector<long double>(yy, yy + M));
	Matrix y(1, M, tmp);
	y = y.transpose();
	Matrix theta(2, 1);
	
	//compute the theta
	theta = (x.transpose() * x).inverse() * x.transpose() * y;
	theta.print();
	
	printf("So our model is y = %Lf + %Lf * x", theta(1, 1), theta(2, 1));
	return 0;
}
