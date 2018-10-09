#include <bits/stdc++.h>
#define M 17
#define N 8 
#define pi 3.1415926
//M是数据数 N是特征数。 
 
using namespace std;

long double x[M][N] = {
    {2,2,2,1,3,1,0.697,0.460},
    {3,2,3,1,3,1,0.744,0.376},
    {3,2,2,1,3,1,0.634,0.264},
    {2,2,3,1,3,1,0.608,0.318},
    {1,2,2,1,3,1,0.556,0.215},
    {2,1,2,1,2,2,0.403,0.237},
    {3,1,2,2,2,2,0.481,0.149},
    {3,1,2,1,2,1,0.437,0.211},
    {3,1,3,2,2,1,0.666,0.091},
    {2,3,1,1,1,2,0.243,0.267},
    {1,3,1,3,1,1,0.245,0.057},
    {1,2,2,3,1,2,0.343,0.099},
    {2,1,2,2,3,1,0.639,0.161},
    {1,1,3,2,3,1,0.657,0.198},
    {3,1,2,1,2,2,0.360,0.370},
    {1,2,2,3,1,1,0.593,0.042},
    {2,2,3,2,2,1,0.719,0.103}
};

int y[M] = {1, 1, 1, 1, 1, 1, 1, 1, 
			0, 0, 0, 0, 0, 0, 0, 0, 0}; 

long double p[N - 2][4][2];  //p(x_i = j|y = k)

long double mu_pos_7, mu_neg_7, sigma_pos_7, sigma_neg_7, mu_pos_8, mu_neg_8, sigma_pos_8, sigma_neg_8;

long double py[1];

//由于第7 8个特征不是连续的，所以假设他们的P(x|y)服从高斯分布，需要计算对应的mu和sigma
pair<long double, long double> train_gauss(int feature_dim, bool is_positive) {
	int dest = is_positive ? 1 : 0;
	int cnt = 0;
	long double summ = 0, mu, sigma;
	for (int i = 0; i < M; i++) {
		if (y[i] == dest) {
			cnt++;
			summ += x[i][feature_dim];
		}
	}
	mu = summ / cnt;
	summ = 0;
	for (int i = 0; i < M; i++) {
		if (y[i] == dest)
			summ += (x[i][feature_dim] - mu) * (x[i][feature_dim] - mu);
	}
	sigma = sqrt(summ / (cnt - 1));
//	cout << mu << " " << sigma << endl;
	return make_pair(mu, sigma);
}

long double cal_gauss_p(long double x, long double mu, long double sigma) {
	return (1 / (sqrt(2 * pi) * sigma)) * exp(-(((x-mu)*(x-mu))/(2*sigma*sigma)));
}

long double cal_p(int feature_dim, long double value_num, bool is_positive) {
	int dest = is_positive ? 1 : 0;
	int cnt = 0;
	int summ = 0;
	for (int i = 0; i < M; i++) {
		if (y[i] == dest) {
			cnt++;
			if (x[i][feature_dim] == value_num)
				summ++;
		}
	}
	return (summ + 1) / (cnt + value_num); //laplacian smoothing
}

void train() {
	int cnt = 0;
	for (int i = 0; i < M; i++)
		if (y[i] == 1) cnt++;
	py[1] = (long double)cnt / M;
	py[0] = 1 - py[1];
	for (int i = 0; i < N - 2; i++) {
		int value_num = (i == N - 3) ? 2 : 3;
		for (int j = 1; j <= value_num; j++) {
			p[i][j][0] = cal_p(i, j, 0);
			p[i][j][1] = cal_p(i, j, 1);
		}
	}
	pair<long double, long double> tmp;
	tmp = train_gauss(N - 2, 1);
	mu_pos_7 = tmp.first; sigma_pos_7 = tmp.second;
	tmp = train_gauss(N - 2, 0);
	mu_neg_7 = tmp.first; sigma_neg_7 = tmp.second;
	tmp = train_gauss(N - 1, 1);
	mu_pos_8 = tmp.first; sigma_pos_8 = tmp.second;
	tmp = train_gauss(N - 1, 0);
	mu_neg_8 = tmp.first; sigma_neg_8 = tmp.second;
}

long double get_prob(const vector<long double> &v, bool lable) {
	int dest = lable ? 1 : 0;
	long double prob = py[dest];
	for (int i = 0; i < N - 2; i++) {
		prob *= p[i][(int)v[i]][dest];
	}
	long double tmp;
	if (dest) {
		prob *= cal_gauss_p(v[6], mu_pos_7, sigma_pos_7) * cal_gauss_p(v[7], mu_pos_8, sigma_pos_8);
	}
	else {
		prob *= cal_gauss_p(v[6], mu_neg_7, sigma_neg_7) * cal_gauss_p(v[7], mu_neg_8, sigma_neg_8);
	}
	return prob;
}

bool get_result(const vector<long double> &v) {
	long double pos_prob, neg_prob;
	pos_prob = get_prob(v, 1);
	neg_prob = get_prob(v, 0);
	cout << "好瓜概率：" << pos_prob / (neg_prob + pos_prob) << endl;
	cout << "坏瓜概率：" << neg_prob / (neg_prob + pos_prob) << endl;
	return pos_prob > neg_prob ? 1 : 0; 
}

int main(int argc, char** argv) {
	train();
	long double a[8] = {2, 2, 2, 1, 3, 1, 0.697, 0.46};
	long double b[8] = {1, 2, 2, 3, 1, 1, 0.593, 0.042};
	vector<long double> v1(a, a + 8);
	vector<long double> v2(b, b + 8);
	for (int i = 0; i < N; i++)
		cout << v1[i] << " ";
	cout << endl;
	cout << "1 代表好瓜，0代表坏瓜，结果是：" << get_result(v1) << endl;
	cout << endl;
	for (int i = 0; i < N; i++)
		cout << v2[i] << " ";
	cout << endl;
	cout << "1 代表好瓜，0代表坏瓜，结果是：" << get_result(v2) << endl;
	return 0;
}
