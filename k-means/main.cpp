#include <bits/stdc++.h>
#define M 96
#define N 5
#define K 3
//96 datas, 5 features, 3 clusterings
using namespace std;

vector<vector<long double> > mu(K, vector<long double>(N, 0));
vector<vector<long double> > old_mu(K, vector<long double>(N, 0));

long double fea_mu[N], fea_sigma[N];

int tag[M] = {0};

vector<vector<long double> > data;

string trim(string &str) {
	str.erase(0, str.find_first_not_of(" \t\r\n"));
	str.erase(str.find_last_not_of(" \t\r\n") + 1);
	return str;
}

long double stringtold(const string &s) {
	long double val;
	istringstream ss(s);
	ss >> val;
	return val;
}

void initialize() {
	int start[K]; 
	long double minnn = 90000, maxxx = -1;
	for (int i = 0; i < M; i++) {
		long double num = 0;
		for (int j = 0; j < N; j++)
			num += data[i][j];
		if (num > maxxx) {
			maxxx = num;
			start[0] = i;
		}
		if (num < minnn) {
			minnn = num;
			start[1] = i;
		}
	}
	for (int i = 0; i < M; i++) {
		if (i != start[0] && i != start[1])
			start[2] = i;
	}
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < N; j++)
			mu[i][j] = data[start[i]][j];
	}
}

void read_data() {
	ifstream fin("data.csv");
	string line;
	long double tmp[N];
	while (getline(fin, line)) {
		istringstream sin(line);
		vector<string> fields;
		string field;
		while (getline(sin, field, ','))
			fields.push_back(field);	
		for (int i = 0; i < N; i++)
			tmp[i] = stringtold(trim(fields[i]));
		data.push_back(vector<long double>(tmp, tmp + N));
	}
}

void normalization() {
	for (int i = 0; i < N; i++) {
		long double tot = 0;
		for (int j = 0; j < M; j++)
			tot += data[j][i];
		fea_mu[i] = tot / M;
		tot = 0;
		for (int j = 0; j < M; j++)
			tot += (data[j][i] - fea_mu[i]) * (data[j][i] - fea_mu[i]);
		fea_sigma[i] = sqrt(tot / (M - 1));
		for (int j = 0; j < M; j++)
			data[j][i] = (data[j][i] - fea_mu[i]) / fea_sigma[i];
	}
}

vector<long double> vec_add(const vector<long double>& v1, const vector<long double>& v2) {
	vector<long double> tmp(N, 0);
	for (int i = 0; i < N; i++)
		tmp[i] = v1[i] + v2[i];
	return tmp;
}

vector<long double> vec_div(const vector<long double>& v, long double num) {
	vector<long double> tmp(N, 0);
	for (int i = 0; i < N; i++)
		tmp[i] = v[i] / num;
	return tmp;
}

bool equals() {
	for (int i = 0; i < K; i++)
		for (int j = 0; j < N; j++)
			if (fabs(mu[i][j] - old_mu[i][j]) > 1e-6)
				return false;
	return true;
}

void assignment() {
	for (int i = 0; i < K; i++)
		for (int j = 0; j < N; j++)
			old_mu[i][j] = mu[i][j];
}

long double cal_dis(const vector<long double> &v1, const vector<long double> &v2) {
	long double res = 0;
	for (int i = 0; i < N; i++)
		res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	return res;
}

int main(int argc, char** argv) {
	read_data();
	normalization();
	initialize();
	while (!equals()) {
		assignment();
		// tag or samples
		for (int i = 0; i < M; i++) {
			long double min_dis = 90000;
			for (int j = 0; j < K; j++) {
				long double dis = cal_dis(data[i], mu[j]);
				if (dis < min_dis) {
					min_dis = dis;
					tag[i] = j;
				}
			}

		}
		
		for (int j = 0; j < M; j++)
			cout << tag[j];
		cout << endl;
		
		// update centroids
		for (int i = 0; i < K; i++) {
			long double cnt = 0;
			vector<long double> tmp(N, 0);
			for (int j = 0; j < M; j++) {
				if (tag[j] == i) {
					cnt++;
					tmp = vec_add(tmp, data[j]);
				}
			}
			tmp = vec_div(tmp, cnt);
			for (int j = 0; j < N; j++)
				mu[i][j] = tmp[j];
		}
		
		for (int j = 0; j < K; j++) {
			for (int k = 0; k < N; k++)
				cout << mu[j][k] << "    ";
			cout << endl;
		}
		
	}
	
	cout << endl;
	for (int i = 0; i < M; i++)
		cout << tag[i] << " ";
	return 0;
}
