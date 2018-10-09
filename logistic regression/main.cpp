#include <bits/stdc++.h>
#include "matrix.h"
#define M 350

using namespace std;

long double xx[5][M] = {0};
long double yy[M] = {0};
int epoch = 200;
long double lr = 0.01;


long double gre_maxx;
long double gre_minn;
long double gpa_maxx;
long double gpa_minn;


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

int stringtoint(const string &s) {
	int val;
	istringstream ss(s);
	ss >> val;
	return val; 
}

void read_data() {
	ifstream fin("binary.csv");
	string line;
	int cnt = -1;
	while (getline(fin, line)) {
		cnt++;
		if (cnt == 0) continue;
		istringstream sin(line);
		vector<string> fields;
		string field;
		while (getline(sin, field, ','))
			fields.push_back(field);
		long double admit = stringtold(trim(fields[0]));
		long double gre = stringtold(trim(fields[1]));
		long double gpa = stringtold(trim(fields[2]));
		int rank = stringtoint(trim(fields[3]));
		yy[cnt - 1] = admit;
		xx[0][cnt - 1] = gre;
		xx[1][cnt - 1] = gpa;
		if (rank != 1) xx[rank][cnt - 1] = 1;
	}
}

void normalization() {
	for (int j = 0; j < 2; j++) {
		long double maxx = -1;
		long double minn = 90000;
		for (int i = 0; i < M; i++) {
			maxx = max(xx[j][i], maxx);
			minn = min(xx[j][i], minn);
		}
		if (j == 0) {
			gre_maxx = maxx;
			gre_minn = minn;
		}
		else {
			gpa_maxx = maxx;
			gpa_minn = minn;
		}
		for (int i = 0; i < M; i++)
			xx[j][i] = (xx[j][i] - minn) / (maxx - minn);
	}
}

Matrix sigmoid(const Matrix &m) {
	if (m.get_col() != 1) {
		cout << "m's col must be 1" << endl;
		return Matrix();
	}
	vector<vector<long double> > tmp;
	for (int i = 0; i < m.get_row(); i++) {
		long double res = 1 / (1 + exp(-m.get_elem(i + 1, 1)));
		tmp.push_back(vector<long double>(1, res));
	}
	return Matrix(m.get_row(), 1, tmp);
}

long double accuracy(const Matrix &theta, const Matrix &x) {
	Matrix m(M, 1);
	m = sigmoid(x * theta);
	long double cnt = 0;
	for (int i = 0; i < M; i++) {
		if (m(i+1, 1) > 0.5 && yy[i] == 1.0) cnt++;
		if (m(i+1, 1) < 0.5 && yy[i] == 0.0) cnt++;
	}
	return cnt / M;
}

long double htheta(long double x) {
	return 1 / (1 + (exp(-x)));
}

int main(int argc, char** argv) {
	read_data();
	normalization();
	vector<vector<long double> > tmp;
	tmp.push_back(vector<long double>(M, 1));
	for (int i = 0; i < 5; i++)
		tmp.push_back(vector<long double>(xx[i], xx[i] + M));
	Matrix x(6, M, tmp);
	x = x.transpose();
	tmp.clear();
	tmp.push_back(vector<long double>(yy, yy + M));
	Matrix y(1, M, tmp);
	y = y.transpose();
	Matrix theta(6, 1, 0);
	
	for (int i = 0; i < epoch; i++) {
		long double loss = 0;
		for (int j = 0; j < M; j++) {
			vector<long double> ele;
			vector<vector<long double> > temp;
			for (int k = 0; k < 6; k++)
				ele.push_back(x.get_elem(j + 1, k + 1)); 
			temp.push_back(ele);
			Matrix t(1, 6, temp);
			loss += y.get_elem(j + 1, 1) * log(htheta((t * theta).get_elem(1, 1))) + 
					(1 - y.get_elem(j + 1, 1)) * log(1 - htheta((t * theta).get_elem(1, 1)));
		}
		cout << loss << endl;
		theta = theta + lr * x.transpose() * (y - sigmoid(x * theta));
	}
	cout << endl;
	cout << "theta is : " <<endl;
	theta.print();
	cout << "gre_max, gre_min" << endl;
	cout << gre_maxx << " " << gre_minn << endl;
	cout << "gpa_max, gpa_min" << endl;
	cout << gpa_maxx << " " << gpa_minn << endl;
	cout << "train accuracy: " << accuracy(theta, x) << endl;
	return 0;
}
