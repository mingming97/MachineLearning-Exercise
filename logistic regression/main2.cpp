#include <iostream>
#include <bits/stdc++.h>
#include "matrix.h"
#define N 50
using namespace std;

long double thetaa[6] = {-1.84562, 1.32959, 1.55533, -0.59802, -1.4063, -1.47568};
long double gre_max = 800, gre_min = 220, gpa_max = 4, gpa_min = 2.26;
long double test_xx[5][N] = {0};
long double test_yy[N];

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
	ifstream fin("test.csv");
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
		test_yy[cnt - 1] = admit;
		test_xx[0][cnt - 1] = gre;
		test_xx[1][cnt - 1] = gpa;
		if (rank != 1) test_xx[rank][cnt - 1] = 1;
	}
}

void normalization() {
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < N; i++) {	
			if (j == 0) test_xx[j][i] = (test_xx[j][i] - gre_min) / (gre_max - gre_min);
			if (j == 1) test_xx[j][i] = (test_xx[j][i] - gpa_min) / (gpa_max - gpa_min);
		}		
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
	Matrix m(N, 1);
	m = sigmoid(x * theta);
	long double cnt = 0;
	for (int i = 0; i < N; i++) {
		if (m(i+1, 1) > 0.5 && test_yy[i] == 1.0) cnt++;
		if (m(i+1, 1) < 0.5 && test_yy[i] == 0.0) cnt++;
	}
	return cnt / N;
}

int main(int argc, char** argv) {
	read_data();
	normalization();
	
	vector<vector<long double> > tmp;
	tmp.push_back(vector<long double>(N, 1));
	for (int i = 0;i < 5; i++)
		tmp.push_back(vector<long double>(test_xx[i], test_xx[i] + N));
	Matrix x(6, N, tmp);
	x = x.transpose();
	tmp.clear();
	tmp.push_back(vector<long double>(test_yy, test_yy + N));
	Matrix y(1, N, tmp);
	y = y.transpose();
	tmp.clear();
	tmp.push_back(vector<long double>(thetaa, thetaa + 6));
	Matrix theta(1, 6, tmp);
	theta = theta.transpose();
	theta.print();
	cout << "accuracy on validation test is " << accuracy(theta, x) << endl;
	return 0;
}
