#include <bits/stdc++.h>
#include "matrix.h"

using namespace std;
const int MAX_ITER = 100000;
const double eps = 1e-7;

Matrix::Matrix(int i, int j, double val) {
	row = i;
	col = j;
	elem = new double[i * j];
	for (int ii = 0; ii < i * j; ii++)
		elem[ii] = val;
}

Matrix::Matrix(int i, int j) {
	row = i;
	col = j;
	elem = new double[i * j];
	srand(time(NULL));
	for (int ii = 0; ii < i * j; ii++)
		elem[ii] = pow(-1, rand()%2+1)*rand()/double(RAND_MAX);
}

Matrix::Matrix(int rows, double val) {
	row = col = rows;
	elem = new double[rows * rows];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < rows; j++)
			elem[i * col + j] = (i == j ? val : 0);
}

Matrix::Matrix(const vector<double> &v) {
	row = v.size();
	col = 1;
	elem = new double[row];
	for (int i = 0; i < row; i++)
		elem[i] = v[i];
}

Matrix::Matrix(int i, int j, const vector<vector<double> >& v) {
	row = i;
	col = j;
	elem = new double[i * j];
	for (int ii = 0; ii < row; ii++)
		for (int jj = 0; jj < col; jj++)
			elem[ii * col + jj] = v[ii][jj];
}

Matrix::Matrix(const vector<vector<double> >& v) {
	row = v.size();
	col = v[0].size();
	elem = new double[row * col];
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			elem[i * col + j] = v[i][j];
}

Matrix::Matrix(const Matrix& m) {
	row = m.row;
	col = m.col;
	elem = new double[row * col];
	for (int i = 0; i < row * col; i++)
		elem[i] = m.elem[i];
}

void Matrix::print() const{
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++)
			cout << elem[i * col + j] << " ";
		cout << endl;
	}
	cout << endl;
}

Matrix Matrix::transpose() const{
	Matrix tmp(col, row);
	for (int i = 0; i < tmp.row; i++) 
		for (int j = 0; j < tmp.col; j++)
			tmp.elem[i * tmp.col + j] = elem[j * col + i];		
	return tmp;
}

Matrix Matrix::inverse() {
	if (row != col) {
		cout << "Cannot inverse" << endl;
		return Matrix();
	}
	Matrix I(row, 1);
	double maxx = get_elem(1, 1);
	int max_pos = 0;
	for (int i = 0; i < row - 1; i++) {
		maxx = get_elem(i + 1, i + 1);
		max_pos = 0;
		for (int j = i + 1; j < row; j++) {
			if (abs(maxx) < abs(get_elem(j + 1, i + 1))) {
				maxx = get_elem(j + 1, i + 1);
				max_pos = j;
			}
		}
		if (maxx == 0) {
			cout << "Cannot inverse" << endl;
			return Matrix();
		}
		if (max_pos != i) {
			for (int j = 0; j < col; j++) {
				double tmp;
				tmp = get_elem(max_pos + 1, j + 1);
				set_elem(max_pos + 1, j + 1, get_elem(i + 1, j + 1));
				set_elem(i + 1, j + 1, tmp);
				tmp = I.get_elem(max_pos + 1, j + 1);
				I.set_elem(max_pos + 1, j + 1, I.get_elem(i + 1, j + 1));
				I.set_elem(i + 1, j + 1, tmp);
			}
		}
		
		for (int j = 0; j < row; j++) {
			if (j == i) continue;
			if (get_elem(j + 1, i + 1) == 0) continue;
			double times = get_elem(j + 1, i + 1) / maxx;
			for (int k = 0; k < col; k++) {
				set_elem(j + 1, k + 1, get_elem(j + 1, k + 1) - times * get_elem(i + 1, k + 1));
				I.set_elem(j + 1, k + 1, I.get_elem(j + 1, k + 1) - times * I.get_elem(i + 1, k + 1));
			}
		}
	} 
	maxx = get_elem(row, row);
	for (int i = 0; i < row - 1; i++) {
		if (abs(get_elem(i + 1, col)) < 1.0E-016)
			continue;
		double times = get_elem(i + 1, col) / maxx;
		for (int j = 0; j < col; j++) {
			set_elem(i + 1, j + 1, get_elem(i + 1, j + 1) - times * get_elem(row, j + 1));
			I.set_elem(i + 1, j + 1, I.get_elem(i + 1, j + 1) - times * I.get_elem(row, j + 1));
		}	
	}
	for (int i = 0; i < row; i++) {
		double times = get_elem(i + 1, i + 1);
		for (int j = 0; j < col; j++)
			I.set_elem(i + 1, j + 1, I.get_elem(i + 1, j + 1) / times);
	}
	return I;
}

Matrix& Matrix::operator=(const Matrix& m) {
	if (this == &m)
		return *this;
	row = m.row;
	col = m.col;
	delete [] elem;
	elem = new double[row * col];
	for (int i = 0; i < row * col; i++)
		elem[i] = m.elem[i];
	return *this;
}

Matrix& Matrix::operator+=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		cout << "Cannot add! row or col is not the same." << endl;
	} else {
		for (int i = 0; i < row * col; i++)
			this->elem[i] += m.elem[i];
	}
	return *this;
}

Matrix& Matrix::operator+=(double val) {
	for (int i = 0; i < row * col; i++)
		this->elem[i] += val;
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		cout << "Cannot subtract! row or col is not the same." << endl;
	} else {
		for (int i = 0; i < row * col; i++)
			this->elem[i] -= m.elem[i];
	}
	return *this;
}

Matrix& Matrix::operator-=(double val) {
	for (int i = 0; i < row * col; i++)
		this->elem[i] -= val;
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& m) {
	if (col != m.row) {
		cout << "Cannot multiply! in *=" << endl;
	} else {
		double tmp_C[row * m.col];
		for (int i = 0; i < row; i++)
			for (int j = 0; j < m.col; j++) {
				double tmp = 0;
				for (int k = 0; k < col; k++)
					tmp += this->get_elem(i + 1, k + 1) * m.get_elem(k + 1, j + 1);
				tmp_C[i * m.col + j] = tmp;
			}
		for (int i = 0; i < row * col; i++)
			this->elem[i] = tmp_C[i];
	}
	this->col = m.col;
	return *this;
}

Matrix& Matrix::operator*=(double val) {
	for (int i = 0; i < row * col; i++)
		this->elem[i] *= val;
	return *this;
}


Matrix operator+(const Matrix& m1, const Matrix& m2) {
	Matrix tmp(m1);
	if (m1.row != m2.row || m1.col != m2.col) {
		cout << "Cannot add!" << endl;
	}
	else {
		for (int i = 0; i < m1.row * m1.col; i++)
			tmp.elem[i] += m2.elem[i];
	}
	return tmp;
}

Matrix operator+(const Matrix& m1, double val) {
	Matrix tmp(m1);
	for (int i = 0; i < m1.row * m1.col; i++)
		tmp.elem[i] += val;
	return tmp;
}

Matrix operator+(double val, const Matrix& m) {
	return m + val;
}

Matrix operator-(const Matrix& m1, const Matrix& m2) {
	Matrix tmp(m1);
	if (m1.row != m2.row || m1.col != m2.col) {
		cout << "Cannot subtract!" << endl;
	}
	else {
		for (int i = 0; i < m1.row * m1.col; i++)
			tmp.elem[i] -= m2.elem[i];
	}
	return tmp;
}

Matrix operator-(const Matrix& m, double val) {
	Matrix tmp(m);
	for (int i = 0; i < m.row * m.col; i++)
		tmp.elem[i] -= val;
	return tmp;
}

Matrix operator-(double val, const Matrix& m) {
	Matrix tmp(m);
	for (int i = 0; i < m.row * m.col; i++)
		tmp.elem[i] = val - tmp.elem[i];
	return tmp;
}

Matrix operator*(const Matrix& m1, const Matrix& m2) {
	Matrix tmp(m1.row, m2.col, 0);
	if (m1.col != m2.row) {
		cout << "Cannot multiply!" << endl;
	}
	else {
		for (int i = 0; i < tmp.row; i++) {
			for (int j = 0; j < tmp.col; j++) {
				double temp = 0;
				for (int k = 0; k < m1.col; k++)
					temp += m1.get_elem(i + 1, k + 1) * m2.get_elem(k + 1, j + 1);
				tmp.elem[i * tmp.col + j] = temp;
			}
		}
	}
	return tmp;
}

Matrix operator*(const Matrix& m1, double val) {
	Matrix tmp(m1);
	for (int i = 0; i < m1.row * m1.col; i++)
		tmp.elem[i] *= val;
	return tmp;
}

Matrix operator*(double val, const Matrix& m) {
	return m * val;
}

double Matrix::get_norm(double *x, int n){
    double r = 0;
    for(int i = 0; i < n; i++)
        r += x[i] * x[i];
    return sqrt(r);
}

double Matrix::normalize(double *x, int n){
    double r = get_norm(x, n);
    if (r < eps)
        return 0;
    for(int i = 0; i < n; i++)
        x[i] /= r;
    return r;
}

double Matrix::product(double *a, double *b, int n) {
    double r = 0;
    for(int i = 0; i < n; i++)
        r += a[i] * b[i];
    return r;
}

void Matrix::orth(double *a, double *b, int n) {//|a|=1
    double r = product(a, b, n);
    for(int i = 0; i < n; i++)
        b[i] -= r * a[i];
}

void Matrix::svd_decompose(int K, vector<vector<double> > &U, vector<double> &S, vector<vector<double> > &V){
	int M = row;
    int N = col;
    U.clear();
    V.clear();
    S.clear();
    S.resize(K, 0);
    U.resize(K);
    for(int i = 0; i < K; i++)
        U[i].resize(M, 0);
    V.resize(K);
    for(int i = 0; i < K; i++)
        V[i].resize(N, 0);

    
    srand(time(0));
    double *left_vector = new double[M];
    double *next_left_vector = new double[M];
    double *right_vector = new double[N];
    double *next_right_vector = new double[N];
    for(int col = 0; col < K; col++){
        double diff = 1;
        double r = -1;
        while(1) {
            for(int i = 0; i < M; i++)
                left_vector[i]= (float)rand() / RAND_MAX;
            if(normalize(left_vector, M) > eps)
                break;
        }

        for(int iter = 0; diff >= eps && iter < MAX_ITER; iter++){
            memset(next_left_vector, 0, sizeof(double)*M);
            memset(next_right_vector, 0, sizeof(double)*N);
            for(int i = 0; i < M; i++)
                for(int j = 0; j < N; j++)
                    next_right_vector[j] += left_vector[i] * this->get_elem(i + 1 , j + 1);

            r = normalize(next_right_vector, N);
            if(r < eps) break;
            for(int i = 0; i < col; i++)
                orth(&V[i][0], next_right_vector, N);
            normalize(next_right_vector, N);

            for(int i = 0; i < M; i++)
                for(int j = 0; j < N; j++)
                    next_left_vector[i] += next_right_vector[j] * this->get_elem(i + 1, j + 1);
            r = normalize(next_left_vector, M);
            if(r < eps) break;
            for(int i = 0; i < col; i++)
                orth(&U[i][0], next_left_vector, M);
            normalize(next_left_vector, M);
            diff = 0;
            for(int i = 0; i < M; i++) {
                double d = next_left_vector[i] - left_vector[i];
                diff += d * d;
            }

            memcpy(left_vector, next_left_vector, sizeof(double)*M);
            memcpy(right_vector, next_right_vector, sizeof(double)*N);
        }
        if(r >= eps) {
            S[col] = r;
            memcpy((char *)&U[col][0], left_vector, sizeof(double)*M);
            memcpy((char *)&V[col][0], right_vector, sizeof(double)*N);
        } else {
            cout << r << endl;
            break;
        }
    }
    delete [] next_left_vector;
    delete [] next_right_vector;
    delete [] left_vector;
    delete [] right_vector;
}

void Matrix::svd(int k, Matrix &U, Matrix &S, Matrix &V) {
	vector<vector<double> > UU;
	vector<double> SS;
    vector<vector<double> > VV;
    this->svd_decompose(k, UU, SS, VV);
    U = Matrix(UU).transpose();
    S = Matrix(SS);
    V = Matrix(VV).transpose();
}

void Matrix::zero() {
	for (int i = 0; i < row * col; i++)
		elem[i] = 0; 
}

void Matrix::sigmoid() {
	for (int i = 0; i < row * col; i++) {
		elem[i] = 1 / (1 + exp(-elem[i]));
	}
}

void Matrix::softmax() {
	double ans = 0;
	for (int i = 0; i < row * col; i++)
		ans += exp(elem[i]);
	for (int i = 0; i < row * col; i++)
		elem[i] = exp(elem[i]) / ans;
}

pair<int, double> Matrix::argmaxx() {
	double maxx = -5;
	int res;
	for (int i = 0; i < row; i++)
		if (elem[i] > maxx) {
			maxx = elem[i];
			res = i;
		}
	return make_pair(res, maxx);
};

Matrix Matrix::hadamard_mul(const Matrix &mat) {
	if (mat.col != 1 || col != 1 || mat.row != row) {
		cout << "cant do hadamard" << endl;
		return Matrix();
	}
	vector<vector<double> > tmp;
	for (int i = 0; i < row; i++)
		tmp.push_back(vector<double>(1, elem[i] * mat.elem[i]));
	return Matrix(tmp);
}



