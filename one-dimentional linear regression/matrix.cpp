#include <bits/stdc++.h>
#include "matrix.h"

using namespace std;

Matrix::Matrix(int i, int j, long double val = 0) {
	row = i;
	col = j;
	elem = new long double[i * j];
	for (int ii = 0; ii < i * j; ii++)
		elem[ii] = val;
}

Matrix::Matrix(int rows, long double val = 1) {
	row = col = rows;
	elem = new long double[rows * rows];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < rows; j++)
			elem[i * col + j] = (i == j ? val : 0);
}

Matrix::Matrix(int i, int j, const vector<vector<long double> >& v) {
	row = i;
	col = j;
	elem = new long double[i * j];
	for (int ii = 0; ii < row; ii++)
		for (int jj = 0; jj < col; jj++)
			elem[ii * col + jj] = v[ii][jj];
}

Matrix::Matrix(const Matrix& m) {
	row = m.row;
	col = m.col;
	elem = new long double[row * col];
	for (int i = 0; i < row * col; i++)
		elem[i] = m.elem[i];
}

void Matrix::print() {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++)
			cout << elem[i * col + j] << " ";
		cout << endl;
	}
}

Matrix Matrix::transpose() {
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
	Matrix I(row);
	long double maxx = get_elem(1, 1);
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
				long double tmp;
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
			long double times = get_elem(j + 1, i + 1) / maxx;
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
		long double times = get_elem(i + 1, col) / maxx;
		for (int j = 0; j < col; j++) {
			set_elem(i + 1, j + 1, get_elem(i + 1, j + 1) - times * get_elem(row, j + 1));
			I.set_elem(i + 1, j + 1, I.get_elem(i + 1, j + 1) - times * I.get_elem(row, j + 1));
		}	
	}
	for (int i = 0; i < row; i++) {
		long double times = get_elem(i + 1, i + 1);
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
	elem = new long double[row * col];
	for (int i = 0; i < row * col; i++)
		elem[i] = m.elem[i];
	return *this;
}

Matrix& Matrix::operator+=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		cout << "Cannot add! row or col is not the same." << endl;
	}
	else {
		for (int i = 0; i < row * col; i++)
			this->elem[i] += m.elem[i];
	}
	return *this;
}

Matrix& Matrix::operator+=(long double val) {
	for (int i = 0; i < row * col; i++)
		this->elem[i] += val;
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		cout << "Cannot subtract! row or col is not the same." << endl;
	}
	else {
		for (int i = 0; i < row * col; i++)
			this->elem[i] -= m.elem[i];
	}
	return *this;
}

Matrix& Matrix::operator-=(long double val) {
	for (int i = 0; i < row * col; i++)
		this->elem[i] -= val;
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& m) {
	if (col != m.row) {
		cout << "Cannot multiply!" << endl;
	}
	else {
		long double tmp_C[row * m.col];
		for (int i = 0; i < row; i++)
			for (int j = 0; j < m.col; j++) {
				long double tmp = 0;
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

Matrix& Matrix::operator*=(long double val) {
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

Matrix operator+(const Matrix& m1, long double val) {
	Matrix tmp(m1);
	for (int i = 0; i < m1.row * m1.col; i++)
		tmp.elem[i] += val;
	return tmp;
}

Matrix operator+(long double val, const Matrix& m) {
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

Matrix operator-(const Matrix& m1, long double val) {
	Matrix tmp(m1);
	for (int i = 0; i < m1.row * m1.col; i++)
		tmp.elem[i] -= val;
	return tmp;
}

Matrix operator-(long double val, const Matrix& m) {
	return m - val;
}

Matrix operator*(const Matrix& m1, const Matrix& m2) {
	Matrix tmp(m1.row, m2.col);
	if (m1.col != m2.row) {
		cout << "Cannot multiply!" << endl;
	}
	else {
		for (int i = 0; i < tmp.row; i++)
			for (int j = 0; j < tmp.col; j++) {
				long double temp = 0;
				for (int k = 0; k < m1.col; k++)
					temp += m1.get_elem(i + 1, k + 1) * m2.get_elem(k + 1, j + 1);
				tmp.set_elem(i + 1, j + 1, temp);
			}
	}
	return tmp;
}

Matrix operator*(const Matrix& m1, long double val) {
	Matrix tmp(m1);
	for (int i = 0; i < m1.row * m1.col; i++)
		tmp.elem[i] *= val;
	return tmp;
}

Matrix operator*(long double val, const Matrix& m) {
	return m * val;
}



