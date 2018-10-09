#include <vector>
#ifndef _M_H
#define _M_H

using namespace std;

class Matrix {
	
private:
	int row;
	int col;
	long double *elem;
	
public:
	Matrix():row(0), col(0), elem(NULL) {};
	Matrix(int, int, long double);
	Matrix(int, long double);
	Matrix(int ,int, const vector<vector<long double> >&);
	Matrix(const Matrix&);
	~Matrix() {delete []elem;};
	
	void print();
	Matrix transpose();
	Matrix inverse();

	
	void set_elem(int i, int j, long double val) {
		elem[col * (i - 1) + (j - 1)] = val;
	};
	long double& operator()(int i, int j) {
		return elem[col * (i - 1) + (j - 1)];
	};
	long double get_elem(int i, int j) const {
		return elem[col * (i - 1) + (j - 1)];
	};
	int get_row() const {
		return this->row;
	};
	int get_col() const {
		return this->col;
	};
	int get_size() const {
		return this->row * this->col;
	};
	
	Matrix& operator=(const Matrix&);
	Matrix& operator+=(const Matrix&);
	Matrix& operator+=(long double);
	Matrix& operator-=(const Matrix&);
	Matrix& operator-=(long double);
	Matrix& operator*=(const Matrix&);
	Matrix& operator*=(long double);

public:
	friend Matrix operator+(const Matrix&, const Matrix&);
	friend Matrix operator+(const Matrix&, long double);
	friend Matrix operator+(long double, const Matrix&);	
	
	friend Matrix operator-(const Matrix&, const Matrix&);
	friend Matrix operator-(const Matrix&, long double);
	friend Matrix operator-(long double, const Matrix&);
	
	friend Matrix operator*(const Matrix&, const Matrix&);
	friend Matrix operator*(const Matrix&, long double);
	friend Matrix operator*(long double, const Matrix&);	
};
#endif
