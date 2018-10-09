#include <vector>
#ifndef _M_H
#define _M_H

using namespace std;

class Matrix {
	
private:
	int row;
	int col;
	double *elem;
	double get_norm(double *, int);
	double normalize(double *, int);
	double product(double *, double *, int);
	void orth(double *, double *, int);
	void svd_decompose(int, vector<vector<double> >&, vector<double>&, vector<vector<double> >&);
	
public:
	Matrix():row(0), col(0), elem(NULL) {};
	Matrix(int, int, double);
	Matrix(int, double);
	Matrix(const vector<double> &);
	Matrix(const vector<vector<double> > &);
	Matrix(int ,int, const vector<vector<double> >&);
	Matrix(const Matrix&);
	~Matrix() {delete []elem;};
	
	void print();
	Matrix transpose();
	Matrix inverse();

	
	void set_elem(int i, int j, double val) {
		elem[col * (i - 1) + (j - 1)] = val;
	};
	double& operator()(int i, int j) {
		return elem[col * (i - 1) + (j - 1)];
	};
	double get_elem(int i, int j) const {
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
	
	void svd(int, Matrix &, Matrix &, Matrix &);
	
	
	Matrix& operator=(const Matrix&);
	Matrix& operator+=(const Matrix&);
	Matrix& operator+=(double);
	Matrix& operator-=(const Matrix&);
	Matrix& operator-=(double);
	Matrix& operator*=(const Matrix&);
	Matrix& operator*=(double);

public:
	friend Matrix operator+(const Matrix&, const Matrix&);
	friend Matrix operator+(const Matrix&, double);
	friend Matrix operator+(double, const Matrix&);	
	
	friend Matrix operator-(const Matrix&, const Matrix&);
	friend Matrix operator-(const Matrix&, double);
	friend Matrix operator-(double, const Matrix&);
	
	friend Matrix operator*(const Matrix&, const Matrix&);
	friend Matrix operator*(const Matrix&, double);
	friend Matrix operator*(double, const Matrix&);	
};
#endif
