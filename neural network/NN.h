#include <bits/stdc++.h>
#include "matrix.h"
#ifndef _NN_H
#define _NN_H
using namespace std;

class NN {
private:
	vector<Matrix> weights;
	vector<Matrix> biases;
	vector<Matrix> unactivate_layers;
	vector<Matrix> activate_layers;
	vector<Matrix> error;
	vector<Matrix> d_biases;
	vector<Matrix> d_weights;
	int input_dim;
	int output_dim;
	void write_matrix(ofstream &, const Matrix &);
	void read_matrix(ifstream &, Matrix &);
public:
	NN() {}
	NN(int input_dim, int output_dim); 
	void zero_grad();
	int forward_pass(const Matrix &x);
	void backward_pass(const Matrix &x, const Matrix &y);
	void update();
	void save();
	void load(); 
};
#endif
