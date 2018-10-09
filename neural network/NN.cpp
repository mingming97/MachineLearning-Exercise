#include <bits/stdc++.h>
#include <fstream>
#include "matrix.h"
#include "NN.h"

const int batch_size = 128;
const double learning_rate = 0.1;
const int layer_num = 3;
const int hidden_dim = 32;

using namespace std;

NN::NN(int input_dim, int output_dim) {
	this->input_dim = input_dim;
	this->output_dim = output_dim;
	weights.resize(layer_num);
	biases.resize(layer_num);
	unactivate_layers.resize(layer_num);
	activate_layers.resize(layer_num);
	error.resize(layer_num);
	d_biases.resize(layer_num);
	d_weights.resize(layer_num);
	weights[0] = Matrix(hidden_dim, input_dim);
	d_weights[0] = Matrix(hidden_dim, input_dim, 0);
	weights[layer_num - 1] = Matrix(output_dim, hidden_dim);
	d_weights[layer_num - 1] = Matrix(output_dim, hidden_dim, 0);
	for (int i = 1; i <= layer_num - 2; i++) {
		weights[i] = Matrix(hidden_dim, hidden_dim);
		d_weights[i] = Matrix(hidden_dim, hidden_dim, 0);
	}
	for (int i = 0; i < layer_num - 1; i++) {
		biases[i] = Matrix(hidden_dim, 1, 0);
		d_biases[i] = Matrix(hidden_dim, 1, 0);
		unactivate_layers[i] = Matrix(hidden_dim, 1, 0);
		activate_layers[i] = Matrix(hidden_dim, 1, 0);
		error[i] = Matrix(hidden_dim, 1, 0);
	}
	biases[layer_num - 1] = Matrix(output_dim, 1, 0);
	d_biases[layer_num - 1] = Matrix(output_dim, 1, 0);
	unactivate_layers[layer_num - 1] = Matrix(output_dim, 1, 0);
	activate_layers[layer_num - 1] = Matrix(output_dim, 1, 0);
	error[layer_num - 1] = Matrix(output_dim, 1, 0); 
	cout << "initialize over" << endl;
}

void NN::zero_grad() {
	for (int i = 0; i < d_biases.size(); i++) {
		d_weights[i].zero();
		d_biases[i].zero();
	}		
}

int NN::forward_pass(const Matrix &x) {
	for (int i = 0; i < layer_num; i++) {
		if (i == 0) unactivate_layers[i] = weights[i] * x + biases[i];	
		else unactivate_layers[i] = weights[i] * activate_layers[i - 1] + biases[i];
		activate_layers[i] = unactivate_layers[i];
		if (i != layer_num - 1) activate_layers[i].sigmoid();
		else activate_layers[i].softmax();
	}
	return activate_layers[layer_num - 1].argmaxx().first;
}

void NN::backward_pass(const Matrix &x, const Matrix &y) {
	error[layer_num - 1] = activate_layers[layer_num - 1] - y;
	for (int i = layer_num - 2; i >= 0; i--) {
		Matrix tmp = activate_layers[i].hadamard_mul(1 - activate_layers[i]);
//		cout << i << endl;
//		tmp.print();
		error[i] = tmp.hadamard_mul(weights[i + 1].transpose() * error[i + 1]);
	}

	for (int i = 0; i < layer_num; i++) {
		if (i == 0) d_weights[i] += error[i] * x.transpose();
		else d_weights[i] += error[i] * (activate_layers[i - 1].transpose());
		d_biases[i] += error[i];
	}
}

void NN::update() {
	for (int i = 0; i < d_biases.size(); i++) {
		d_weights[i] *= (1.0 / batch_size);
//		d_weights[i].print();
		weights[i] -= learning_rate * d_weights[i];
		d_biases[i] *= (1.0 / batch_size);
		biases[i] -= learning_rate * d_biases[i];
	}
}
void NN::write_matrix(ofstream &os, const Matrix &mat) {
	int m = mat.get_row();
	int n = mat.get_col();
	os << m << " " << n << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			os << mat.get_elem(i + 1, j + 1) << " ";
		os << endl;
	}
}

void NN::read_matrix(ifstream &is, Matrix &mat) {
	int m, n;
	is >> m >> n;
	cout << m << "  " << n << endl;
	vector<vector<double> > tmp;
	vector<double> line; 
	for (int i = 0; i < m; i++) {
		line.clear();
		for (int j = 0; j < n; j++) {
			double p;
			is >> p;
			line.push_back(p);
		}
		tmp.push_back(line);	
	}
	mat = Matrix(tmp);
}

void NN::save() {
	ofstream os;
	os.open("./myweights", ios::trunc);
	for (int i = 0; i < layer_num; i++) {
		write_matrix(os, weights[i]);
		write_matrix(os, biases[i]);
	}
}

void NN::load() {
	ifstream is;
	is.open("./myweights", ios::in);
	for (int i = 0; i < layer_num; i++) {
		read_matrix(is, weights[i]);
		read_matrix(is, biases[i]);
	}
}

