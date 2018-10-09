#include <bits/stdc++.h>
#include "matrix.h"
#include "NN.h"

using namespace std;
 
int ReverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
 
void read_Mnist_Label(string filename, vector<double>&labels) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		
		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels.push_back((double)label);
		}
		
	}
}
 
void read_Mnist_Images(string filename, vector<vector<double>>&images) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);
 
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;
 
		for (int i = 0; i < number_of_images; i++) {
			vector<double>tp;
			for (int r = 0; r < n_rows; r++) {
				for (int c = 0; c < n_cols; c++) {
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					tp.push_back(image* 1.0/ 255);
				}
			}
			images.push_back(tp);
		}
	}
}

void get_onehot(const vector<double> &nums, vector<vector<double> > &one_hot) {
	one_hot.resize(nums.size(), vector<double>(10, 0));
	for (int i = 0; i < nums.size(); i++)
		one_hot[i][nums[i]] = 1;
}

void train() {
	vector<vector<double>> images;
	vector<double> tmp;
	vector<vector<double>> labels;
	read_Mnist_Label("train-labels.idx1-ubyte", tmp);
	get_onehot(tmp, labels);
	read_Mnist_Images("train-images.idx3-ubyte", images);
	
	NN nn(784, 10);
	int data_size = images.size();
	for (int i = 0; i < 2000; i++) {
		int ans = 0;
		nn.zero_grad();
		for (int j = 0; j < 128; j++) {
			int pred, idx = (i * 128 + j) % data_size;
			pred = nn.forward_pass(Matrix(images[idx]));
			if (labels[idx][pred] == 1) ans++;
			nn.backward_pass(Matrix(images[idx]), Matrix(labels[idx]));
		}
		nn.update();
		cout << "accuracy:   " << ans * 1.0 / 128 << endl;
	}
	nn.save();
}

void test() {
	vector<vector<double>> images;
	vector<double> tmp;
	vector<vector<double>> labels;
	read_Mnist_Label("t10k-labels.idx1-ubyte", tmp);
	get_onehot(tmp, labels);
	read_Mnist_Images("t10k-images.idx3-ubyte", images);
	NN nn(784, 10);
	nn.load();
	int ans = 0;
	for (int i = 0; i < 9999; i++) {
		int	pred = nn.forward_pass(Matrix(images[i]));
		if (labels[i][pred] == 1) ans++;
	}
	cout << "accuracy:   " << ans * 1.0 / 9999 << endl;
} 
int main()
{
//	train();
	test();
	return 0;
}
