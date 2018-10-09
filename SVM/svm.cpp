#include "svm.h"
#include <bits/stdc++.h>
using namespace std;


double SMO::ui(int k) {
	double s = 0;
	for(int i = 0; i < train_num; i++)
		if(alpha[i] > 0)
			s += alpha[i] * target_label[i] * kernel(i,k);
	s -= neg_b;
	return s;
}
 

int SMO::predict(double *array, int length) {
	double s = 0;
	for(int i = 0; i < train_num; i++)
		if(alpha[i] > 0)
			s += alpha[i] * target_label[i] * kernel(i,array);
	s -= neg_b;
	return s > 0 ? 1 : -1;
}
 

double SMO::dotProduct(int i1,int i2) {
	double dot = 0;
	for(int i = 0; i < dimension; i++)
		dot += all_data[i1][i] * all_data[i2][i];
	return dot;
}
 
//��˹�˺��� ||(a-b)||^2 = ||a||^2 + ||b||^2 - 2ab
double SMO::kernel(int i1, int i2) {
	double s = dotProduct(i1, i2);
	s *= -2;
	s += dot_prod_cache[i1] + dot_prod_cache[i2];
	return exp(-s / two_sigma_squared);
}
 
double SMO::kernel(int i1, double *inputData) {
	double s = 0, input_dot_prod = 0, i1_dot_prod = 0;
	for (int i = 0; i < dimension; i++) {
	   input_dot_prod += all_data[i1][i] * all_data[i1][i];
	   i1_dot_prod += inputData[i] * inputData[i];
	   s += all_data[i1][i] * inputData[i];
	}
	s *= -2;
	s += input_dot_prod + i1_dot_prod;
	return exp(-s / two_sigma_squared);
}
 
//�Ż�������������ϵ��
bool SMO::takeStep(int i1, int i2) {
	if ( i1 == i2 ) 
		return 0;
 
	double s  = 0;
	double E1 = 0, E2 = 0;
	double L  = 0, H = 0;
	double k11, k12, k22;
	double eta = 0;
	double a1, a2;
	double Lobj, Hobj;
 
	double alpha1 = alpha[i1];
	double alpha2 = alpha[i2];
	double y1 = target_label[i1];
	double y2 = target_label[i2];
 
	if( error_cache[i1] > 0 && error_cache[i1] < big_C )
		E1 = error_cache[i1];
	else
		E1 = ui(i1) - y1;
 
	s = y1 * y2;
 
    //Compute L, H
	if( y1 == y2 ) {
		L = MAX(alpha2 + alpha1 - big_C , 0);
		H = MIN(alpha1 + alpha2, big_C);
	}
	else {
		L = MAX(alpha2 - alpha1, 0);
		H = MIN(big_C, big_C + alpha1 + alpha2);
	}
 
	if (L == H) 
		return 0;
 
	k11 = kernel(i1,i1);
	k12 = kernel(i1,i2);
	k22 = kernel(i2,i2);
 
	eta = k11 + k22 - 2 * k12;
   
	if (eta > 0) {
		a2 = alpha2 + y2 * (E1 - E2) / eta;
		if (a2 < L) a2 = L;
		else if (a2 > H) a2 = H;
	}
	else {
		double f1 = y1 * (E1 + neg_b) - alpha1 * k11 - s * alpha2 * k12;
		double f2 = y2 * (E2 + neg_b) - s * alpha1 * k12 - alpha2 * k22;
		double L1 = alpha1 + s * (alpha2 - L);
		double H1 = alpha1 + s * (alpha2 - H);
        Lobj = L1 * f1 + L * f2 + (L1*L1*k11 + L*L*k22)/2 + s*L*L1*k12;
		Hobj = H1 * f1 + H * f2 + (H1*H1*k11 + H*H*k22)/2 + s*H*H1*k12;
       
		if (Lobj < Hobj - eps)
			a2 = L;
		else if (Lobj > Hobj + eps)
			a2 = H;
		else 
			a2 = alpha2;
	}
 
	if (abs(a2-alpha2) < eps*(a2+alpha2+eps))
		return 0;
 
	a1 = alpha1 + s * (alpha2 - a2);
 
	//Update threshold to reflect change in Lagrange multipliers 
    double b1 = E1 + y1*(a1-alpha1)*k11 + y2*(a2-alpha2)*k12 + neg_b;
	double b2 = E2 + y1*(a1-alpha1)*k12 + y2*(a2-alpha2)*k22 + neg_b;
	double delta_b = neg_b;
	neg_b = (b1 + b2) / 2.0;
	delta_b = neg_b - delta_b;
	
	//Update error cache using new Lagrange multipliers
	double t1 = y1 * (a1 - alpha1);
	double t2 = y2 * (a2 - alpha2);
	for(int i = 0; i < train_num; i++)
		if(alpha[i] > 0 && alpha[i] < big_C)
			error_cache[i] += t1 * kernel(i1,i) + t2 * (kernel(i2,i)) - delta_b;
	error_cache[i1] = 0;
	error_cache[i2] = 0;
	
	//Store a1,a2 in the alpha array
    alpha[i1] = a1;
	alpha[i2] = a2;
 
	return 1;
 
}
 
//ʹ������ʽ�ķ�����ʵ��inner loop��ѡ��ڶ�������
int SMO::examineExample(int i1) {
	double y1 = target_label[i1];
	double alpha1 = alpha[i1];
	double E1;
 
	if (error_cache[i1] > 0 && error_cache[i1] < big_C)
		E1 = error_cache[i1];
	else
		E1 = ui(i1) - y1;
 
	double r1 = E1 * y1;
 
	if ((r1 < - m_dT && alpha1 < big_C) || (r1 > m_dT && alpha1 > 0)) {
	/*
		ʹ�����ַ���ѡ��ڶ������� 
		1����non-bound������Ѱ��maximum fabs(E1-E2)������ 
		2���������ûȡ�ý�չ,��ô�����λ�ò���non-boundary ���� 
		3���������Ҳʧ�ܣ�������λ�ò�����������,��Ϊbound���� 
		*/
		if(examineFirstChoice(i1,E1))  return 1;
     	if(examineNonBound(i1))  return 1; 
      	if(examineBound(i1))  return 1;
	}
	return 0;
 
}
 
 
//1����non-bound������Ѱ��maximum fabs(E1-E2)������ 
bool SMO::examineFirstChoice(int i1, double E1) {
	int k, i2;
	double tmax;
	double E2, temp;
	for(i2 = - 1,tmax = 0,k = 0; k < train_num; k++) {
		if(alpha[k] > 0 && alpha[k] < big_C) {
			E2 = error_cache[k];
			temp = fabs(E1 - E2);
			if(temp > tmax) {
				tmax = temp;
				i2 = k;
			}
		}
	}
	if(i2 >= 0 && takeStep(i1,i2))  return true;
	return false;
}
 
//2.�������ûȡ�ý�չ,��ô�����λ�ò���non-boundary���� 
bool SMO::examineNonBound(int i1) {
	int k0 = rand() % train_num;
	int k, i2;
	for(k = 0; k < train_num; k++) {
		i2 = (k + k0) % train_num;
		if((alpha[i2] > 0 && alpha[i2] < big_C) && takeStep(i1,i2))  return true;
	}
	return false;
}
 
//  3���������Ҳʧ�ܣ�������λ�ò�����������,(��Ϊbound����) 
bool SMO::examineBound(int i1) {
	int k0 = rand() % train_num;
	int k, i2;
 
	for(k = 0; k < train_num; k++) {
		i2 = (k + k0) % train_num;
		if(takeStep(i1,i2)) return true;
	}
	return false;
}
 

void SMO::train(const char *inputDataPath, const SMOParams &s) {
	if (is_load) {
		cerr << "�������Ѿ��õ���������ѵ����" << endl;
		return;
	}
 
	init(s);
	readFile(inputDataPath);
 
	//����Ԥ����������ѵ�����������ã����ڲ�������ҲҪ���ǣ� 
	for(int i = 0; i < sample_num; i++)  
		dot_prod_cache[i] = dotProduct(i,i);
	outerLoop();
}
 

void SMO::readFile(const char* filePath) {
	ifstream f(filePath);
	if(!f) {
		cerr << "ѵ�����ݶ���ʧ�ܣ�" << endl;
		exit(1);
	}
	int i = 0, j = 0;
	int k;
	int num;  //���ݱ�ŵĶ�ȡ
	char ch;  //�����еġ������Ŷ�ȡ
	int count = 0;
	while(f >> target_label[i]) {
		//if ( i == 270 ) break;
		count++;
		for(k = 1; k <= dimension; k++) {
			f >> num >> ch;
			f>>all_data[i][num-1];
 
			if (num == dimension) 
				break;
			j++;
		}
		i++;
	    if (i >= sample_num)
		   break;
		j = 0;
	}
}

//������������ 
void SMO::error_rate() {
	int ac = 0;
	double accuracy, tar;
	cout << "train over" << endl;
	for(int i = train_num; i < sample_num; i++) {
		tar = ui(i);
		if(tar > 0 && target_label[i] > 0 || tar < 0 && target_label[i] < 0)   ac++;
		//cout<<"The "<<i - train_num + 1<<"th test value is  "<<tar<<endl;
	}
	accuracy = (double)ac / (sample_num - train_num);
	cout << "��ȷ�ȣ�" << accuracy * 100 << "��" << endl;
}
 

void SMO::init(const SMOParams &s) {
	sample_num = s.sample_num;				//���е�������
	train_num = s.train_num;				//ѵ����������
	dimension = s.dimension;			 //���ݵ�ά��
	big_C = s.big_C;				//�ͷ����� 
	m_dT = s.m_dT;				//��KKT���������̷�Χ
	eps = s.eps;               //�������� 
	two_sigma_squared = s.two_sigma_squared;  //�˺����еĲ��� 
 
	is_load = false;
	neg_b = 0.0;

	all_data = new double*[sample_num];
	for ( int i = 0; i < sample_num; i++ )
		all_data[i] = new double[dimension];
 
    for ( int i = 0; i < sample_num; i++ )
		for (int j = 0; j < dimension; j++ )
			all_data[i][j] = 0.0;
 
	target_label.resize(sample_num,0);
	alpha.resize(train_num,0);
	error_cache.resize(train_num,0);
	dot_prod_cache.resize(sample_num,0);
}
 
//Ѱ�ҵ�һ��Ҫ�Ż��ĳ���
void SMO::outerLoop() {
	int numChanged = 0;
	bool examineAll = 1;
 
	while (numChanged > 0 || examineAll) {
		numChanged = 0;
		if (examineAll) {
			for (int i = 0; i < train_num; i++)
				numChanged += examineExample(i);
		}
		else {
			for (int i = 0; i < train_num; i++) {
				if (alpha[i] > 0 && alpha[i] < big_C)
					numChanged += examineExample(i);
			}
		}
		if (examineAll == 1)
			examineAll = 0;
		else if (numChanged == 0)
			examineAll = 1;
	}
}
 
//��֧����������ر�Ҫ����Ϣ��������
void SMO::save() {
	ofstream outfile("svm.txt");
 
	int countVec = 0;   //֧�������ĸ���
	for (int i = 0; i < train_num; i++)
		if (alpha[i] > 0)
			countVec++;
 
	outfile<<countVec<<' '<<dimension<<' '<<two_sigma_squared<<' '<<neg_b<<'\n';
	for (int i = 0; i < train_num; i++) {
		if (alpha[i] > 0) {
			outfile << target_label[i] << ' ' << alpha[i] << ' ';
			for (int j = 0; j < dimension; j++)
				outfile << all_data[i][j] << ' ';
			outfile << '\n';
		}
	}
}
 
//�������ѵ��������������������ؽ��������ڷ���
void SMO::load(const char *filePath) {
	ifstream infile(filePath);
	if(!infile) {
		cerr<<"����������ʧ�ܣ�"<<endl;
		exit(1);
	}
 
	is_load = true;
 
	infile>>sample_num>>dimension>>two_sigma_squared>>neg_b;
	train_num = sample_num;  //Ϊ��ʹ��predict����������
	
 
	all_data = new double*[sample_num];
	for ( int i = 0; i < sample_num; i++ )
		all_data[i] = new double[dimension];
 
	for ( int i = 0; i < sample_num; i++ )
		for (int j = 0; j < dimension; j++ )
			all_data[i][j] = 0.0;
 
	target_label.resize(sample_num,0);
	alpha.resize(sample_num,0);
 
	int i = 0;
	while(infile>>target_label[i]) {
		infile>>alpha[i];
		for (int j = 0; j < dimension; j++)
			infile>>all_data[i][j];
		i++;
		if (i == sample_num) 
			break;
	}
}
 
SMO::~SMO() {
	for (int i = 0; i < sample_num; i++ ) {
		delete [] all_data[i];
	}
	delete []all_data;
}
