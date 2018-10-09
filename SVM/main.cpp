#include "svm.h"
#include <iostream>
using namespace std;
 
int main() {
	SMOParams params;
	params.sample_num = 270;				//���е�������
	params.train_num = 200;				//ѵ����������
	params.dimension = 13;			 //���ݵ�ά��
	params.big_C = 1.0;				//�ͷ����� 
	params.m_dT = 0.001;				//��KKT���������̷�Χ
	params.eps = 1.0E-12;               //�������� 
	params.two_sigma_squared = 2.0;  //RBF�˺����еĲ��� 
 
 
	SMO tmp;
	tmp.train("heart_scale.txt", params);
	tmp.error_rate();
	tmp.save();
 
//	tmp.load("svm.txt");
//	//���������heart_scale�ĵ�17�У���Ӧ�����Ӧ����+1
//	double mydata[] = {-0.291667, 1, 1, -0.132075, -0.155251, -1, -1, -0.251908, 1, -0.419355, 0, 0.333333, 1};
//	//���������heart_scale�ĵ�264�У���Ӧ�����Ӧ����-1
//	double mydata2[] = { -0.166667, 1, -0.333333, -0.320755, -0.360731, -1, -1, 0.526718, -1, -0.806452, -1, -1, -1,};
//	cout<<"Ԥ�������ǣ�"<< tmp.predict(mydata,13)<<endl;
//	cout<<"��ʵ������ǣ� 1"<<endl;
//	cout<<endl;
//	cout<<"Ԥ�������ǣ�"<< tmp.predict(mydata2,13)<<endl;
//	cout<<"��ʵ������ǣ� -1"<<endl;
//	cout<<endl;
	return 0;
}
