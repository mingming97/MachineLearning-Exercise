#include "svm.h"
#include <iostream>
using namespace std;
 
int main() {
	SMOParams params;
	params.sample_num = 270;				//所有的样本数
	params.train_num = 200;				//训练的样本数
	params.dimension = 13;			 //数据的维数
	params.big_C = 1.0;				//惩罚参数 
	params.m_dT = 0.001;				//在KKT条件中容忍范围
	params.eps = 1.0E-12;               //限制条件 
	params.two_sigma_squared = 2.0;  //RBF核函数中的参数 
 
 
	SMO tmp;
	tmp.train("heart_scale.txt", params);
	tmp.error_rate();
	tmp.save();
 
//	tmp.load("svm.txt");
//	//这个数据是heart_scale的第17行，对应的输出应该是+1
//	double mydata[] = {-0.291667, 1, 1, -0.132075, -0.155251, -1, -1, -0.251908, 1, -0.419355, 0, 0.333333, 1};
//	//这个数据是heart_scale的第264行，对应的输出应该是-1
//	double mydata2[] = { -0.166667, 1, -0.333333, -0.320755, -0.360731, -1, -1, 0.526718, -1, -0.806452, -1, -1, -1,};
//	cout<<"预测的类别是："<< tmp.predict(mydata,13)<<endl;
//	cout<<"真实的类别是： 1"<<endl;
//	cout<<endl;
//	cout<<"预测的类别是："<< tmp.predict(mydata2,13)<<endl;
//	cout<<"真实的类别是： -1"<<endl;
//	cout<<endl;
	return 0;
}
