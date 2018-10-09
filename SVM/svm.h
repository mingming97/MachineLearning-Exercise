#include <bits/stdc++.h>
using namespace std;
 
#define MAX(a,b)  ((a)>(b)?(a):(b))
#define MIN(a,b)  ((a)<(b)?(a):(b))
 
 
 
//ѵ�������Ľṹ��
struct SMOParams {
	int sample_num;				//���е�������
	int train_num;				//ѵ����������
	int dimension;			 //���ݵ�ά��
	double big_C;				//�ͷ����� 
	double m_dT;				//��KKT���������̷�Χ ????
	double eps;               //�������� 
	double two_sigma_squared;  //RBF�˺����еĲ��� 
};
 
 
 
class SMO {
public:
	SMO() {}
	~SMO();
 
	void train(const char *, const SMOParams &);
	void save();  //����������������
	void load(const char *);
	void error_rate();   //���������ȷ��
	int predict(double *, int);
 
private:
	bool takeStep(int, int);
	double ui(int i1);  //�������
	double kernel(int, int);
	double kernel(int, double*); 
	double dotProduct(int, int);   //����ѵ�������ĵ��
	int examineExample(int);
	void  readFile(const char *);
	bool examineFirstChoice(int, double);
	bool examineNonBound(int);
	bool examineBound(int);
	void init(const SMOParams &);
	void outerLoop();
 
 
private:
	int sample_num;				//���е�������
	int train_num;				//ѵ����������
	int dimension;		    	 //���ݵ�ά��
	double big_C;				   //�ͷ�ϵ��
	double m_dT;			    	//��KKT���������̷�Χ
	double eps;                  //�������� 
	double two_sigma_squared;     //RBF�˺����еĲ��� 
	double neg_b;                          //��ֵ 
	bool is_load = false;	   //�������Ƿ�ͨ���������з������õ�������ǣ��Ͳ��ܵ���train�Ⱥ���
	vector <int> target_label;           //����ǩ
	double **all_data;                 //���ѵ����������� 
	vector<double> alpha;               //�������ճ��� 
	vector<double> error_cache;         //���non-bound������� 
	vector<double> dot_prod_cache;    //Ԥ�������ĵ���Լ��ټ����� 
};
