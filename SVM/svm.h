#include <bits/stdc++.h>
using namespace std;
 
#define MAX(a,b)  ((a)>(b)?(a):(b))
#define MIN(a,b)  ((a)<(b)?(a):(b))
 
 
 
//训练参数的结构体
struct SMOParams {
	int sample_num;				//所有的样本数
	int train_num;				//训练的样本数
	int dimension;			 //数据的维数
	double big_C;				//惩罚参数 
	double m_dT;				//在KKT条件中容忍范围 ????
	double eps;               //限制条件 
	double two_sigma_squared;  //RBF核函数中的参数 
};
 
 
 
class SMO {
public:
	SMO() {}
	~SMO();
 
	void train(const char *, const SMOParams &);
	void save();  //将分类器保存下来
	void load(const char *);
	void error_rate();   //计算分类正确率
	int predict(double *, int);
 
private:
	bool takeStep(int, int);
	double ui(int i1);  //分类输出
	double kernel(int, int);
	double kernel(int, double*); 
	double dotProduct(int, int);   //两个训练样本的点积
	int examineExample(int);
	void  readFile(const char *);
	bool examineFirstChoice(int, double);
	bool examineNonBound(int);
	bool examineBound(int);
	void init(const SMOParams &);
	void outerLoop();
 
 
private:
	int sample_num;				//所有的样本数
	int train_num;				//训练的样本数
	int dimension;		    	 //数据的维数
	double big_C;				   //惩罚系数
	double m_dT;			    	//在KKT条件中容忍范围
	double eps;                  //限制条件 
	double two_sigma_squared;     //RBF核函数中的参数 
	double neg_b;                          //阈值 
	bool is_load = false;	   //分类器是否通过加载已有分类器得到。如果是，就不能调用train等函数
	vector <int> target_label;           //类别标签
	double **all_data;                 //存放训练与测试样本 
	vector<double> alpha;               //拉格朗日乘子 
	vector<double> error_cache;         //存放non-bound样本误差 
	vector<double> dot_prod_cache;    //预存向量的点积以减少计算量 
};
