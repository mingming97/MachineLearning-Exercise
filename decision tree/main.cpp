#include <bits/stdc++.h>

using namespace std;

vector<vector<string> > data;
vector<string> all_fea_name;
//map<string, int> feaname2col; 

struct Node {
	string fea_name; //该节点代表的特征名称 
	vector<Node *> son; //子节点 
	vector<string> fea_values; //子节点下标的值
	Node() {
		fea_name = "";
	} 
	Node(string name) {
		fea_name = name;
	}
};

int feaname2col(const vector<string> &fea_names, string fea_name) {
	for (int i = 0; i < fea_names.size(); i++)
		if (fea_name == fea_names[i])
			return i;
	return -1;
}

string trim(string &str) {
	str.erase(0, str.find_first_not_of(" \t\r\n"));
	str.erase(str.find_last_not_of(" \t\r\n") + 1);
	return str;
}

void read_data() {
	ifstream fin("lenses.csv");
	string line;
	int cnt = -1;
	while (getline(fin, line)) {
		cnt++;
		istringstream sin(line);
		vector<string> fields;
		string field;
		while (getline(sin, field, ',')) {
			if (cnt == 0) {
				all_fea_name.push_back(field);
			}
			else
				fields.push_back(field);
		}
		if (cnt != 0)
			data.push_back(fields);
	}
}

void cut_feaname(const vector<string> &fea_names, string fea_name, vector<string> &res) {
	for (int i = 0; i < fea_names.size(); i++)
		if (fea_names[i] != fea_name)
			res.push_back(fea_names[i]);
}

void cut_dataset(const vector<vector<string> > &dataset, int fea_col, 
				string fea_value, vector<vector<string> > &res) {
					for (int i = 0; i < dataset.size(); i++) {
						if (dataset[i][fea_col] != fea_value)
							continue;
						vector<string> tmp;
						for (int j = 0; j < dataset[i].size(); j++) {
							if (j == fea_col) continue;
							tmp.push_back(dataset[i][j]);
						}
						res.push_back(tmp);
					} 	
				}

double compute_ent(const vector<vector<string> >& dataset) {
	map<string, int> freq;
	for (int i = 0; i < dataset.size(); i++)
		freq[dataset[i][dataset[i].size() - 1]]++;
	double res = 0;
	for (auto &x: freq){
		double p = x.second * 1.0 / dataset.size();
		if (p == 0) continue;
		res += (-p * log2(p));
	}
	return res;
}

double compute_infogain(const vector<vector<string> >& dataset, int fea_col) {
	double start_ent = compute_ent(dataset);
	double info_ent = 0;
	
	// 目标特征中各个值的出现次数 
	map<string, int> fea_value_freq;
	for (int i = 0; i < dataset.size(); i++)
		fea_value_freq[dataset[i][fea_col]]++;
	
	for (auto &x: fea_value_freq) {
		vector<vector<string> > subset;
		cut_dataset(dataset, fea_col, x.first, subset);
		double p = subset.size() * 1.0 / dataset.size();
		info_ent += p * compute_ent(subset);
	}
	return start_ent - info_ent;
}

string get_best_fea(const vector<vector<string> > &dataset, const vector<string> &fea_names) {
	double maxx = -999999;
	string maxx_name;
	for (int i = 0; i < fea_names.size(); i++) {
		double tmp = compute_infogain(dataset, feaname2col(fea_names, fea_names[i]));
		if (tmp > maxx) {
			maxx = tmp;
			maxx_name = fea_names[i];
		}
	}
	return maxx_name;
}

string get_max_prob(const vector<vector<string> > &dataset, int fea_col, string fea_value) {
	map<string, int> label_value;
	for (int i = 0; i < dataset.size(); i++) {
		if (dataset[i][fea_col] == fea_value)
			label_value[dataset[i][dataset[0].size() - 1]]++;
	}
	int maxx = 0;
	string maxx_name;
	for (auto &x: label_value) {
		if (x.second > maxx) {
			maxx = x.second;
			maxx_name = x.first;
		}
	}
	return maxx_name;
}

Node *build_decision_tree(const vector<vector<string> > &dataset, const vector<string> &fea_names) {
	Node *root = new Node();
	if (fea_names.size() == 1) {
		root->fea_name = fea_names[0];
		map<string, bool> all_fea_value;
		for (int i = 0; i < dataset.size(); i++)
			all_fea_value[dataset[i][0]] = 1;
		for (auto &x: all_fea_value) {
			string res = get_max_prob(dataset, 0, x.first);
			Node* tmp = new Node(res);
			root->son.push_back(tmp);
			root->fea_values.push_back(x.first);
		}
		return root;
	}
	
	string fea_name = get_best_fea(dataset, fea_names);
	root->fea_name = fea_name;	
	
	int fea_col = feaname2col(fea_names, fea_name);
	map<string, bool> fea_values;
	for (int i = 0; i < dataset.size(); i++)
		fea_values[dataset[i][fea_col]] = 1;
	for (auto &x: fea_values) {
		vector<vector<string> > subset;
		cut_dataset(dataset, fea_col, x.first, subset);
		Node *node = NULL;
		if (compute_ent(subset) == 0)
			node = new Node(subset[0][subset[0].size() - 1]);
		else {
			vector<string> new_fea_name;
			cut_feaname(fea_names, fea_name, new_fea_name);
			node = build_decision_tree(subset, new_fea_name);
		}
		root->son.push_back(node);
		root->fea_values.push_back(x.first);
	}
	return root;
}

string classify(Node *root, const vector<string> &input, const vector<string> &fea_names) {
	Node *node = root;
	while (node->son.size() > 0) {
		bool can_judge = false;
		for (int i = 0; i < fea_names.size(); i++) {
			if (node->fea_name == fea_names[i]) {
				for (int j = 0; j < node->fea_values.size(); j++)
					if (input[i] == node->fea_values[j]) {
						node = node->son[j];
						can_judge = true;
						break;
					}
				break;
			}
		}
		if (!can_judge) return "cannot judge";
	}
	return node->fea_name;
}

void print_dataset(const vector<vector<string> > &dataset) {
	for (int i = 0; i < dataset.size(); i++) {
		for (int j = 0; j < dataset[0].size(); j++)
			cout << dataset[i][j] << "   ";
		cout << endl;
	}
}
int main() {
	read_data();

	cout << compute_ent(data) << endl;
	
	vector<vector<string> > subset;
	cut_dataset(data, 0, "pre", subset);
	print_dataset(subset);
	cout << compute_infogain(data, 0) << endl;
	
	vector<string> fea_name;
	cut_feaname(all_fea_name, "age", fea_name);
	for (int i = 0; i < fea_name.size(); i++)
		cout << fea_name[i] << endl;
	cout << endl;
	
	cout << get_max_prob(data, 0, "young") << endl;
	
	Node* root = build_decision_tree(data, all_fea_name);
	cout << root->fea_name << endl;
	
	vector<string> data1{"pre", "myope", "no", "normal"};
    vector<string> data2{"pre", "myope", "no", "reduced"};
    vector<string> data3{"presbyopic", "myope", "yes", "normal"};
    vector<string> data4{"prees", "myope", "no", "normal"};
 	cout << endl;
	cout << classify(root, data1, all_fea_name) << endl;
    cout << classify(root, data2, all_fea_name) << endl;
    cout << classify(root, data3, all_fea_name) << endl;
    cout << classify(root, data4, all_fea_name) << endl;
	return 0;
} 
