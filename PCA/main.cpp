#include <bits/stdc++.h>
#include "matrix.h"

using namespace std;

int main(int argc, char** argv) {
	int k = 3; // num of sigular values
    vector<vector<double> > B{ {0, 0, 1, 1, 0, 0, 0, 0, 0}, 
							   {0, 0, 0, 0, 0, 1, 0, 0, 1},
							   {0, 1, 0, 0, 0, 0, 0, 1, 0},
							   {0, 0, 0, 0, 0, 0, 1, 0, 1},
							   {1, 0, 0, 0, 0, 1, 0, 0, 0},
							   {1, 1, 1, 1, 1, 1, 1, 1, 1},
							   {1, 0, 1, 0, 0, 0, 0, 0, 0},
							   {0, 0, 0, 0, 0, 0, 1, 0, 1},
							   {0, 0, 0, 0, 0, 2, 0, 0, 1},
							   {1, 0, 1, 0, 0, 0, 0, 1, 0},
							   {0, 0, 0, 1, 1, 0, 0, 0, 0} };
    cout << "Original matrix =" << endl;
    Matrix tmp(B);
    tmp.print();
    Matrix U, S, V;
    tmp.svd(k, U, S, V);
    cout << "U =" << endl;
    U.print();
    cout<< "S =" <<endl;
    S.print();
    cout << endl;
    cout << "V =" << endl;
    V.print();
    cout << "V transpose = " << endl;
    V.transpose().print();
    cout << "new Matrix = " << endl; 
    Matrix tmp2 = U * S * V.transpose();
    tmp2.print();
}
