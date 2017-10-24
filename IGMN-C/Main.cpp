#include <iostream>
#include <Eigen/Dense>
#include "igmn.h"

using Eigen::MatrixXd;

using namespace std;

int main()
{
	std::cout << Statistic::chi2inv(12, 2) <<std::endl;

    vector<double> range (2);
	range[0] = range[1] = 2;

    liac::IGMN loigmn(range, 0.1, 0.3);

    int num = 63;
	MatrixXd m(2, num);
	int i = 0;

	for (float x = 0; x <= 2 * Constants::PI && i < num; x += 0.1f)
	{
		m(0, i) = x;
		m(1, i++) = std::sin(x);
	}

    loigmn.train(m);

	for (int j = 0; j < i; j++)
	{
		MatrixXd input = m.block(0, j, 1, 1);
		MatrixXd res = loigmn.recall(input);

		cout << "Input: " << input  << " Real: " << m(1, j) << " Recall: " << res << endl;
	}

	system("PAUSE");
}
