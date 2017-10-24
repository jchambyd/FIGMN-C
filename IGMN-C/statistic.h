// ============================================================================
// Federal University of Rio Grande do Sul (UFRGS)
// Connectionist Artificial Intelligence Laboratory (LIAC)
// Jorge Cristhian Chamby Diaz - jccdiaz@inf.ufrgs.br
// ============================================================================
// Copyright (c) 2017 Jorge Cristhian Chamby Diaz, jchambyd at gmail dot com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to 
// deal in the Software without restriction, including without limitation the 
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
// sell copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ============================================================================

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <math.h>
#include "constants.h"

using Eigen::MatrixXd;
using std::pow;
using std::exp;
using std::sqrt;
using std::vector;
using liac::Constants;

namespace liac 
{
	class Statistic
	{
	public:
		Statistic();

		static double dot(MatrixXd x, MatrixXd y);
		double static mvnpdf(MatrixXd& x, MatrixXd& mu, MatrixXd& sigma);
		double static mvnpdf(MatrixXd& x, MatrixXd& mu, MatrixXd& inverse, double determinant);
		double static mvnpdf(MatrixXd& x, MatrixXd& mu, MatrixXd& inverse, double& determinant, double* pdistance);
		static vector<double> rates(double alpha, int categories);
		static double chi2inv(double p, double v);
		static double gammaintegral(double p, double x);
		static double gauinv(double p);
		static double gammaln(double xx);

	private:
		static const double p0;
		static const double p1;
		static const double p2;
		static const double p3;
		static const double p4;
		static const double q0;
		static const double q1;
		static const double q2;
		static const double q3;
		static const double q4;
		static const double OFLO;
		static const double E;
		static const double COF[14];
	};
}
