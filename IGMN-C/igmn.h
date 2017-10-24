#ifndef __INCLUDED_IGMN_H__
#define __INCLUDED_IGMN_H__

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

#include <vector>
#include <Eigen/Dense>
#include "statistic.h"

using liac::Statistic;
using std::vector;
using Eigen::MatrixXd;

namespace liac
{
	class IGMN {
	private:
		int dimension;
		int size;
		double delta;
		double tau;
		double spMin;
		double vMin;
		double chisq;
		double eta;
		vector<double> dataRange;
		MatrixXd invSigmaIni;
		double detSigmaIni;

        //Attributes for each component
		vector<double> priors;
		vector<MatrixXd> means;
		vector<MatrixXd> invCovs;
		vector<double> like;
		vector<double> post;
		vector<double> detCovs;
		vector<double> sps;
		vector<double> vs;
		vector<double> distances;

		void computeLikelihood(MatrixXd);
		bool hasAcceptableDistance(MatrixXd);
		void addComponent(MatrixXd);
		void computePosterior();
		void incrementalEstimation(MatrixXd);
		void updatePriors();
        void removeSpuriousComponents();
        void calculateValuesInitialSigma();

    public:
        void init(vector<double> dataRange, double tau, double delta, double spMin, double vMin);
        IGMN(vector<double> dataRange, double tau, double delta);
        IGMN(double tau, double delta);

        void learn(MatrixXd);
        void train(MatrixXd);
        MatrixXd recall(MatrixXd);

		int getSize();
        double getChisq();
        double getDetSigmaIni();
        vector<MatrixXd> getInvCovs();
        vector<double> getDetCovs();
        MatrixXd getInvSigmaIni();
	};
}
/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_IGMN_H__