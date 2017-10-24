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

#include "igmn.h"

using liac::IGMN;

void IGMN::init(vector<double> dataRange, double tau, double delta, double spMin, double vMin)
{
    this->dataRange = dataRange;
    this->dimension = dataRange.size();
    this->size = 0;
    this->delta = delta;
    this->tau = tau;
    this->spMin = spMin;
    this->vMin = vMin;
	this->eta = std::numeric_limits<double>::min();
    this->chisq = Statistic::chi2inv(1 - tau, this->dimension);
    this->calculateValuesInitialSigma();
}
IGMN::IGMN(vector<double> dataRange, double tau, double delta)
{
    this->init(dataRange, tau, delta, dataRange.size() + 1, 2 * dataRange.size());
}
IGMN::IGMN(double tau, double delta)
{
    vector<double> data;
    this->init(data, tau, delta, 0, 0);
}

void IGMN::learn(MatrixXd x)
{
	this->computeLikelihood(x);

	if (!this->hasAcceptableDistance(x))
	{
		this->addComponent(x);
		int i = this->size - 1;
		this->like[i] = liac::Statistic::mvnpdf(x, this->means[i], this->invCovs[i], this->detCovs[i]) + this->eta;
		this->updatePriors();
	}

	this->computePosterior();
	this->incrementalEstimation(x);
	this->updatePriors();
    this->removeSpuriousComponents();
}
void IGMN::computeLikelihood(MatrixXd x)
{
	double value, distance;
	for (int i = 0; i < this->size; i++)
	{
		value = liac::Statistic::mvnpdf(x, this->means[i], this->invCovs[i], this->detCovs[i], &distance) + this->eta;
		this->like[i] = value;
		this->distances[i] = distance;
	}
}
bool IGMN::hasAcceptableDistance(MatrixXd x)
{
	for (int i = 0; i < this->size; i++)
		if (this->distances[i] < this->chisq)
			return true;
	return false;
}
void IGMN::addComponent(MatrixXd x)
{
	this->size += 1;
	this->priors.push_back(1);
	this->means.push_back(*(new MatrixXd(x)));
	this->invCovs.push_back(*(new MatrixXd(this->invSigmaIni)));
	this->detCovs.push_back(this->detSigmaIni);
	this->sps.push_back(1);
	this->vs.push_back(1);
	this->like.push_back(0);
	this->distances.push_back(0);
	this->post.push_back(0);
}
void IGMN::computePosterior()
{
	double sum = 0;
	for (int i = 0; i < this->size; i++)
		sum += (this->post[i] = this->like[i] * this->priors[i]);
	//Normalize
	for (int i = 0; i < this->size; i++)
		this->post[i] /= sum;	
}
void IGMN::incrementalEstimation(MatrixXd x)
{
	for (int i = 0; i < this->size; i++)
	{
		//Update age
		this->vs[i] += 1;
		//Update accumulator of the posteriori probability
		this->sps[i] += this->post[i];
		//Update mean
		MatrixXd oldmeans = this->means[i];
		double w = this->post[i] / this->sps[i]; //Learning rate
		MatrixXd diff = (x - oldmeans) * w;
		this->means[i] += diff;
		diff = this->means[i] - oldmeans; //delta mu
		MatrixXd diff2 = x - this->means[i]; //e*

		//Update invert covariance matrix
		//Plus a rank-one update
		MatrixXd invCov = this->invCovs[i];
		MatrixXd v = diff2 * sqrt(w); //v = u = e*.sqrt(w)
		MatrixXd tmp1 = invCov * (1.0 / (1.0 - w)); //A(t-1) / (1 - w)
		MatrixXd tmp2 = tmp1 * v; // matrix D x 1
		double tmp3 = 1 + Statistic::dot(tmp2, v);
		invCov = tmp1 - ((tmp2 * tmp2.transpose()) * (1.0 / tmp3));
		//Subtract a rank-one update
		MatrixXd tmp4 = invCov * diff; // matrix D x 1
		double tmp5 = 1 - Statistic::dot(tmp4, diff);
		invCov = invCov + ((tmp4 * tmp4.transpose()) * (1.0 / tmp5));
		this->invCovs[i] = invCov;

		//Update Determinant Covariance
		//Plus a rank-one update
		double detCov = this->detCovs[i];
		detCov = detCov * pow(1.0 - w, this->dimension) * (tmp3);
		//Subtract a rank-one update
		detCov = detCov * tmp5;
		this->detCovs[i] = detCov;
	}
}
void IGMN::updatePriors()
{
	double sum = 0;
	for (int i = 0; i < this->size; i++)
		sum += this->sps[i];
	//Normalize
	for (int i = 0; i < this->size; i++)
		this->priors[i] = this->sps[i] / sum;
}

void IGMN::removeSpuriousComponents()
{
    for(int i = size - 1; i >= 0; i--)
    {
        if (this->vs[i] > this->vMin && this->sps[i] < this->spMin)
        {
            this->vs.erase(this->vs.begin() + i);
            this->sps.erase(this->sps.begin() + i);
            this->priors.erase(this->priors.begin() + i);
            this->detCovs.erase(this->detCovs.begin() + i);
            this->means.erase(this->means.begin() + i);
            this->invCovs.erase(this->invCovs.begin() + i);
            this->distances.erase(this->distances.begin() + i);
            this->like.erase(this->like.begin() + i);
            this->post.erase(this->post.begin() + i);
            this->size -= 1;
        }
    }
}

MatrixXd IGMN::recall(MatrixXd x)
{
    int num_inp = x.rows();
    int num_out = this->dimension - num_inp;
    double sum = 0;
    MatrixXd result(num_inp, 1);
    result.setZero();
    vector<double> pajs;
    vector<MatrixXd> xm;

    for (int i = 0; i < this->size; ++i)
    {
        MatrixXd blockZ = this->invCovs[i].block(num_inp, 0, num_out, num_inp);
        MatrixXd blockW = this->invCovs[i].block(num_inp, num_inp, num_out, num_out);
        MatrixXd blockX = this->invCovs[i].block(0, 0, num_inp, num_inp);

        MatrixXd meanA = this->means[i].block(0, 0, num_inp, 1);
        MatrixXd meanB = this->means[i].block(num_inp, 0, num_out, 1);

        MatrixXd invBlockW = blockW.inverse();
        MatrixXd invBlockA = blockX - ( blockZ.transpose() * invBlockW * blockZ);

        pajs.push_back(Statistic::mvnpdf(x, meanA, invBlockA, this->detCovs[i] * invBlockW.determinant()) + this->eta);
        //Recall for the component
        xm.push_back(meanB - (invBlockW * blockZ * (x - meanA)));
    }

    //Normalize values
    for (int i = 0; i < this->size; i++)
        sum += pajs[i];

    for (int i = 0; i < this->size; i++)
        result = result + (xm[i] * pajs[i] / sum);

    return result;
}

void IGMN::calculateValuesInitialSigma()
{
    double determinant = 1;
    MatrixXd sigma(this->dataRange.size(), this->dataRange.size());

    sigma.setZero();

    for (int i = 0; i < sigma.rows(); ++i)
        sigma(i, i) = pow (this->dataRange[i] * this->delta, 2);

    for(int i = 0; i < sigma.cols(); i++)
    {
        determinant *= sigma(i, i);
        sigma(i, i) = 1.0 / sigma(i, i);
    }

    this->invSigmaIni = sigma;
    this->detSigmaIni = determinant;
}

void IGMN::train(MatrixXd x)
{
	for (int i = 0; i < x.cols(); i++)
		this->learn(x.block(0, i, x.rows(), 1));
}

int IGMN::getSize()
{
	return this->size;
}

MatrixXd IGMN::getInvSigmaIni()
{
    return this->invSigmaIni;
}

double IGMN::getDetSigmaIni()
{
    return this->detSigmaIni;
}

vector<MatrixXd> IGMN::getInvCovs()
{
    return this->invCovs;
}

vector<double> IGMN::getDetCovs()
{
    return this->detCovs;
}

double IGMN::getChisq()
{
    return this->chisq;
}