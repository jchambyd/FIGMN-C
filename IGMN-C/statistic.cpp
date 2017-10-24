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

#include "statistic.h"

using liac::Statistic;

Statistic::Statistic()
{
}

double Statistic::dot(MatrixXd x, MatrixXd y)
{
    return (x.array() * y.array()).sum();
}

double Statistic::mvnpdf(MatrixXd& x, MatrixXd& mu, MatrixXd& sigma)
{
    int dimension = x.size();
    double determinant = sigma.determinant();
    MatrixXd inverse = sigma.inverse();
    MatrixXd distance = x - mu;
    return exp(-0.5 * dot(distance, inverse * distance)) * 1.0 / (pow((2 *  Constants::PI), (dimension / 2.0)) * sqrt(determinant));
}

double Statistic::mvnpdf(MatrixXd& x, MatrixXd& mu, MatrixXd& inverse, double determinant)
{
    int dimension = x.size();
    MatrixXd distance = x - mu;

    return exp(-0.5 * dot(distance, inverse * distance)) * 1.0 / (pow((2 * Constants::PI), (dimension / 2.0)) * sqrt(determinant));
}

double Statistic::mvnpdf(MatrixXd& x, MatrixXd& mu, MatrixXd& inverse, double& determinant, double* pdistance)
{
    int dimension = x.size();
    MatrixXd distance = x - mu;
    //Save the distance
    *pdistance = dot(distance, inverse * distance);

    double pdf = exp(-0.5 * (*pdistance)) * 1.0 / (pow((2 * Constants::PI), (dimension / 2.0)) * sqrt(determinant));

    pdf = isnan(pdf) ? 0 : pdf;
    pdf = isinf(pdf) ? std::numeric_limits<double>::max() : pdf;

    return pdf;
}

vector<double> Statistic::rates(double alpha, int categories)
{
    if (alpha == 0.0)
    {
        vector<double> res(categories);
        for (int i = 0; i < res.size(); i++)
            res[i] = 1.0;
        return res;
    }
    else
    {
        vector<double> res(categories);
        double total = 0.0;
        for (int i = 0; i < categories; i ++)
        {
            double pa = chi2inv(( (double) i)    / ((double) categories),2*alpha) / (2*alpha);
            double pb = chi2inv(( (double) i+1.0) / ((double) categories),2*alpha) / (2*alpha);

            double ia = gammaintegral(alpha+1,pa*alpha);
            double ib = gammaintegral(alpha+1,pb*alpha);

            res[i] = (ib - ia) / categories;
            total += (res[i] / categories);
        }

        for (int i = 0; i < categories; i ++)
            res[i] = res[i] / total;

        return res;
    }
}

double Statistic::chi2inv(double p, double v)
{
    if (p < 0.000002)
    {
        return 0.0;
    }
    if (p > 0.999998)
    {
        p = 0.999998;
    }

    double xx = 0.5 * v;
    double  c = xx - 1.0;
    double aa = std::log(2);
    double g = gammaln(v/2.0);
    double ch;
    if (v > (-1.24 * std::log(p)))
    {
        if (v > 0.32)
        {
            //3
            double x = gauinv(p);
            double p1 = 0.222222 / v;
            ch = v * std::pow(x * std::sqrt(p1) + 1.0 - p1, 3);
            if (ch > (2.2 * v + 6.0))
                ch = -2.0 * (std::log(1.0 - p) - c * std::log(0.5 * ch) + g);
        }
        else
        {
            //1+2
            ch = 0.4;
            double q;
            double a = std::log(1.0-p);
            do
            {
                q = ch;
                double p1 = 1.0 + ch * (4.67 + ch);
                double p2 = ch * (6.73 + ch * (6.66 + ch));
                double t = -0.5 + (4.67 + 2.0 * ch) / p1 -
                           (6.73 + ch * (13.32 + 3.0 * ch)) / p2;
                ch = ch - (1.0 - std::exp(a + g + 0.5 * ch + c * aa) * p2 / p1) / t;
            }
            while (std::abs(q / ch - 1.0) >= 0.01);
        }
    }
    else
    {
        //START
        ch = std::pow(p * xx * std::exp(g + xx * aa), 1.0 / xx);
    }
    double q;
    do
    {
        //4 + 5
        q = ch;
        double p1 = 0.5 * ch;
        double p2 = p - gammaintegral(xx,p1);
        double t = p2 * std::exp(xx * aa + g + p1 - c * std::log(ch));
        double b = t / ch;
        double a = 0.5 * t - b * c;
        double s1 = (210.0 + a * (140.0 + a * (105.0 + a * (84.0 + a * (70.0 + 60.0 * a))))) / 420.0;
        double s2 = (420.0 + a * (735.0 + a * (966.0 + a * (1141.0 + 1278.0 * a)))) / 2520.0;
        double s3 = (210.0 + a * (462.0 + a * (707.0 + 932.0 * a))) / 2520.0;
        double s4 = (252.0 + a * (672.0 + 1182.0 * a) + c * (294.0 + a * (889.0 + 1740.0 * a))) / 5040.0;
        double s5 = (84.0 + 264.0 * a + c * (175.0 + 606.0 * a)) / 2520.0;
        double s6 = (120.0 + c * (346.0 + 127.0 *c)) / 5040.0;
        ch = ch + t * (1.0+0.5*t*s1-b*c*(s1-b*(s2-b*(s3-b*(s4-b*(s5-b*s6))))));
    }
    while (std::abs(q / ch - 1.0) > E);
    return ch;
}


double Statistic::gammaintegral(double p, double x)
{
    double g = gammaln(p);
    double factor = std::exp(p * std::log(x) - x - g);
    double gin;
    if ((x > 1.0) && (x > p))
    {
        bool end = false;
        double a = 1.0 - p;
        double b = a+x+1.0;
        double term = 0.0;
        vector<double> pn (6);
        pn[0] = 1.0;
        pn[1] = x;
        pn[2] = x+1.0;
        pn[3] = x*b;
        gin = pn[2] / pn[3];
        do
        {
            double rn;
            a++;
            b = b + 2.0;
            term++;
            double an = a * term;
            for (int i = 0; i <= 1; i++)
            {
                pn[i+4] = b * pn[i+2]-an*pn[i];
            }
            if (pn[5] != 0.0)
            {
                rn = pn[4] / pn[5];
                double diff = std::abs(gin - rn);
                if (diff < E*rn)
                    end = true;
                else
                    gin = rn;
            }
            if (!end)
            {
                for (int i = 0; i < 4; i++)
                {
                    pn[i] = pn[i+2];
                }
                if (std::abs(pn[5]) >= OFLO)
                {
                    for (int i = 0; i < 4; i++)
                        pn[i] = pn[i] / OFLO;
                }
            }
        }
        while (!end);

        gin = 1.0 - factor*gin;
    }
    else
    {
        gin = 1.0;
        double term = 1.0;
        double rn = p;
        do
        {
            rn++;
            term = term * x / rn;
            gin = gin + term;
        }
        while (term > E);
        gin = gin * factor / p;
    }
    return gin;
}

double Statistic::gauinv(double p)
{
    if (p == 0.5)
    {
        return 0.0;
    }
    double ps = p;
    if (ps > 0.5)
    {
        ps = 1 - ps;
    }
    double yi = std::sqrt(std::log(1.0 / (ps * ps)));
    double gauinv = yi + ((((yi * p4 + p3) * yi + p2) * yi + p1) * yi + p0) /
                         ((((yi * q4 + q3) * yi + q2) * yi + q1) * yi + q0);
    if (p < 0.5)
        return -gauinv;
    else
        return gauinv;
}

double Statistic::gammaln(double xx)
{
    double y = xx;
    double x = xx;
    double tmp = x + 5.2421875;
    tmp = (x + 0.5) * std::log(tmp) - tmp;
    double ser = 0.999999999999997092;
    for (int i = 0; i < 14; i++)
        ser += COF[i]/++y;

    return tmp + std::log(2.5066282746310005 * ser / x);
}

const double Statistic::p0 = -0.322232431088;
const double Statistic::p1 = -1.0;
const double Statistic::p2 = -0.342242088547;
const double Statistic::p3 = -0.204231210245e-1;
const double Statistic::p4 = -0.453642210148e-4;
const double Statistic::q0 = 0.993484626060e-1;
const double Statistic::q1 = 0.588581570495;
const double Statistic::q2 = 0.531103462366;
const double Statistic::q3 = 0.103537752850;
const double Statistic::q4 = 0.38560700634e-2;
const double Statistic::OFLO = 10e30;
const double Statistic::E = 10e-6;
const double Statistic::COF[] = {57.1562356658629235,
								-59.5979603554754912,
								14.1360979747417471,
								-0.491913816097620199,
								0.339946499848118887e-4,
								0.465236289270485756e-4,
								-0.983744753048795646e-4,
								0.158088703224912494e-3,
								-0.210264441724104883e-3,
								0.217439618115212643e-3,
								-0.164318106536763890e-3,
								0.844182239838527433e-4,
								-0.261908384015714087e-4,
								0.368991826595316234e-5 };