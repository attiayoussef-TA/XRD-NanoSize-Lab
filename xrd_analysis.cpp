#include "xrd_analysis.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <Eigen/Dense>


using namespace Eigen
using namespace std;

/* ===================== CSV LOADING ===================== */
XRDData loadCSV(const string& filename) {
    XRDData data;
    ifstream file(filename);
    string line;
    getline(file, line); // skip header
    while(getline(file,line)){
        stringstream ss(line);
        string t,i;
        getline(ss,t,','); getline(ss,i,',');
        data.theta.push_back(stod(t));
        data.intensity.push_back(stod(i));
    }
    return data;
}

/* ===================== PREPROCESSING ===================== */
void normalize(vector<double>& y){
    double maxVal = *max_element(y.begin(),y.end());
    for(auto &v:y) v/=maxVal;
}

vector<double> smooth(const vector<double>& y,int window){
    vector<double> out(y.size());
    int h=window/2;
    for(size_t i=0;i<y.size();i++){
        double sum=0; int cnt=0;
        for(int j=-h;j<=h;j++){
            int idx=i+j;
            if(idx>=0 && idx<y.size()){sum+=y[idx]; cnt++;}
        }
        out[i]=sum/cnt;
    }
    return out;
}

/* ===================== PEAK DETECTION ===================== */
int detectPeak(const vector<double>& theta,const vector<double>& intensity,double expected,double window){
    double maxI=0; int idx=-1;
    for(size_t i=0;i<theta.size();i++){
        if(theta[i]>expected-window && theta[i]<expected+window){
            if(intensity[i]>maxI){maxI=intensity[i]; idx=i;}
        }
    }
    return idx;
}

/* ===================== VOIGT FUNCTION ===================== */
double voigt(double x,const VoigtParams& p){
    double eta=0.5;
    double g=p.A*exp(-pow(x-p.mu,2)/(2*p.sigma*p.sigma));
    double l=p.A*(p.gamma*p.gamma)/((x-p.mu)*(x-p.mu)+p.gamma*p.gamma);
    return eta*g+(1-eta)*l;
}

// ===================== LEVENBERG-MARQUARDT WITH EIGEN =====================
VoigtParams fitVoigtLM(const std::vector<double>& x,
                       const std::vector<double>& y,
                       VoigtParams p,
                       int iterations,
                       double lambda_init)
{
    double lambda = lambda_init;
    int nParams = 4;

    for (int it = 0; it < iterations; it++)
    {
        VectorXd r(x.size());           // residual vector
        MatrixXd J(x.size(), nParams);  // Jacobian matrix

        for (size_t i = 0; i < x.size(); i++)
        {
            double xi = x[i];
            double g = p.A * exp(-pow(xi - p.mu, 2) / (2 * p.sigma * p.sigma));
            double l = p.A * (p.gamma * p.gamma) / ((xi - p.mu) * (xi - p.mu) + p.gamma * p.gamma);
            double f = 0.5 * g + 0.5 * l;

            // residual
            r(i) = y[i] - f;

            // Jacobian entries (positive derivatives)
            J(i, 0) = 0.5 * exp(-pow(xi - p.mu, 2) / (2 * p.sigma * p.sigma))
                      + 0.5 * (p.gamma * p.gamma) / ((xi - p.mu) * (xi - p.mu) + p.gamma * p.gamma); // dF/dA
            J(i, 1) = 0.5 * g * (xi - p.mu) / (p.sigma * p.sigma)
                      + p.A * p.gamma * p.gamma * (xi - p.mu) / pow((xi - p.mu)*(xi - p.mu) + p.gamma*p.gamma, 2); // dF/dMu
            J(i, 2) = 0.5 * g * pow(xi - p.mu, 2) / pow(p.sigma, 3); // dF/dSigma
            J(i, 3) = p.A * p.gamma * (xi - p.mu)*(xi - p.mu) / pow((xi - p.mu)*(xi - p.mu) + p.gamma*p.gamma, 2); // dF/dGamma
        }

        // Levenberg-Marquardt step: (J^T J + lambda * I) delta = J^T r
        MatrixXd H = J.transpose() * J;
        VectorXd g = J.transpose() * r;

        // damping
        H.diagonal().array() *= (1.0 + lambda);

        // solve for parameter update
        VectorXd delta = H.ldlt().solve(g);

        // update parameters
        p.A     += delta(0);
        p.mu    += delta(1);
        p.sigma += delta(2);
        p.gamma += delta(3);

        // enforce physical constraints
        p.A     = std::max(1e-9, p.A);
        p.sigma = std::max(1e-9, p.sigma);
        p.gamma = std::max(1e-9, p.gamma);

        // optional: keep mu within data range
        p.mu = std::min(std::max(p.mu, x.front()), x.back());

        // dynamic damping
        lambda *= 0.99;
    }

    return p;
}

/* ===================== FWHM & SCHERRER ===================== */
double fwhmVoigt(const VoigtParams& p){
    double f=0.5346*2*p.gamma + sqrt(0.2166*4*p.gamma*p.gamma + 5.551*p.sigma*p.sigma);
    return f;
}

double scherrer(double fwhm_deg,double theta_deg,double wavelength){
    double beta=fwhm_deg*M_PI/180.0;
    double theta=(theta_deg/2.0)*M_PI/180.0;
    const double K=0.9;
    return (K*wavelength)/(beta*cos(theta));
}

/* ===================== RESIDUALS & STATISTICAL ANOMALY ===================== */
vector<double> computeResiduals(const vector<double>& x,const vector<double>& y,const VoigtParams& p){
    vector<double> r;
    for(size_t i=0;i<x.size();i++) r.push_back(y[i]-voigt(x[i],p));
    return r;
}

double residualAnomalyScore(const vector<double>& residuals){
    double mean=accumulate(residuals.begin(),residuals.end(),0.0)/residuals.size();
    double var=0;
    for(double r:residuals) var+=pow(r-mean,2);
    double stdv=sqrt(var/(residuals.size()-1));
    double score=0;
    for(double r:residuals) score+=fabs((r-mean)/stdv);
    return score/residuals.size();
}

/* ===================== MONTE CARLO FWHM ===================== */
double monteCarloFWHM(const vector<double>& x,const vector<double>& y,const VoigtParams& p,int n){
    default_random_engine rng(42);
    vector<double> res=computeResiduals(x,y,p);
    double res_std=sqrt(accumulate(res.begin(),res.end(),0.0,[](double s,double v){return s+v*v;})/res.size());
    normal_distribution<double> dist(0.0,res_std);
    double fwhm_sum=0;
    for(int i=0;i<n;i++){
        vector<double> y_mc=y;
        for(size_t j=0;j<y_mc.size();j++) y_mc[j]+=dist(rng);
        VoigtParams p_mc=fitVoigtLM(x,y_mc,p,100,1e-4);
        fwhm_sum+=fwhmVoigt(p_mc);
    }
    return fwhm_sum/n;
}

/* ===================== FULL ANALYSIS ===================== */
void analyze(const string& label,const string& file,double wavelength){
    cout<<"\n===== "<<label<<" =====\n";
    XRDData d=loadCSV(file);
    d.intensity=smooth(d.intensity);
    normalize(d.intensity);

    int peakIdx=detectPeak(d.theta,d.intensity);
    double mu0=d.theta[peakIdx];

    vector<double> x,y;
    for(size_t i=0;i<d.theta.size();i++)
        if(fabs(d.theta[i]-mu0)<1.5){x.push_back(d.theta[i]); y.push_back(d.intensity[i]);}

    VoigtParams init{1.0,mu0,0.15,0.05};
    VoigtParams fit=fitVoigtLM(x,y,init,200,1e-4);

    vector<double> residuals=computeResiduals(x,y,fit);
    double fwhm=fwhmVoigt(fit);
    double size=scherrer(fwhm,fit.mu,wavelength);
    double anomaly=residualAnomalyScore(residuals);
    double fwhm_unc=monteCarloFWHM(x,y,fit,50);

    cout<<"Peak position (2θ)              : "<<fit.mu<<" deg\n";
    cout<<"Gaussian σ                       : "<<fit.sigma<<"\n";
    cout<<"Lorentz γ                        : "<<fit.gamma<<"\n";
    cout<<"FWHM                             : "<<fwhm<<" deg ± "<<fwhm_unc-fwhm<<"\n";
    cout<<"Crystallite size                 : "<<size<<" Å\n";
    cout<<"Statistical Residual Anomaly     : "<<anomaly<<"\n";
}
