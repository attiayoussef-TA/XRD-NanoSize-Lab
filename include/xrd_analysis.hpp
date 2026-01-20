#pragma once
#include <vector>
#include <string>

struct XRDData {
    std::vector<double> theta;
    std::vector<double> intensity;
};

struct VoigtParams {
    double A;      // amplitude
    double mu;     // peak center
    double sigma;  // Gaussian width
    double gamma;  // Lorentzian width
};

/* ===================== FUNCTION PROTOTYPES ===================== */

// Load CSV file
XRDData loadCSV(const std::string& filename);

// Preprocessing
void normalize(std::vector<double>& y);
std::vector<double> smooth(const std::vector<double>& y, int window=7);

// Peak detection
int detectPeak(const std::vector<double>& theta,
               const std::vector<double>& intensity,
               double expected=35.0,double window=2.0);

// Voigt function
double voigt(double x, const VoigtParams& p);

// Levenberg-Marquardt Voigt fitting using Eigen
VoigtParams fitVoigtLM(const std::vector<double>& x,
                       const std::vector<double>& y,
                       VoigtParams p,
                       int iterations=300,
                       double lambda_init=1e-4);

// FWHM calculation
double fwhmVoigt(const VoigtParams& p);

// Scherrer crystallite size (wavelength as argument)
double scherrer(double fwhm_deg,double theta_deg,double wavelength=1.5406);

// Residuals & statistical anomaly
std::vector<double> computeResiduals(const std::vector<double>& x,
                                     const std::vector<double>& y,
                                     const VoigtParams& p);

double residualAnomalyScore(const std::vector<double>& residuals);

// Monte Carlo uncertainty
double monteCarloFWHM(const std::vector<double>& x,
                      const std::vector<double>& y,
                      const VoigtParams& p,
                      int n=100);

// Full analysis
void analyze(const std::string& label,
             const std::string& file,
             double wavelength=1.5406);
