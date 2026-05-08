#pragma once
#include "SignalFeatures.hpp"
#include <vector>
#include <complex>

namespace analysis {

class FeatureExtractor {
public:
    explicit FeatureExtractor(int fft_size = 4096);
    ~FeatureExtractor();

    // Main entry: CF32 interleaved I/Q samples → feature vector
    SignalFeatures extract(const std::vector<float>& iq,
                           double sample_rate_sps,
                           double center_freq_hz) const;

private:
    int fft_size_;
    void* fft_plan_  = nullptr;  // FFTW plan
    void* fft_plan2_ = nullptr;  // second plan for cyclostationary

    void computePsd(const std::vector<std::complex<float>>& x,
                    std::vector<float>& psd_out) const;

    void computeCumulants(const std::vector<std::complex<float>>& x,
                          double& c40r, double& c40i,
                          double& c41r, double& c42) const;

    double estimateSymbolRate(const std::vector<std::complex<float>>& x,
                               double sample_rate_sps) const;

    bool detectOfdm(const std::vector<std::complex<float>>& x,
                    double sample_rate_sps,
                    int& fft_size_out, double& cp_ratio_out) const;

    bool detectFhss(const std::vector<std::complex<float>>& x,
                    double sample_rate_sps,
                    double& hop_rate_hz_out) const;

    bool detectChirp(const std::vector<std::complex<float>>& x,
                     double sample_rate_sps,
                     double& chirp_rate_out) const;

    void computeInstFreqStats(const std::vector<std::complex<float>>& x,
                               double sample_rate_sps,
                               double& mean_hz, double& std_hz) const;

    bool detectBurst(const std::vector<std::complex<float>>& x,
                     double& duty_cycle, double& period_samples) const;
};

} // namespace analysis
