#pragma once
/**
 * @file FeatureExtractor.hpp
 * @brief Extracts a SignalFeatures vector from a block of IQ samples.
 *
 * Feature extraction covers:
 * - PSD computation (Welch), spectral flatness, symmetry
 * - Envelope statistics and instantaneous frequency/phase
 * - Higher-order cumulants (C40, C42, C41) for modulation discrimination
 * - Symbol rate estimation via cyclostationary analysis
 * - Structural detection: OFDM (CP autocorrelation), FHSS (spectrogram),
 *   DSSS (chip-rate peak), chirp (Wigner-Ville), burst (envelope gating)
 *
 * One FeatureExtractor per analysis pipeline (owns FFTW plans).
 * Thread-safe: extract() is const; FFTW plans are read-only after construction.
 */
#include "SignalFeatures.hpp"
#include <au/units/hertz.hh>
#include <vector>
#include <complex>

namespace analysis {

/// @brief Extracts a SignalFeatures vector from raw IQ samples.
class FeatureExtractor {
public:
    /**
     * @brief Construct the extractor and allocate FFTW plans.
     * @param fft_size FFT size for PSD and cyclostationary analysis.
     */
    explicit FeatureExtractor(int fft_size = 4096);
    ~FeatureExtractor();

    /**
     * @brief Extract all features from a block of IQ samples.
     * @param iq              Interleaved float32 samples (I,Q,I,Q,…).
     * @param sample_rate_sps Sample rate (samples/s).
     * @param center_freq_hz  Centre frequency (Hz); stored verbatim in the result.
     * @return Fully populated SignalFeatures.
     */
    SignalFeatures extract(const std::vector<float>& iq,
                           au::QuantityD<au::Hertz> sample_rate_sps,
                           au::QuantityD<au::Hertz> center_freq_hz) const;

private:
    int fft_size_;
    void* fft_plan_  = nullptr;  ///< FFTW plan for PSD.
    void* fft_plan2_ = nullptr;  ///< FFTW plan for cyclostationary analysis.

    void computePsd(const std::vector<std::complex<float>>& x,
                    std::vector<float>& psd_out) const;

    void computeCumulants(const std::vector<std::complex<float>>& x,
                          double& c40r, double& c40i,
                          double& c41r, double& c42) const;

    double estimateSymbolRate(const std::vector<std::complex<float>>& x,
                               au::QuantityD<au::Hertz> sample_rate_sps) const;

    bool detectOfdm(const std::vector<std::complex<float>>& x,
                    au::QuantityD<au::Hertz> sample_rate_sps,
                    int& fft_size_out, double& cp_ratio_out) const;

    bool detectFhss(const std::vector<std::complex<float>>& x,
                    au::QuantityD<au::Hertz> sample_rate_sps,
                    au::QuantityD<au::Hertz>& hop_rate_out) const;

    bool detectChirp(const std::vector<std::complex<float>>& x,
                     au::QuantityD<au::Hertz> sample_rate_sps,
                     double& chirp_rate_out) const;

    void computeInstFreqStats(const std::vector<std::complex<float>>& x,
                               au::QuantityD<au::Hertz> sample_rate_sps,
                               au::QuantityD<au::Hertz>& mean,
                               au::QuantityD<au::Hertz>& std_dev) const;

    bool detectBurst(const std::vector<std::complex<float>>& x,
                     double& duty_cycle, double& period_samples) const;
};

} // namespace analysis
