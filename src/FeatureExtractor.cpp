#include "FeatureExtractor.hpp"
#include <fftw3.h>
#include <numbers>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace analysis {

using cf32 = std::complex<float>;

// ── Constructor / Destructor ──────────────────────────────────────────────────

FeatureExtractor::FeatureExtractor(int fft_size)
    : fft_size_(fft_size)
{
    // Allocate persistent FFTW plans for the primary FFT size
    auto* in1  = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size_));
    auto* out1 = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size_));
    fft_plan_ = fftwf_plan_dft_1d(fft_size_, in1, out1,
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_free(in1);
    fftwf_free(out1);

    // Second plan for cyclostationary / symbol-rate FFT (same size)
    auto* in2  = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size_));
    auto* out2 = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size_));
    fft_plan2_ = fftwf_plan_dft_1d(fft_size_, in2, out2,
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_free(in2);
    fftwf_free(out2);
}

FeatureExtractor::~FeatureExtractor()
{
    if (fft_plan_)  fftwf_destroy_plan(reinterpret_cast<fftwf_plan>(fft_plan_));
    if (fft_plan2_) fftwf_destroy_plan(reinterpret_cast<fftwf_plan>(fft_plan2_));
}

// ── Utility helpers ───────────────────────────────────────────────────────────

static std::vector<cf32> toComplex(const std::vector<float>& iq)
{
    std::vector<cf32> out;
    out.reserve(iq.size() / 2);
    for (size_t i = 0; i + 1 < iq.size(); i += 2)
        out.emplace_back(iq[i], iq[i+1]);
    return out;
}

static std::vector<float> hannWindow(int n)
{
    std::vector<float> w(n);
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    for (int i = 0; i < n; ++i)
        w[i] = static_cast<float>(0.5 * (1.0 - std::cos(two_pi * i / (n - 1))));
    return w;
}

// FFT a block of complex<float> samples; result has fft_size_ bins
static void runFftPlan(fftwf_plan plan,
                       const cf32* in, int fft_size,
                       std::vector<cf32>& out)
{
    auto* fin  = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size));
    auto* fout = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size));

    for (int i = 0; i < fft_size; ++i) {
        fin[i][0] = in[i].real();
        fin[i][1] = in[i].imag();
    }
    fftwf_execute_dft(plan, fin, fout);

    out.resize(fft_size);
    for (int i = 0; i < fft_size; ++i)
        out[i] = cf32(fout[i][0], fout[i][1]);

    fftwf_free(fin);
    fftwf_free(fout);
}

// ── PSD (Welch method, 8 frames, 50% overlap, Hann window) ───────────────────

void FeatureExtractor::computePsd(const std::vector<cf32>& x,
                                   std::vector<float>& psd_out) const
{
    const int N   = fft_size_;
    const int hop = N / 2;                       // 50% overlap
    const int frames = 8;
    auto hann = hannWindow(N);

    psd_out.assign(N, 0.0f);

    int frame_count = 0;
    for (int f = 0; f < frames; ++f) {
        int offset = f * hop;
        if (offset + N > static_cast<int>(x.size())) break;

        std::vector<cf32> windowed(N);
        for (int i = 0; i < N; ++i)
            windowed[i] = x[offset + i] * hann[i];

        std::vector<cf32> spec;
        runFftPlan(reinterpret_cast<fftwf_plan>(fft_plan_),
                   windowed.data(), N, spec);

        for (int i = 0; i < N; ++i)
            psd_out[i] += std::norm(spec[i]);

        ++frame_count;
    }

    if (frame_count == 0) return;

    // Normalize: divide by frame count and window power, then scale so bins
    // sum to total power.
    float win_power = 0;
    for (float w : hann) win_power += w * w;

    float total = 0;
    for (int i = 0; i < N; ++i) {
        psd_out[i] /= (frame_count * win_power);
        total += psd_out[i];
    }

    // Rearrange to DC-centered (fftshift)
    std::rotate(psd_out.begin(), psd_out.begin() + N/2, psd_out.end());
}

// ── Cumulants ─────────────────────────────────────────────────────────────────

void FeatureExtractor::computeCumulants(const std::vector<cf32>& x,
                                         double& c40r, double& c40i,
                                         double& c41r, double& c42) const
{
    const int N = static_cast<int>(x.size());
    if (N == 0) { c40r = c40i = c41r = c42 = 0; return; }

    // Moments
    std::complex<double> M20{0,0};
    double               M21 = 0;
    std::complex<double> M40{0,0};
    std::complex<double> M41{0,0};
    double               M42 = 0;

    for (const auto& s : x) {
        std::complex<double> sd(s.real(), s.imag());
        double mag2 = std::norm(sd);

        M20 += sd * sd;
        M21 += mag2;
        M40 += sd * sd * sd * sd;
        M41 += sd * sd * sd * std::conj(sd);
        M42 += mag2 * mag2;
    }

    M20 /= N;
    M21 /= N;
    M40 /= N;
    M41 /= N;
    M42 /= N;

    double M21_sq = M21 * M21;
    if (M21_sq < 1e-30) { c40r = c40i = c41r = c42 = 0; return; }

    // Cumulants
    std::complex<double> C40 = M40 - 3.0 * M20 * M20;
    std::complex<double> C41 = M41 - 3.0 * M20 * M21;
    double               C42 = M42 - std::norm(M20) - 2.0 * M21_sq;

    c40r = C40.real() / M21_sq;
    c40i = C40.imag() / M21_sq;
    c41r = C41.real() / M21_sq;
    c42  = C42        / M21_sq;
}

// ── Symbol-rate estimation ────────────────────────────────────────────────────

double FeatureExtractor::estimateSymbolRate(const std::vector<cf32>& x,
                                             double sample_rate_sps) const
{
    const int N = fft_size_;
    if (static_cast<int>(x.size()) < N) return 0.0;

    // Compute power signal y[n] = |x[n]|^2 - mean and fourth power z[n] = x[n]^4
    std::vector<float> env2(N);
    for (int i = 0; i < N; ++i) env2[i] = std::norm(x[i]);
    float mean_env2 = std::accumulate(env2.begin(), env2.end(), 0.0f) / N;

    // Squared spectrum
    std::vector<cf32> y_sq(N);
    for (int i = 0; i < N; ++i)
        y_sq[i] = cf32(env2[i] - mean_env2, 0.0f);

    std::vector<cf32> spec_sq;
    runFftPlan(reinterpret_cast<fftwf_plan>(fft_plan2_),
               y_sq.data(), N, spec_sq);

    // Fourth-power spectrum
    std::vector<cf32> y4(N);
    for (int i = 0; i < N; ++i) {
        cf32 s2 = x[i] * x[i];
        y4[i] = s2 * s2;
    }
    std::vector<cf32> spec_4;
    runFftPlan(reinterpret_cast<fftwf_plan>(fft_plan_),
               y4.data(), N, spec_4);

    // Search for peak in [1 .. N/2-1] bins (positive frequencies)
    // Skip DC (bin 0)
    float best_sq  = 0, best_4 = 0;
    int   bin_sq   = 0, bin_4  = 0;

    for (int k = 1; k < N/2; ++k) {
        float sq_mag = std::abs(spec_sq[k]);
        float fo_mag = std::abs(spec_4[k]);
        if (sq_mag > best_sq) { best_sq = sq_mag; bin_sq = k; }
        if (fo_mag > best_4)  { best_4  = fo_mag; bin_4  = k; }
    }

    // Convert winning bin to frequency
    double sr_from_sq = (bin_sq > 0) ? (double)bin_sq * sample_rate_sps / N : 0.0;
    double sr_from_4  = (bin_4  > 0) ? (double)bin_4  * sample_rate_sps / N / 4.0 : 0.0;

    // For fourth-power method, the symbol rate = peak_freq / 2 for M-PSK
    // (the squarer gives symbol-rate spurs at symbol_rate * n, fourth-power
    //  gives spurs at 4 * symbol_rate for BPSK/QPSK).
    // Pick whichever has a stronger normalized peak.
    float noise_floor_sq = 0;
    for (int k = 1; k < N/2; ++k) noise_floor_sq += std::abs(spec_sq[k]);
    noise_floor_sq /= (N/2 - 1);

    double snr_sq = (noise_floor_sq > 1e-12f) ? best_sq / noise_floor_sq : 0.0;
    double snr_4  = 0;
    {
        float nf4 = 0;
        for (int k = 1; k < N/2; ++k) nf4 += std::abs(spec_4[k]);
        nf4 /= (N/2 - 1);
        snr_4 = (nf4 > 1e-12f) ? best_4 / nf4 : 0.0;
    }

    // Minimum SNR needed to trust the estimate
    const double MIN_SR_SNR = 5.0;
    if (snr_sq < MIN_SR_SNR && snr_4 < MIN_SR_SNR) return 0.0;

    return (snr_sq >= snr_4) ? sr_from_sq : sr_from_4;
}

// ── OFDM detection (cyclic-prefix correlation) ───────────────────────────────

bool FeatureExtractor::detectOfdm(const std::vector<cf32>& x,
                                   double /*sample_rate_sps*/,
                                   int& fft_size_out,
                                   double& cp_ratio_out) const
{
    // Candidate OFDM FFT sizes and CP fractions
    static const int   nffts[]    = {64, 128, 256, 512, 1024, 2048};
    static const double cp_fracs[] = {1.0/4, 1.0/8, 1.0/16};

    const int total = static_cast<int>(x.size());
    double best_score = 0.0;
    bool found = false;

    for (int nfft : nffts) {
        for (double cp_frac : cp_fracs) {
            int ncp = static_cast<int>(std::round(nfft * cp_frac));
            if (ncp < 1) continue;
            int sym_len = nfft + ncp;
            if (sym_len * 4 > total) continue;  // need several symbols

            // Compute cyclic-prefix correlator:
            // R = sum_k |x[n+k] * conj(x[n+k+Nfft])| for k=0..Ncp-1
            // Averaged over several symbol positions.
            double sum_corr  = 0.0;
            double sum_power = 0.0;
            int    n_syms    = 0;

            for (int n = 0; n + sym_len <= total; n += sym_len) {
                double corr = 0.0;
                double pwr  = 0.0;
                for (int k = 0; k < ncp; ++k) {
                    cf32 a = x[n + k];
                    cf32 b = x[n + k + nfft];
                    corr += std::abs(a * std::conj(b));
                    pwr  += (std::norm(a) + std::norm(b)) * 0.5f;
                }
                sum_corr  += corr;
                sum_power += pwr;
                ++n_syms;
            }

            if (n_syms == 0 || sum_power < 1e-12) continue;

            double score = sum_corr / sum_power / ncp;

            if (score > best_score) {
                best_score   = score;
                fft_size_out = nfft;
                cp_ratio_out = cp_frac;
            }
        }
    }

    if (best_score > 0.7) {
        found = true;
        spdlog::debug("OFDM detected: score={:.3f} Nfft={} cp_ratio={:.3f}",
                      best_score, fft_size_out, cp_ratio_out);
    }
    return found;
}

// ── FHSS detection ────────────────────────────────────────────────────────────

bool FeatureExtractor::detectFhss(const std::vector<cf32>& x,
                                   double sample_rate_sps,
                                   double& hop_rate_hz_out) const
{
    const int total = static_cast<int>(x.size());
    const int num_segs = 32;
    const int seg_len  = total / num_segs;
    if (seg_len < fft_size_) return false;

    // For each segment compute spectral centroid
    std::vector<double> centroids(num_segs);
    for (int s = 0; s < num_segs; ++s) {
        int offset = s * seg_len;
        int n = std::min(fft_size_, seg_len);

        std::vector<cf32> block(x.begin() + offset, x.begin() + offset + n);
        auto hann = hannWindow(n);
        for (int i = 0; i < n; ++i) block[i] *= hann[i];

        std::vector<cf32> spec;
        runFftPlan(reinterpret_cast<fftwf_plan>(fft_plan_),
                   block.data(), n, spec);

        double num = 0, den = 0;
        for (int k = 0; k < n; ++k) {
            double f  = (double)k / n * sample_rate_sps - sample_rate_sps / 2.0;
            double pw = std::norm(spec[(k + n/2) % n]);
            num += f * pw;
            den += pw;
        }
        centroids[s] = (den > 1e-20) ? num / den : 0.0;
    }

    // Compute std of centroids
    double mean_c = std::accumulate(centroids.begin(), centroids.end(), 0.0) / num_segs;
    double var_c  = 0;
    for (double c : centroids) var_c += (c - mean_c) * (c - mean_c);
    var_c /= num_segs;
    double std_c = std::sqrt(var_c);

    // Bandwidth estimate from signal itself
    double sig_bw = sample_rate_sps * 0.5; // conservative

    if (std_c > sig_bw / 4.0) {
        // Estimate hop rate from transitions between segments
        int transitions = 0;
        for (int s = 1; s < num_segs; ++s)
            if (std::abs(centroids[s] - centroids[s-1]) > std_c * 0.5)
                ++transitions;

        double total_time_s = (double)total / sample_rate_sps;
        hop_rate_hz_out = (double)transitions / total_time_s;
        return true;
    }
    return false;
}

// ── Chirp detection ───────────────────────────────────────────────────────────

bool FeatureExtractor::detectChirp(const std::vector<cf32>& x,
                                    double sample_rate_sps,
                                    double& chirp_rate_out) const
{
    const int N = std::min(1000, static_cast<int>(x.size()) - 1);
    if (N < 10) return false;

    // Instantaneous frequency
    std::vector<double> fi(N);
    for (int i = 0; i < N; ++i) {
        cf32 prod = x[i + 1] * std::conj(x[i]);
        fi[i] = std::arg(prod) * sample_rate_sps / (2.0 * std::numbers::pi_v<double>);
    }

    // Linear regression on fi vs sample index
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int i = 0; i < N; ++i) {
        sum_x  += i;
        sum_y  += fi[i];
        sum_xx += (double)i * i;
        sum_xy += (double)i * fi[i];
    }
    double denom = N * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-12) return false;

    double slope     = (N * sum_xy - sum_x * sum_y) / denom;
    double intercept = (sum_y - slope * sum_x) / N;

    // Compute R^2
    double ss_res = 0, ss_tot = 0;
    double mean_y = sum_y / N;
    for (int i = 0; i < N; ++i) {
        double pred = slope * i + intercept;
        ss_res += (fi[i] - pred) * (fi[i] - pred);
        ss_tot += (fi[i] - mean_y) * (fi[i] - mean_y);
    }
    double r2 = (ss_tot > 1e-12) ? 1.0 - ss_res / ss_tot : 0.0;

    // Slope in Hz/sample → convert to Hz/s
    double slope_hz_s = slope * sample_rate_sps;

    if (r2 > 0.95 && std::abs(slope_hz_s) > sample_rate_sps / 100.0) {
        chirp_rate_out = slope_hz_s;
        return true;
    }
    return false;
}

// ── Instantaneous frequency statistics ───────────────────────────────────────

void FeatureExtractor::computeInstFreqStats(const std::vector<cf32>& x,
                                             double sample_rate_sps,
                                             double& mean_hz, double& std_hz) const
{
    const int N = static_cast<int>(x.size()) - 1;
    if (N < 2) { mean_hz = std_hz = 0; return; }

    std::vector<double> fi(N);
    for (int i = 0; i < N; ++i) {
        cf32 prod = x[i + 1] * std::conj(x[i]);
        fi[i] = std::arg(prod) * sample_rate_sps / (2.0 * std::numbers::pi_v<double>);
    }

    double sum = std::accumulate(fi.begin(), fi.end(), 0.0);
    mean_hz = sum / N;

    double var = 0;
    for (double f : fi) var += (f - mean_hz) * (f - mean_hz);
    std_hz = std::sqrt(var / N);
}

// ── Burst detection ───────────────────────────────────────────────────────────

bool FeatureExtractor::detectBurst(const std::vector<cf32>& x,
                                    double& duty_cycle,
                                    double& period_samples) const
{
    const int total    = static_cast<int>(x.size());
    const int windows  = 100;
    const int win_size = total / windows;
    if (win_size < 1) { duty_cycle = 1.0; period_samples = 0; return false; }

    std::vector<double> power(windows);
    double peak_pwr = 0;
    for (int w = 0; w < windows; ++w) {
        double p = 0;
        for (int i = w * win_size; i < (w + 1) * win_size; ++i)
            p += std::norm(x[i]);
        power[w] = p / win_size;
        if (power[w] > peak_pwr) peak_pwr = power[w];
    }

    const double threshold = peak_pwr * 0.1;
    int on_count = 0;
    std::vector<int> on_off(windows);
    for (int w = 0; w < windows; ++w) {
        on_off[w] = (power[w] > threshold) ? 1 : 0;
        on_count += on_off[w];
    }

    duty_cycle = (double)on_count / windows;

    if (duty_cycle >= 0.85) {
        period_samples = 0;
        return false;
    }

    // Estimate period via autocorrelation of on/off sequence
    int best_lag = 0;
    double best_ac = -1e9;
    for (int lag = 2; lag < windows / 2; ++lag) {
        double ac = 0;
        for (int w = 0; w < windows - lag; ++w)
            ac += on_off[w] * on_off[w + lag];
        if (ac > best_ac) {
            best_ac  = ac;
            best_lag = lag;
        }
    }

    period_samples = (best_lag > 0)
                     ? (double)best_lag * win_size
                     : 0.0;
    return true;
}

// ── Main extraction entry ─────────────────────────────────────────────────────

SignalFeatures FeatureExtractor::extract(const std::vector<float>& iq,
                                          double sample_rate_sps,
                                          double center_freq_hz) const
{
    SignalFeatures f;
    f.center_freq_hz  = center_freq_hz;
    f.sample_rate_sps = sample_rate_sps;
    f.sample_count    = static_cast<int>(iq.size() / 2);

    if (f.sample_count < fft_size_) {
        spdlog::warn("FeatureExtractor: too few samples ({} < {})",
                     f.sample_count, fft_size_);
        return f;
    }

    auto x = toComplex(iq);
    const int N_samp = static_cast<int>(x.size());

    // ── PSD ──────────────────────────────────────────────────────────────
    std::vector<float> psd;
    computePsd(x, psd);

    const int N_psd = static_cast<int>(psd.size());
    double total_power = 0;
    for (float p : psd) total_power += p;

    // Peak bin and noise floor (bottom 20%)
    float peak_psd = *std::max_element(psd.begin(), psd.end());

    std::vector<float> sorted_psd(psd);
    std::sort(sorted_psd.begin(), sorted_psd.end());
    int noise_bins = std::max(1, N_psd / 5);
    double noise_sum = 0;
    for (int i = 0; i < noise_bins; ++i) noise_sum += sorted_psd[i];
    double noise_floor = noise_sum / noise_bins;
    double noise_power = noise_floor * N_psd;

    double signal_power = total_power - noise_power;
    if (signal_power < 0) signal_power = 0;

    // SNR
    if (noise_power > 1e-30)
        f.snr_db = 10.0 * std::log10(signal_power / noise_power + 1e-20);
    else
        f.snr_db = 60.0;

    // Occupied bandwidth at -10 dB from peak
    float threshold = peak_psd / 10.0f;   // -10 dB
    int peak_bin = static_cast<int>(
        std::max_element(psd.begin(), psd.end()) - psd.begin());

    int lo = peak_bin, hi = peak_bin;
    while (lo > 0         && psd[lo - 1] > threshold) --lo;
    while (hi < N_psd - 1 && psd[hi + 1] > threshold) ++hi;

    f.bandwidth_hz = (double)(hi - lo) * sample_rate_sps / N_psd;

    // Spectral flatness in signal band (geometric mean / arithmetic mean)
    {
        double log_sum = 0, lin_sum = 0;
        int band_bins = hi - lo + 1;
        for (int k = lo; k <= hi; ++k) {
            float p = std::max(psd[k], 1e-20f);
            log_sum += std::log(p);
            lin_sum += p;
        }
        double geom = std::exp(log_sum / band_bins);
        double arith = lin_sum / band_bins;
        f.spectral_flatness = (arith > 1e-30) ? geom / arith : 0.0;
    }

    // Spectral symmetry (correlation of left/right halves around center)
    {
        int half = (hi - lo) / 2;
        if (half > 0) {
            int center = (lo + hi) / 2;
            double cov = 0, var_l = 0, var_r = 0;
            double mean_l = 0, mean_r = 0;
            for (int k = 0; k < half; ++k) {
                mean_l += psd[center - k - 1];
                mean_r += psd[center + k + 1];
            }
            mean_l /= half; mean_r /= half;
            for (int k = 0; k < half; ++k) {
                double dl = psd[center - k - 1] - mean_l;
                double dr = psd[center + k + 1] - mean_r;
                cov   += dl * dr;
                var_l += dl * dl;
                var_r += dr * dr;
            }
            double denom = std::sqrt(var_l * var_r);
            f.spectral_symmetry = (denom > 1e-20) ? cov / denom : 0.0;
            f.spectral_symmetry = std::max(-1.0, std::min(1.0, f.spectral_symmetry));
        } else {
            f.spectral_symmetry = 1.0;
        }
    }

    // ── Envelope statistics ───────────────────────────────────────────────
    {
        std::vector<float> env(N_samp);
        float env_min = 1e30f, env_max = 0.0f;
        double env_sum = 0;
        for (int i = 0; i < N_samp; ++i) {
            float e = std::abs(x[i]);
            env[i]   = e;
            env_sum += e;
            if (e < env_min) env_min = e;
            if (e > env_max) env_max = e;
        }
        f.envelope_mean = env_sum / N_samp;

        double var = 0;
        for (float e : env) var += ((double)e - f.envelope_mean) * ((double)e - f.envelope_mean);
        f.envelope_std = std::sqrt(var / N_samp);

        double mean2 = f.envelope_mean * f.envelope_mean;
        f.envelope_variance_norm = (mean2 > 1e-30)
                                   ? (var / N_samp) / mean2
                                   : 0.0;

        // AM index
        float sum_env_fb = env_max + env_min;
        f.am_index = (sum_env_fb > 1e-12f)
                     ? (double)(env_max - env_min) / sum_env_fb
                     : 0.0;
    }

    // ── Instantaneous frequency stats ─────────────────────────────────────
    computeInstFreqStats(x, sample_rate_sps,
                         f.inst_freq_mean_hz, f.inst_freq_std_hz);

    // Instantaneous phase std
    {
        // Phase of each sample
        double prev_phase = 0, phase_sum = 0;
        std::vector<double> phases(N_samp);
        for (int i = 0; i < N_samp; ++i) {
            phases[i] = std::arg(x[i]);
            phase_sum += phases[i];
        }
        double mean_ph = phase_sum / N_samp;
        double var_ph = 0;
        for (double ph : phases) var_ph += (ph - mean_ph) * (ph - mean_ph);
        f.inst_phase_std = std::sqrt(var_ph / N_samp);
        (void)prev_phase;
    }

    // FM deviation estimate
    f.fm_deviation_hz = f.inst_freq_std_hz;

    // ── Higher-order cumulants ────────────────────────────────────────────
    computeCumulants(x, f.c40_real, f.c40_imag, f.c41_real, f.c42);

    // ── Symbol rate ───────────────────────────────────────────────────────
    f.symbol_rate_sps = estimateSymbolRate(x, sample_rate_sps);

    // ── Structural features ───────────────────────────────────────────────
    double period_samp = 0;
    f.is_burst = detectBurst(x, f.burst_duty_cycle, period_samp);
    if (period_samp > 0)
        f.burst_period_ms = period_samp / sample_rate_sps * 1000.0;

    f.ofdm_detected = detectOfdm(x, sample_rate_sps,
                                  f.ofdm_fft_size_est, f.ofdm_cp_ratio);

    f.fhss_detected = detectFhss(x, sample_rate_sps, f.fhss_hop_rate_hz);

    double chirp_rate = 0;
    f.chirp_detected  = detectChirp(x, sample_rate_sps, chirp_rate);
    f.chirp_rate_hz_s = chirp_rate;

    return f;
}

} // namespace analysis
