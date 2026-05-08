#include <gtest/gtest.h>
#include "FeatureExtractor.hpp"
#include <cmath>
#include <numbers>
#include <vector>
#include <random>

// ── IQ generation helpers ─────────────────────────────────────────────────────

static constexpr double TWO_PI = 2.0 * std::numbers::pi_v<double>;
static constexpr double SR    = 2'000'000.0;   // 2 MHz sample rate
static constexpr int    N     = 65536;          // sample count

// Pure complex tone at frequency f_hz
static std::vector<float> genTone(double f_hz, int n = N, double sr = SR)
{
    std::vector<float> iq;
    iq.reserve(n * 2);
    for (int i = 0; i < n; ++i) {
        double ph = TWO_PI * f_hz * i / sr;
        iq.push_back(static_cast<float>(std::cos(ph)));
        iq.push_back(static_cast<float>(std::sin(ph)));
    }
    return iq;
}

// BPSK: random ±1 symbols at symbol_rate, upsampled to sr
static std::vector<float> genBpsk(double symbol_rate, int n = N, double sr = SR)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1);

    int sps = static_cast<int>(sr / symbol_rate);
    std::vector<float> iq;
    iq.reserve(n * 2);

    int sym = 0;
    float phase_sign = 1.0f;
    int sym_idx = 0;

    for (int i = 0; i < n; ++i) {
        if (sym_idx == 0)
            phase_sign = (dist(rng) == 0) ? 1.0f : -1.0f;

        iq.push_back(phase_sign);  // I
        iq.push_back(0.0f);        // Q  (real-valued BPSK)

        if (++sym_idx >= sps) sym_idx = 0;
    }
    return iq;
}

// FM modulated with a single tone message
// x[n] = exp(j * 2*pi * deviation/sr * integral(cos(2*pi*fm*t)))
static std::vector<float> genFm(double deviation_hz, double fm_hz,
                                  int n = N, double sr = SR)
{
    std::vector<float> iq;
    iq.reserve(n * 2);
    double phase = 0.0;
    double ph_inc_per_sample = TWO_PI * deviation_hz / sr;
    for (int i = 0; i < n; ++i) {
        double msg = std::cos(TWO_PI * fm_hz * i / sr);
        phase += ph_inc_per_sample * msg;
        iq.push_back(static_cast<float>(std::cos(phase)));
        iq.push_back(static_cast<float>(std::sin(phase)));
    }
    return iq;
}

// AM DSB-LC: x[n] = (1 + m*cos(wm*t)) * cos(wc*t)
static std::vector<float> genAmDsbLc(double mod_index, double fc_hz, double fm_hz,
                                      int n = N, double sr = SR)
{
    std::vector<float> iq;
    iq.reserve(n * 2);
    for (int i = 0; i < n; ++i) {
        double envelope = 1.0 + mod_index * std::cos(TWO_PI * fm_hz * i / sr);
        double carrier_I = std::cos(TWO_PI * fc_hz * i / sr);
        double carrier_Q = std::sin(TWO_PI * fc_hz * i / sr);
        iq.push_back(static_cast<float>(envelope * carrier_I));
        iq.push_back(static_cast<float>(envelope * carrier_Q));
    }
    return iq;
}

// OFDM: sum of N_carriers subcarriers with random QPSK symbols
static std::vector<float> genOfdm(int Nfft, int Ncp, int n_symbols,
                                   int n = N, double /*sr*/ = SR)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 3);

    // QPSK constellation
    static const std::complex<double> qpsk[4] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}
    };

    std::vector<float> iq;
    iq.reserve(n * 2);

    int sym_len = Nfft + Ncp;

    for (int sym = 0; sym < n_symbols && static_cast<int>(iq.size()) / 2 < n; ++sym) {
        // Generate one OFDM symbol in frequency domain
        std::vector<std::complex<double>> fd(Nfft, {0, 0});
        for (int k = 1; k < Nfft; ++k)
            fd[k] = qpsk[dist(rng)];

        // IFFT (manual DFT for small sizes)
        std::vector<std::complex<double>> td(Nfft, {0, 0});
        for (int n_td = 0; n_td < Nfft; ++n_td) {
            for (int k = 0; k < Nfft; ++k) {
                double angle = TWO_PI * k * n_td / Nfft;
                td[n_td] += fd[k] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            td[n_td] /= Nfft;
        }

        // Add cyclic prefix
        for (int k = Nfft - Ncp; k < Nfft; ++k) {
            if (static_cast<int>(iq.size()) / 2 >= n) break;
            iq.push_back(static_cast<float>(td[k].real()));
            iq.push_back(static_cast<float>(td[k].imag()));
        }
        for (int k = 0; k < Nfft; ++k) {
            if (static_cast<int>(iq.size()) / 2 >= n) break;
            iq.push_back(static_cast<float>(td[k].real()));
            iq.push_back(static_cast<float>(td[k].imag()));
        }
    }

    // Pad to n if short
    while (static_cast<int>(iq.size()) / 2 < n) {
        iq.push_back(0.0f);
        iq.push_back(0.0f);
    }

    return iq;
}

// OOK (on-off keying): burst of 50% duty cycle
static std::vector<float> genOok(double fc_hz, int n = N, double sr = SR)
{
    std::vector<float> iq;
    iq.reserve(n * 2);
    int half = n / 2;
    for (int i = 0; i < n; ++i) {
        float on = (i < half) ? 1.0f : 0.0f;
        double ph = TWO_PI * fc_hz * i / sr;
        iq.push_back(static_cast<float>(on * std::cos(ph)));
        iq.push_back(static_cast<float>(on * std::sin(ph)));
    }
    return iq;
}

// Linear chirp: phase = pi * chirp_rate * t^2
static std::vector<float> genChirp(double start_hz, double chirp_rate_hz_s,
                                    int n = N, double sr = SR)
{
    std::vector<float> iq;
    iq.reserve(n * 2);
    for (int i = 0; i < n; ++i) {
        double t  = i / sr;
        double ph = TWO_PI * (start_hz * t + 0.5 * chirp_rate_hz_s * t * t);
        iq.push_back(static_cast<float>(std::cos(ph)));
        iq.push_back(static_cast<float>(std::sin(ph)));
    }
    return iq;
}

// ── Test fixtures ─────────────────────────────────────────────────────────────

class FeatureExtractorTest : public ::testing::Test {
protected:
    analysis::FeatureExtractor extractor_{4096};
};

// ── BPSK test ─────────────────────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, BpskCumulants)
{
    double symbol_rate = 50000.0;  // 50 ksps
    auto iq = genBpsk(symbol_rate);

    auto f = extractor_.extract(iq, SR, 0.0);

    // BPSK theoretical: c40 = -2, c42 = 0
    // Allow generous tolerance for finite-sample estimation
    EXPECT_NEAR(f.c40_real, -2.0, 0.6)
        << "c40_real for BPSK should be near -2";
    EXPECT_NEAR(f.c42, 0.0, 0.8)
        << "c42 for BPSK should be near 0";

    // Symbol rate detection
    if (f.symbol_rate_sps > 0) {
        EXPECT_NEAR(f.symbol_rate_sps, symbol_rate, symbol_rate * 0.3)
            << "Symbol rate estimate should be near " << symbol_rate;
    }

    // Constant envelope (BPSK with rectangular pulse has very low variance)
    EXPECT_LT(f.envelope_variance_norm, 0.2)
        << "BPSK envelope variance should be low";
}

// ── FM test ───────────────────────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, FmDeviation)
{
    double deviation = 75000.0;   // FM broadcast ±75 kHz
    double fm_freq   = 1000.0;    // 1 kHz message
    auto iq = genFm(deviation, fm_freq);

    auto f = extractor_.extract(iq, SR, 100e6);

    // FM: envelope should be constant (unit circle)
    EXPECT_LT(f.envelope_variance_norm, 0.05)
        << "FM has constant envelope, variance should be < 0.05";

    // FM deviation estimate from instantaneous frequency std
    EXPECT_GT(f.fm_deviation_hz, deviation * 0.4)
        << "FM deviation estimate too low";
    EXPECT_LT(f.fm_deviation_hz, deviation * 2.0)
        << "FM deviation estimate too high";
}

// ── AM test ───────────────────────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, AmDsbLc)
{
    double mod_index = 0.7;
    double fc = 50000.0;   // 50 kHz carrier offset
    double fm = 1000.0;    // 1 kHz message
    auto iq = genAmDsbLc(mod_index, fc, fm);

    auto f = extractor_.extract(iq, SR, 1e6);

    // AM-DSB-LC: envelope varies, high envelope_variance_norm
    EXPECT_GT(f.envelope_variance_norm, 0.1)
        << "AM DSB-LC should have significant envelope variance";

    // AM index estimate
    EXPECT_NEAR(f.am_index, mod_index, 0.4)
        << "AM index estimate should be near " << mod_index;
}

// ── OFDM detection test ───────────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, OfdmDetected)
{
    // Small OFDM: Nfft=64, Ncp=16 (1/4 CP ratio)
    // Need enough symbols: at min 4 * (64+16) = 320 samples, we have 65536
    int Nfft = 64, Ncp = 16;
    int n_symbols = N / (Nfft + Ncp) + 1;
    auto iq = genOfdm(Nfft, Ncp, n_symbols);

    auto f = extractor_.extract(iq, SR, 0.0);

    EXPECT_TRUE(f.ofdm_detected)
        << "OFDM should be detected for Nfft=" << Nfft << " Ncp=" << Ncp;
    if (f.ofdm_detected) {
        EXPECT_EQ(f.ofdm_fft_size_est, Nfft)
            << "OFDM FFT size estimate should match";
        EXPECT_NEAR(f.ofdm_cp_ratio, 0.25, 0.05)
            << "OFDM CP ratio estimate should match";
    }
}

// ── OOK / Burst test ─────────────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, OokBurstDetection)
{
    auto iq = genOok(100000.0);  // 100 kHz tone, first half ON, second half OFF

    auto f = extractor_.extract(iq, SR, 0.0);

    EXPECT_TRUE(f.is_burst)
        << "OOK should be detected as burst";
    EXPECT_NEAR(f.burst_duty_cycle, 0.5, 0.15)
        << "OOK duty cycle should be ~0.5";
}

// ── Chirp detection test ──────────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, ChirpDetected)
{
    // Chirp from -200 kHz to +200 kHz over the full sample duration
    double chirp_rate = 400000.0 / ((double)N / SR);   // Hz/s
    auto iq = genChirp(-200000.0, chirp_rate);

    auto f = extractor_.extract(iq, SR, 0.0);

    EXPECT_TRUE(f.chirp_detected)
        << "Linear chirp should be detected";
    if (f.chirp_detected) {
        EXPECT_GT(std::abs(f.chirp_rate_hz_s), chirp_rate * 0.3)
            << "Chirp rate estimate should be non-trivial";
    }
}

// ── SNR test ─────────────────────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, SnrIsPositiveForCleanTone)
{
    auto iq = genTone(100000.0);
    auto f = extractor_.extract(iq, SR, 0.0);

    EXPECT_GT(f.snr_db, 10.0)
        << "Clean tone should have SNR > 10 dB";
    EXPECT_GT(f.bandwidth_hz, 0.0)
        << "Bandwidth should be positive";
}

// ── Sample count passthrough ──────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, SampleCountMatchesInput)
{
    auto iq = genTone(50000.0, N);
    auto f = extractor_.extract(iq, SR, 0.0);
    EXPECT_EQ(f.sample_count, N);
    EXPECT_DOUBLE_EQ(f.sample_rate_sps, SR);
}

// ── Empty input guard ─────────────────────────────────────────────────────────

TEST_F(FeatureExtractorTest, EmptyInputReturnsZeroFeatures)
{
    std::vector<float> empty;
    auto f = extractor_.extract(empty, SR, 0.0);
    EXPECT_EQ(f.sample_count, 0);
    EXPECT_DOUBLE_EQ(f.snr_db, 0.0);
}
