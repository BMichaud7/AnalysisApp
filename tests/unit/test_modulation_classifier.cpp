#include <gtest/gtest.h>
#include "ModulationClassifier.hpp"
#include "SignalFeatures.hpp"
#include "AnalysisResult.hpp"

namespace {

// Build a feature set with reasonable defaults and override specific fields
analysis::SignalFeatures makeFeatures()
{
    analysis::SignalFeatures f;
    f.snr_db             = 25.0;
    f.center_freq_hz     = 100e6;
    f.bandwidth_hz       = 200e3;
    f.sample_rate_sps    = 2e6;
    f.sample_count       = 100000;
    f.envelope_mean      = 1.0;
    f.burst_duty_cycle   = 1.0;
    return f;
}

} // anonymous namespace

class ModulationClassifierTest : public ::testing::Test {
protected:
    analysis::ModulationClassifier clf;
};

// ── Low SNR → unclassified ────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, LowSnrUnclassified)
{
    auto f = makeFeatures();
    f.snr_db = 2.0;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_FALSE(r.classified);
    EXPECT_FALSE(r.reject_reason.empty());
}

// ── BPSK features ─────────────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, BpskClassification)
{
    auto f = makeFeatures();
    // BPSK: constant envelope, c40 ≈ -2
    f.envelope_variance_norm = 0.02;
    f.c40_real               = -2.0;
    f.c40_imag               = 0.0;
    f.c42                    = 0.05;
    f.symbol_rate_sps        = 50000.0;
    f.fm_deviation_hz        = 1000.0;   // low

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_TRUE(r.classified);
    EXPECT_EQ(r.digital_modulation, "BPSK");
    EXPECT_EQ(r.m_ary, 2);
}

// ── QPSK features ─────────────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, QpskClassification)
{
    auto f = makeFeatures();
    f.envelope_variance_norm = 0.03;
    f.c40_real               = -2.0;
    f.c40_imag               = 0.0;
    f.c42                    = 0.0;
    f.symbol_rate_sps        = 100000.0;
    f.fm_deviation_hz        = 1000.0;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_TRUE(r.classified);
    EXPECT_EQ(r.digital_modulation, "QPSK");
    EXPECT_EQ(r.m_ary, 4);
}

// ── FM_NB features ────────────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, FmNbClassification)
{
    auto f = makeFeatures();
    f.bandwidth_hz           = 25000.0;
    f.envelope_variance_norm = 0.01;    // constant envelope
    f.fm_deviation_hz        = 5000.0;  // 5 kHz deviation → FM_NB
    f.inst_freq_std_hz       = 5000.0;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_TRUE(r.classified);
    EXPECT_EQ(r.analog_modulation, "FM_NB");
}

// ── FM_WB features ────────────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, FmWbClassification)
{
    auto f = makeFeatures();
    f.bandwidth_hz           = 200000.0;
    f.envelope_variance_norm = 0.01;
    f.fm_deviation_hz        = 75000.0;
    f.inst_freq_std_hz       = 75000.0;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_TRUE(r.classified);
    EXPECT_EQ(r.analog_modulation, "FM_WB");
}

// ── OFDM features ─────────────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, OfdmClassification)
{
    auto f = makeFeatures();
    f.ofdm_detected      = true;
    f.ofdm_fft_size_est  = 512;
    f.ofdm_cp_ratio      = 0.25;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_TRUE(r.classified);
    EXPECT_EQ(r.digital_modulation, "OFDM");
    EXPECT_TRUE(r.is_ofdm);
}

// ── AM DSB-LC features ────────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, AmDsbLcClassification)
{
    auto f = makeFeatures();
    f.bandwidth_hz           = 10000.0;
    f.envelope_variance_norm = 0.35;
    f.am_index               = 0.7;
    f.fm_deviation_hz        = 500.0;    // low deviation
    f.spectral_symmetry      = 0.9;      // symmetric
    f.is_burst               = false;
    f.spectral_flatness      = 0.1;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_TRUE(r.classified);
    EXPECT_EQ(r.analog_modulation, "AM_DSB_LC");
}

// ── Chirp → CSS ───────────────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, ChirpCssClassification)
{
    auto f = makeFeatures();
    f.chirp_detected  = true;
    f.chirp_rate_hz_s = 400000.0;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_TRUE(r.classified);
    EXPECT_EQ(r.digital_modulation, "CSS");
}

// ── QAM16 features ────────────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, Qam16Classification)
{
    auto f = makeFeatures();
    f.envelope_variance_norm = 0.15;    // variable amplitude
    f.c40_real               = -0.68;
    f.c42                    = -0.68;
    f.symbol_rate_sps        = 10e6;
    f.fm_deviation_hz        = 100.0;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_TRUE(r.classified);
    EXPECT_EQ(r.digital_modulation, "QAM16");
    EXPECT_EQ(r.m_ary, 16);
}

// ── Bit-rate computation ──────────────────────────────────────────────────────

TEST_F(ModulationClassifierTest, BitRateFromSymbolRate)
{
    auto f = makeFeatures();
    f.envelope_variance_norm = 0.03;
    f.c40_real               = -2.0;
    f.c42                    = 0.0;
    f.symbol_rate_sps        = 9600.0;
    f.fm_deviation_hz        = 100.0;

    analysis::AnalysisResult r{};
    clf.classify(f, r);

    EXPECT_EQ(r.m_ary, 4);
    EXPECT_NEAR(r.bit_rate_bps, 9600.0 * 2.0, 1.0);  // QPSK → 2 bits/sym
}
