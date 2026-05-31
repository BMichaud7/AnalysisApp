#include <au/prefix.hh>
#include <au/units/seconds.hh>
#include <au/units/hertz.hh>
#include <gtest/gtest.h>
#include "ProtocolMapper.hpp"
#include "SignalFeatures.hpp"
#include "AnalysisResult.hpp"

namespace {

analysis::SignalFeatures makeAisFeatures()
{
    analysis::SignalFeatures f;
    f.center_freq_hz = au::hertz(162.025e6);   // AIS channel 2
    f.bandwidth_hz = au::hertz(25000.0);
    f.snr_db             = 22.0;
    f.sample_rate_sps = au::hertz(2e6);
    f.sample_count       = 200000;
    f.envelope_variance_norm = 0.01;    // constant envelope (GMSK)
    f.is_burst           = true;
    f.burst_duty_cycle   = 0.8;
    f.burst_period_ms = au::milli(au::seconds)(26.67);       // TDMA slot timing
    return f;
}

analysis::AnalysisResult makeAisResult(const analysis::SignalFeatures& f)
{
    analysis::AnalysisResult r{};
    r.center_freq_hz     = f.center_freq_hz;
    r.bandwidth_hz       = f.bandwidth_hz;
    r.snr_db             = f.snr_db;
    r.digital_modulation = "GMSK";
    r.symbol_rate_sps = au::hertz(9600.0);
    r.m_ary              = 2;
    r.is_burst           = true;
    r.is_tdma            = true;
    r.burst_duty_cycle   = 0.8;
    r.classified         = true;
    return r;
}

analysis::SignalFeatures makeFmBroadcastFeatures()
{
    analysis::SignalFeatures f;
    f.center_freq_hz = au::hertz(100.1e6);
    f.bandwidth_hz = au::hertz(200000.0);
    f.snr_db             = 35.0;
    f.sample_rate_sps = au::hertz(2e6);
    f.sample_count       = 200000;
    f.envelope_variance_norm = 0.01;
    f.fm_deviation_hz = au::hertz(75000.0);
    f.is_burst           = false;
    f.burst_duty_cycle   = 1.0;
    return f;
}

analysis::AnalysisResult makeFmBroadcastResult(const analysis::SignalFeatures& f)
{
    analysis::AnalysisResult r{};
    r.center_freq_hz     = f.center_freq_hz;
    r.bandwidth_hz       = f.bandwidth_hz;
    r.snr_db             = f.snr_db;
    r.analog_modulation  = "FM_WB";
    r.analog_index       = 75000.0;
    r.symbol_rate_sps = au::hertz(0.0);
    r.is_burst           = false;
    r.burst_duty_cycle   = 1.0;
    r.classified         = true;
    return r;
}

analysis::SignalFeatures makeGsmFeatures()
{
    analysis::SignalFeatures f;
    f.center_freq_hz = au::hertz(935.2e6);   // GSM 900 downlink
    f.bandwidth_hz = au::hertz(200000.0);
    f.snr_db             = 20.0;
    f.sample_rate_sps = au::hertz(2e6);
    f.sample_count       = 200000;
    f.envelope_variance_norm = 0.01;
    f.is_burst           = true;
    f.burst_duty_cycle   = 0.75;
    f.burst_period_ms = au::milli(au::seconds)(4.615);     // GSM TDMA frame
    return f;
}

analysis::AnalysisResult makeGsmResult(const analysis::SignalFeatures& f)
{
    analysis::AnalysisResult r{};
    r.center_freq_hz     = f.center_freq_hz;
    r.bandwidth_hz       = f.bandwidth_hz;
    r.snr_db             = f.snr_db;
    r.digital_modulation = "GMSK";
    r.symbol_rate_sps = au::hertz(270833.0);
    r.m_ary              = 2;
    r.is_burst           = true;
    r.is_tdma            = true;
    r.burst_duty_cycle   = 0.75;
    r.classified         = true;
    return r;
}

} // anonymous namespace

class ProtocolMapperTest : public ::testing::Test {
protected:
    analysis::ProtocolMapper mapper;
};

// ── AIS test ──────────────────────────────────────────────────────────────────

TEST_F(ProtocolMapperTest, AisTopHypothesis)
{
    auto f = makeAisFeatures();
    auto r = makeAisResult(f);

    mapper.map(f, r);

    ASSERT_FALSE(r.hypotheses.empty())
        << "Should have at least one hypothesis";

    EXPECT_EQ(r.hypotheses[0].system, "AIS")
        << "Top hypothesis should be AIS for 162 MHz GMSK 9600 baud TDMA";

    EXPECT_GT(r.hypotheses[0].confidence, 0.8f)
        << "AIS confidence should be > 0.8";

    EXPECT_EQ(r.hypotheses[0].category, "Marine")
        << "AIS category should be Marine";
}

// ── FM Broadcast test ─────────────────────────────────────────────────────────

TEST_F(ProtocolMapperTest, FmBroadcastTopHypothesis)
{
    auto f = makeFmBroadcastFeatures();
    auto r = makeFmBroadcastResult(f);

    mapper.map(f, r);

    ASSERT_FALSE(r.hypotheses.empty())
        << "Should have at least one hypothesis";

    EXPECT_EQ(r.hypotheses[0].system, "FM Broadcast")
        << "Top hypothesis should be FM Broadcast for 100 MHz FM_WB 200 kHz BW";

    EXPECT_GT(r.hypotheses[0].confidence, 0.7f)
        << "FM Broadcast confidence should be > 0.7";
}

// ── GSM test ──────────────────────────────────────────────────────────────────

TEST_F(ProtocolMapperTest, GsmTopHypothesis)
{
    auto f = makeGsmFeatures();
    auto r = makeGsmResult(f);

    mapper.map(f, r);

    ASSERT_FALSE(r.hypotheses.empty())
        << "Should have at least one hypothesis";

    // GSM 900 is at 935 MHz
    bool found_gsm = false;
    for (const auto& h : r.hypotheses) {
        if (h.system.find("GSM") != std::string::npos) {
            found_gsm = true;
            EXPECT_EQ(h.category, "Cellular");
            break;
        }
    }
    EXPECT_TRUE(found_gsm)
        << "GSM should appear in top hypotheses for 935 MHz GMSK 270833 baud TDMA";
}

// ── Hypothesis count ──────────────────────────────────────────────────────────

TEST_F(ProtocolMapperTest, AtMostThreeHypotheses)
{
    auto f = makeAisFeatures();
    auto r = makeAisResult(f);
    mapper.map(f, r);
    EXPECT_LE(r.hypotheses.size(), 3u);
}

// ── Confidence ordering ───────────────────────────────────────────────────────

TEST_F(ProtocolMapperTest, HypothesesSortedDescending)
{
    auto f = makeFmBroadcastFeatures();
    auto r = makeFmBroadcastResult(f);
    mapper.map(f, r);

    for (size_t i = 1; i < r.hypotheses.size(); ++i) {
        EXPECT_GE(r.hypotheses[i-1].confidence, r.hypotheses[i].confidence)
            << "Hypotheses should be sorted by confidence descending";
    }
}

// ── Reasoning string non-empty ────────────────────────────────────────────────

TEST_F(ProtocolMapperTest, ReasoningIsNonEmpty)
{
    auto f = makeAisFeatures();
    auto r = makeAisResult(f);
    mapper.map(f, r);

    for (const auto& h : r.hypotheses) {
        EXPECT_FALSE(h.reasoning.empty())
            << "Reasoning should not be empty for " << h.system;
    }
}

// ── LoRa test ─────────────────────────────────────────────────────────────────

TEST_F(ProtocolMapperTest, LoRaHypothesis)
{
    analysis::SignalFeatures f;
    f.center_freq_hz = au::hertz(868.1e6);
    f.bandwidth_hz = au::hertz(125000.0);
    f.snr_db             = 10.0;
    f.chirp_detected     = true;
    f.chirp_rate_hz_s    = 800000.0;
    f.is_burst           = true;
    f.burst_duty_cycle   = 0.3;
    f.sample_rate_sps = au::hertz(1e6);

    analysis::AnalysisResult r{};
    r.center_freq_hz     = f.center_freq_hz;
    r.bandwidth_hz       = f.bandwidth_hz;
    r.digital_modulation = "CSS";
    r.is_burst           = true;
    r.classified         = true;

    mapper.map(f, r);

    ASSERT_FALSE(r.hypotheses.empty());

    bool found_lora = false;
    for (const auto& h : r.hypotheses) {
        if (h.system.find("LoRa") != std::string::npos) {
            found_lora = true;
            break;
        }
    }
    EXPECT_TRUE(found_lora)
        << "LoRa 868 should appear in top hypotheses for 868 MHz CSS burst";
}

// ── Wi-Fi test ────────────────────────────────────────────────────────────────

TEST_F(ProtocolMapperTest, WiFiHypothesis)
{
    analysis::SignalFeatures f;
    f.center_freq_hz = au::hertz(2437e6);   // Wi-Fi channel 6
    f.bandwidth_hz = au::hertz(20e6);
    f.snr_db             = 25.0;
    f.ofdm_detected      = true;
    f.ofdm_fft_size_est  = 64;
    f.ofdm_cp_ratio      = 0.25;
    f.is_burst           = false;
    f.burst_duty_cycle   = 1.0;
    f.sample_rate_sps = au::hertz(40e6);

    analysis::AnalysisResult r{};
    r.center_freq_hz     = f.center_freq_hz;
    r.bandwidth_hz       = f.bandwidth_hz;
    r.digital_modulation = "OFDM";
    r.is_ofdm            = true;
    r.classified         = true;

    mapper.map(f, r);

    ASSERT_FALSE(r.hypotheses.empty());

    bool found_wifi = false;
    for (const auto& h : r.hypotheses) {
        if (h.system.find("Wi-Fi") != std::string::npos) {
            found_wifi = true;
            break;
        }
    }
    EXPECT_TRUE(found_wifi)
        << "Wi-Fi 2.4 GHz should appear in top hypotheses for 2437 MHz OFDM 20 MHz";
}
