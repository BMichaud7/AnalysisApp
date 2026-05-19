#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace analysis {

struct Hypothesis {
    std::string system;      // e.g. "GSM", "AIS", "FM Broadcast"
    std::string category;    // e.g. "Cellular", "Marine", "Broadcast"
    float       confidence;  // 0.0–1.0
    std::string reasoning;   // human-readable explanation
};

struct AnalysisResult {
    std::string detection_id;   // UUID
    std::string scanner_id;
    double      center_freq_hz;
    double      bandwidth_hz;
    double      snr_db;
    int64_t     timestamp_ms;

    // Layer 1: Analog modulation
    std::string analog_modulation;  // "AM_DSB_LC","AM_DSB_SC","SSB_USB","SSB_LSB","CW","FM_NB","FM_WB","PM","ANALOG_TV" or ""
    double      analog_index;       // AM modulation index or FM deviation Hz

    // Layer 2: Digital carrier
    std::string digital_modulation; // "BPSK","QPSK","8PSK","QAM16",... or ""
    double      symbol_rate_sps;
    int         m_ary;
    bool        is_ofdm;
    bool        is_spread;          // DSSS or FHSS

    // Layer 3: Channel structure
    bool        is_burst;
    double      burst_duty_cycle;
    bool        is_tdma;
    bool        is_fhss;
    bool        is_dsss;
    double      ofdm_subcarrier_spacing_hz;

    // Layer 4: Bitstream traits
    double      bit_rate_bps;
    std::string line_code;       // "NRZ","NRZI","Manchester",""
    bool        fec_detected;
    bool        has_sync_pattern;

    // Layer 5: Protocol hypotheses (top 3, descending confidence)
    std::vector<Hypothesis> hypotheses;

    bool  classified;  // false if SNR too low or features inconclusive
    std::string reject_reason;

    // Path metadata
    float rule_confidence = 0.f;  // 0–1: certainty of rule-based result
                                  // (1.0 = structural detection, 0.0 = UNKNOWN)
    bool  onnx_used = false;      // true if ONNX classifier overrode or augmented
    float onnx_confidence = 0.f;  // softmax probability from ONNX (if used)
    bool  fast_path = false;      // true = classified from embedded IQ snapshot (~5ms)
                                  // false = full IQ collection + feature extraction (~165ms)
};

} // namespace analysis
