/*
========================================================================
Project: OpenRFStack
Author:  Brendan Michaud
Year:    2026
Part of OpenRFStack (https://github.com/OpenRFStack)

Licensed under the Personal Use License.
Do not use for commercial, organizational, or military purposes.
Contact author for permission: https://github.com/OpenRFStack
========================================================================
*/
#pragma once
/**
 * @file AnalysisResult.hpp
 * @note Frequency fields use au::QuantityD<au::Hertz>;
 *       timestamp_ms uses au::QuantityD<au::Seconds>.
 * @brief Full signal characterisation result from the analysis pipeline.
 *
 * AnalysisResult captures five layers of signal description:
 * - **Layer 1** — analog modulation type and index
 * - **Layer 2** — digital carrier modulation and symbol rate
 * - **Layer 3** — channel structure (burst, TDMA, FHSS, OFDM)
 * - **Layer 4** — bitstream traits (bit rate, line code, FEC)
 * - **Layer 5** — protocol hypotheses matched against the signature database
 */
#include <au/units/hertz.hh>
#include <au/units/seconds.hh>
#include <string>
#include <vector>
#include <cstdint>

namespace analysis {

/**
 * @brief One protocol identification hypothesis (top-3 returned per signal).
 */
struct Hypothesis {
    std::string system;      ///< Protocol / system name (e.g. "GSM", "AIS", "FM Broadcast").
    std::string category;    ///< Broad category (e.g. "Cellular", "Marine", "Broadcast").
    float       confidence;  ///< Match score normalised to [0, 1].
    std::string reasoning;   ///< Human-readable explanation of why this hypothesis scored.
};

/**
 * @brief Complete characterisation of one detected signal.
 *
 * Produced by AnalysisEngine::analyze() or AnalysisEngine::analyzeSnapshot()
 * and serialised to JSON for publication on the @c rf.analysis AMQP topic.
 *
 * The @p fast_path flag indicates whether classification came from the
 * embedded IQ snapshot (~5 ms, ONNX only) or the full slow path
 * (IQ re-acquisition + feature extraction, ~80–165 ms).
 */
struct AnalysisResult {
    std::string detection_id;   ///< UUID matching the originating RF_DETECTION message.
    std::string scanner_id;     ///< Scanner that produced the detection.
    au::QuantityD<au::Hertz>   center_freq_hz{au::hertz(0.0)}; ///< Centre frequency.
    au::QuantityD<au::Hertz>   bandwidth_hz{au::hertz(0.0)};   ///< Occupied bandwidth.
    double      snr_db{0.0};    ///< Signal-to-noise ratio (dB).
    au::QuantityD<au::Seconds> timestamp_ms{au::seconds(0.0)}; ///< UTC epoch of original detection.

    // ── Layer 1: Analog modulation ──────────────────────────────────────────
    /// Analog modulation type: "AM_DSB_LC", "AM_DSB_SC", "SSB_USB", "SSB_LSB",
    /// "CW", "FM_NB", "FM_WB", "PM", "ANALOG_TV", or "" (not analog).
    std::string analog_modulation;
    double      analog_index;   ///< AM modulation index or FM deviation (Hz).

    // ── Layer 2: Digital carrier ────────────────────────────────────────────
    /// Digital modulation: "BPSK", "QPSK", "8PSK", "QAM16", "QAM64", … or "".
    std::string digital_modulation;
    au::QuantityD<au::Hertz> symbol_rate_sps{au::hertz(0.0)}; ///< Estimated symbol rate (symbols/s).
    int         m_ary{0};        ///< Modulation order (e.g. 2 for BPSK, 4 for QPSK).
    bool        is_ofdm;         ///< True if OFDM subcarrier structure detected.
    bool        is_spread;       ///< True if DSSS or FHSS spreading detected.

    // ── Layer 3: Channel structure ──────────────────────────────────────────
    bool        is_burst;                    ///< True if burst (non-continuous) transmission.
    double      burst_duty_cycle;            ///< Fraction of time the carrier is on (0–1).
    bool        is_tdma;                     ///< True if time-division multiple access detected.
    bool        is_fhss{false};              ///< True if frequency-hopping spread spectrum.
    bool        is_dsss{false};              ///< True if direct-sequence spread spectrum.
    au::QuantityD<au::Hertz> ofdm_subcarrier_spacing_hz{au::hertz(0.0)}; ///< OFDM subcarrier spacing (0 if not OFDM).

    // ── Layer 4: Bitstream traits ───────────────────────────────────────────
    double      bit_rate_bps;    ///< Estimated bit rate (bits/s).
    std::string line_code;       ///< Line coding: "NRZ", "NRZI", "Manchester", or "".
    bool        fec_detected;    ///< True if forward error correction markers found.
    bool        has_sync_pattern; ///< True if a recognisable preamble/sync word detected.

    // ── Layer 5: Protocol hypotheses ───────────────────────────────────────
    /// Top protocol matches (up to 3), sorted by confidence descending.
    std::vector<Hypothesis> hypotheses;

    bool        classified;      ///< False if SNR too low or features inconclusive.
    std::string reject_reason;   ///< Non-empty when classified == false.

    // ── Path metadata ───────────────────────────────────────────────────────
    float rule_confidence = 0.f;  ///< Certainty of the rule-based result (0–1; 1 = structural detection).
    bool  onnx_used       = false; ///< True if ONNX classifier overrode or augmented the result.
    float onnx_confidence = 0.f;  ///< Softmax probability of the top ONNX class (if used).
    /// True = classified from the embedded IQ snapshot (~5 ms, no SDR re-acquisition).
    /// False = full IQ collection + feature extraction (~80–165 ms).
    bool  fast_path       = false;
};

} // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
