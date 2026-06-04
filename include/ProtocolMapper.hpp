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
 * @file ProtocolMapper.hpp
 * @brief Matches signal features against a built-in protocol signature database.
 *
 * ProtocolMapper scores each entry in a hard-coded signature database against
 * the observed SignalFeatures and populates AnalysisResult::hypotheses with
 * the top-3 matches (Layer 5).
 *
 * Each Signature encodes typical frequency bands, modulation types, bandwidth
 * range, symbol rate range, burst/OFDM/FHSS requirements, and a scoring
 * function.  The signature database covers common signals in the 70 MHz–6 GHz
 * range (cellular, broadcast, navigation, maritime, ISM, military, etc.).
 */
#include "SignalFeatures.hpp"
#include "AnalysisResult.hpp"
#include <vector>

namespace analysis {

/// @brief Scores signal features against the built-in protocol signature database.
class ProtocolMapper {
public:
    /// @brief Construct and load the built-in signature database.
    ProtocolMapper();

    /**
     * @brief Populate result.hypotheses with the top-3 protocol matches.
     *
     * Each signature in the database is scored against @p f and @p r
     * (which contains the modulation classification from Layers 1–4).
     * The top-3 non-zero-confidence hypotheses are written to result.hypotheses,
     * sorted by confidence descending.
     *
     * @param f Signal features from FeatureExtractor::extract().
     * @param r Analysis result (Layers 1–4 must be populated; Layer 5 is written).
     */
    void map(const SignalFeatures& f, AnalysisResult& r) const;

private:
    /// @brief One protocol entry in the signature database.
    struct Signature {
        std::string name;      ///< Protocol / system name (e.g. "GSM 900", "ADS-B").
        std::string category;  ///< Broad category (e.g. "Cellular", "Aviation").

        /// Frequency bands (MHz) this signal is expected in.  Empty = any band.
        std::vector<std::pair<double,double>> freq_bands_mhz;
        /// Accepted modulation strings (from AnalysisResult).  Empty = any.
        std::vector<std::string> modulations;

        double bw_min_hz   = 0; ///< Minimum occupied bandwidth (Hz). 0 = don't care.
        double bw_max_hz   = 0; ///< Maximum occupied bandwidth (Hz). 0 = don't care.
        double sr_min_sps  = 0; ///< Minimum symbol rate (symbols/s). 0 = don't care.
        double sr_max_sps  = 0; ///< Maximum symbol rate (symbols/s). 0 = don't care.
        bool   burst_required  = false; ///< True if signal must be bursty.
        bool   burst_forbidden = false; ///< True if signal must be continuous.
        bool   ofdm_required   = false; ///< True if OFDM structure is required.
        bool   fhss_required   = false; ///< True if FHSS is required.

        /**
         * @brief Score this signature against observed features.
         * @param f  Feature vector.
         * @param r  Classification result (Layers 1–4).
         * @return   Confidence in [0, 1]; 0 = definite mismatch, 1 = perfect match.
         */
        double score(const SignalFeatures& f, const AnalysisResult& r) const;
    };

    std::vector<Signature> db_; ///< Built-in signature database.
    void buildDatabase();
};

} // namespace analysis
