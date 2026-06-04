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
#include "ProtocolMapper.hpp"
#include <au/units/hertz.hh>
#include <au/prefix.hh>
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace analysis {

// ── Signature::score ──────────────────────────────────────────────────────────

double ProtocolMapper::Signature::score(const SignalFeatures& f,
                                         const AnalysisResult& r) const
{
    double freq_match = 1.0;
    const double cf_mhz = f.center_freq_hz.in(au::mega(au::hertz));

    if (!freq_bands_mhz.empty()) {
        bool in_band = false;
        for (const auto& [lo, hi] : freq_bands_mhz)
            if (cf_mhz >= lo && cf_mhz <= hi) { in_band = true; break; }
        freq_match = in_band ? 1.0 : 0.0;
    }

    // Modulation match
    double mod_match = 0.5;  // neutral if empty
    if (!modulations.empty()) {
        std::string mod = r.digital_modulation.empty()
                          ? r.analog_modulation
                          : r.digital_modulation;
        mod_match = 0.0;
        for (const auto& m : modulations) {
            if (m == mod) { mod_match = 1.0; break; }
            // Partial: GMSK ↔ FSK, QPSK ↔ PSK, etc.
            if ((m == "GMSK" && mod == "FSK") || (m == "FSK" && mod == "GMSK"))
                mod_match = std::max(mod_match, 0.6);
            if ((m == "QPSK" && mod == "PSK") || (m == "PSK" && mod == "QPSK"))
                mod_match = std::max(mod_match, 0.6);
            if ((m == "OFDM" && r.is_ofdm) || (r.is_ofdm && m == "OFDM"))
                mod_match = std::max(mod_match, 1.0);
            if ((m == "CSS"  && f.chirp_detected))
                mod_match = std::max(mod_match, 1.0);
            if ((m == "FHSS" && f.fhss_detected))
                mod_match = std::max(mod_match, 1.0);
            if ((m == "DSSS" && f.dsss_detected))
                mod_match = std::max(mod_match, 1.0);
        }
    }

    // Bandwidth match: Gaussian-like score
    double bw_match = 1.0;
    if (bw_min_hz > 0 || bw_max_hz > 0) {
        const double bw_hz = f.bandwidth_hz.in(au::hertz);
        if (bw_hz < bw_min_hz * 0.5 || bw_hz > bw_max_hz * 2.0) {
            bw_match = 0.0;
        } else {
            double bw_mid = (bw_min_hz + bw_max_hz) / 2.0;
            double bw_range = bw_max_hz - bw_min_hz;
            double sigma = bw_range > 0 ? bw_range : bw_mid * 0.3;
            double d = bw_hz - bw_mid;
            bw_match = std::exp(-0.5 * (d / sigma) * (d / sigma));
        }
    }

    // Symbol rate match
    double sr_match = 1.0;
    if (sr_min_sps > 0 || sr_max_sps > 0) {
        const double sr_hz = r.symbol_rate_sps.in(au::hertz);
        if (sr_hz <= 0) {
            sr_match = 0.3;  // unknown → slight penalty
        } else if (sr_hz < sr_min_sps * 0.5 || sr_hz > sr_max_sps * 2.0) {
            sr_match = 0.0;
        } else {
            double sr_mid   = (sr_min_sps + sr_max_sps) / 2.0;
            double sr_range = sr_max_sps - sr_min_sps;
            double sigma    = sr_range > 0 ? sr_range : sr_mid * 0.3;
            double d        = sr_hz - sr_mid;
            sr_match = std::exp(-0.5 * (d / sigma) * (d / sigma));
        }
    }

    // Structural match
    double struct_match = 1.0;
    if (burst_required  && !f.is_burst) struct_match *= 0.3;
    if (burst_forbidden && f.is_burst)  struct_match *= 0.3;
    if (ofdm_required   && !f.ofdm_detected) struct_match *= 0.2;
    if (fhss_required   && !f.fhss_detected) struct_match *= 0.2;

    double total = freq_match * 0.40
                 + mod_match  * 0.30
                 + bw_match   * 0.15
                 + sr_match   * 0.15;

    total *= struct_match;

    return total;
}

// ── Database builder ──────────────────────────────────────────────────────────

void ProtocolMapper::buildDatabase()
{
    // Helper lambda
    auto add = [&](Signature s) { db_.push_back(std::move(s)); };

    // ── Broadcast ─────────────────────────────────────────────────────────

    add({ "AM Broadcast", "Broadcast",
          {{0.52, 1.71}},
          {"AM_DSB_LC"},
          9000, 10000, 0, 0, false, false, false, false });

    add({ "FM Broadcast", "Broadcast",
          {{87.5, 108.0}},
          {"FM_WB"},
          100000, 250000, 0, 0, false, true, false, false });

    add({ "HD Radio", "Broadcast",
          {{87.5, 108.0}},
          {"OFDM"},
          300000, 500000, 0, 0, false, false, true, false });

    add({ "DAB", "Broadcast",
          {{174.0, 240.0}, {1452.0, 1492.0}},
          {"OFDM"},
          1000000, 2000000, 0, 0, false, false, true, false });

    add({ "DRM", "Broadcast",
          {{3.0, 30.0}},
          {"OFDM"},
          5000, 15000, 0, 0, false, false, true, false });

    // ── Aviation ──────────────────────────────────────────────────────────

    add({ "VHF Airband", "Aviation",
          {{118.0, 137.0}},
          {"AM_DSB_LC"},
          6000, 30000, 0, 0, false, false, false, false });

    add({ "ADS-B", "Aviation",
          {{1089.0, 1091.0}},
          {},
          2000000, 4000000, 0, 0, true, false, false, false });

    add({ "ACARS", "Aviation",
          {{129.0, 130.0}, {136.8, 137.0}},
          {"AM_DSB_LC"},
          1500, 3500, 0, 0, true, false, false, false });

    add({ "VOR", "Aviation",
          {{108.0, 118.0}},
          {"AM_DSB_LC", "CW"},
          8000, 50000, 0, 0, false, false, false, false });

    // ── Marine ────────────────────────────────────────────────────────────

    add({ "AIS", "Marine",
          {{161.9, 162.1}},
          {"GMSK", "FSK"},
          20000, 30000, 9600, 9600, true, false, false, false });

    add({ "Marine VHF", "Marine",
          {{156.0, 174.0}},
          {"FM_NB"},
          10000, 30000, 0, 0, false, false, false, false });

    add({ "DSC", "Marine",
          {{156.5, 156.6}},
          {"FSK"},
          800, 1600, 1200, 1200, true, false, false, false });

    // ── Amateur ───────────────────────────────────────────────────────────

    add({ "APRS", "Amateur",
          {{144.3, 144.5}, {144.7, 144.9}},
          {"FSK", "GMSK"},
          10000, 25000, 1200, 1200, true, false, false, false });

    add({ "FT8", "Amateur",
          {{3.57, 3.58}, {7.07, 7.08}, {14.07, 14.08}},
          {"FSK"},
          30, 80, 6.25, 6.25, false, false, false, false });

    add({ "WSPR", "Amateur",
          {{14.09, 14.10}, {10.13, 10.15}},
          {"FSK"},
          1, 12, 0, 0, false, false, false, false });

    add({ "PSK31", "Amateur",
          {{14.06, 14.08}},
          {"BPSK"},
          20, 50, 31.25, 31.25, false, false, false, false });

    add({ "RTTY", "Amateur",
          {{3.0, 30.0}},
          {"FSK"},
          200, 500, 45, 110, false, false, false, false });

    add({ "D-STAR", "Amateur",
          {{144.0, 148.0}, {430.0, 440.0}},
          {"GMSK"},
          4000, 8000, 4800, 4800, false, false, false, false });

    add({ "DMR Amateur", "Amateur",
          {{144.0, 148.0}, {430.0, 440.0}},
          {"FSK", "QAM16"},
          10000, 15000, 4800, 4800, true, false, false, false });

    add({ "P25", "Amateur",
          {{136.0, 174.0}, {380.0, 512.0}, {763.0, 870.0}},
          {"QPSK", "QAM16"},
          6000, 13000, 4800, 9600, false, false, false, false });

    // ── Public Safety ─────────────────────────────────────────────────────

    add({ "TETRA", "PublicSafety",
          {{380.0, 400.0}, {410.0, 430.0}, {450.0, 470.0}},
          {"QPSK"},
          20000, 30000, 36000, 36000, true, false, false, false });

    add({ "DMR Tier II", "PublicSafety",
          {{136.0, 174.0}, {403.0, 527.0}, {762.0, 870.0}},
          {"FSK", "GMSK"},
          10000, 15000, 4800, 4800, true, false, false, false });

    add({ "DMR Tier III", "PublicSafety",
          {{136.0, 174.0}, {403.0, 527.0}, {762.0, 870.0}},
          {"FSK"},
          10000, 15000, 4800, 4800, true, false, false, false });

    add({ "MPT1327", "PublicSafety",
          {{138.0, 174.0}, {450.0, 470.0}},
          {"FSK"},
          8000, 25000, 1200, 4800, true, false, false, false });

    // ── Cellular ──────────────────────────────────────────────────────────

    add({ "GSM 850", "Cellular",
          {{824.0, 894.0}},
          {"GMSK"},
          150000, 250000, 270833, 270833, true, false, false, false });

    add({ "GSM 900", "Cellular",
          {{880.0, 960.0}},
          {"GMSK"},
          150000, 250000, 270833, 270833, true, false, false, false });

    add({ "GSM 1800", "Cellular",
          {{1710.0, 1880.0}},
          {"GMSK"},
          150000, 250000, 270833, 270833, true, false, false, false });

    add({ "GSM 1900", "Cellular",
          {{1850.0, 1990.0}},
          {"GMSK"},
          150000, 250000, 270833, 270833, true, false, false, false });

    add({ "UMTS/WCDMA", "Cellular",
          {{850.0, 900.0}, {1700.0, 2200.0}},
          {"QPSK", "DSSS"},
          4000000, 6000000, 3840000, 3840000, false, false, false, false });

    add({ "LTE 700", "Cellular",
          {{698.0, 798.0}},
          {"OFDM"},
          1400000, 20000000, 0, 0, false, false, true, false });

    add({ "LTE 800", "Cellular",
          {{790.0, 862.0}},
          {"OFDM"},
          1400000, 20000000, 0, 0, false, false, true, false });

    add({ "LTE 1800", "Cellular",
          {{1710.0, 1880.0}},
          {"OFDM"},
          1400000, 20000000, 0, 0, false, false, true, false });

    add({ "LTE 2100", "Cellular",
          {{1900.0, 2200.0}},
          {"OFDM"},
          1400000, 20000000, 0, 0, false, false, true, false });

    add({ "LTE 2600", "Cellular",
          {{2500.0, 2690.0}},
          {"OFDM"},
          1400000, 20000000, 0, 0, false, false, true, false });

    add({ "5G NR Sub-6", "Cellular",
          {{600.0, 700.0}, {2500.0, 2700.0}, {3300.0, 4200.0}},
          {"OFDM"},
          5000000, 100000000, 0, 0, false, false, true, false });

    add({ "5G NR mmWave", "Cellular",
          {{24250.0, 52600.0}},
          {"OFDM"},
          50000000, 400000000, 0, 0, false, false, true, false });

    // ── ISM / IoT ─────────────────────────────────────────────────────────

    add({ "Wi-Fi 2.4 GHz", "ISM",
          {{2400.0, 2484.0}},
          {"OFDM"},
          15000000, 45000000, 0, 0, false, false, true, false });

    add({ "Wi-Fi 5 GHz", "ISM",
          {{5150.0, 5850.0}},
          {"OFDM"},
          15000000, 85000000, 0, 0, false, false, true, false });

    add({ "Bluetooth Classic", "ISM",
          {{2400.0, 2484.0}},
          {"FSK", "GMSK"},
          800000, 1200000, 0, 0, true, false, false, true });

    add({ "Bluetooth LE", "ISM",
          {{2400.0, 2484.0}},
          {"GMSK"},
          1500000, 2500000, 0, 0, true, false, false, true });

    add({ "Zigbee", "ISM",
          {{2400.0, 2484.0}},
          {"DSSS"},
          1500000, 2500000, 250000, 250000, false, false, false, false });

    add({ "LoRa 433", "ISM",
          {{433.0, 435.0}},
          {"CSS"},
          100000, 600000, 0, 0, true, false, false, false });

    add({ "LoRa 868", "ISM",
          {{863.0, 870.0}},
          {"CSS"},
          100000, 600000, 0, 0, true, false, false, false });

    add({ "LoRa 915", "ISM",
          {{902.0, 928.0}},
          {"CSS"},
          100000, 600000, 0, 0, true, false, false, false });

    add({ "Z-Wave EU", "ISM",
          {{868.3, 868.6}},
          {"FSK"},
          150000, 250000, 0, 0, true, false, false, false });

    add({ "Z-Wave US", "ISM",
          {{908.3, 908.6}},
          {"FSK"},
          150000, 250000, 0, 0, true, false, false, false });

    add({ "TPMS 315", "Automotive",
          {{314.5, 315.5}},
          {"FSK"},
          100000, 200000, 19200, 38400, true, false, false, false });

    add({ "TPMS 433", "Automotive",
          {{433.0, 434.0}},
          {"FSK"},
          100000, 200000, 19200, 38400, true, false, false, false });

    add({ "Key Fob 315", "Automotive",
          {{314.5, 315.5}},
          {"FSK", "CW"},
          100000, 400000, 0, 0, true, false, false, false });

    add({ "Key Fob 433", "Automotive",
          {{433.0, 434.0}},
          {"FSK", "CW"},
          100000, 400000, 0, 0, true, false, false, false });

    add({ "Key Fob 868", "Automotive",
          {{868.0, 869.0}},
          {"FSK"},
          100000, 400000, 0, 0, true, false, false, false });

    // ── Satellite ─────────────────────────────────────────────────────────

    add({ "GPS L1", "Satellite",
          {{1575.0, 1576.0}},
          {"BPSK", "DSSS"},
          1500000, 2500000, 0, 0, false, false, false, false });

    add({ "NOAA APT", "Satellite",
          {{137.0, 138.0}},
          {"FM_NB", "FM_WB"},
          25000, 45000, 2400, 2400, false, false, false, false });

    add({ "Meteor LRPT", "Satellite",
          {{137.0, 138.0}},
          {"QPSK"},
          100000, 150000, 72000, 72000, false, false, false, false });

    add({ "Iridium", "Satellite",
          {{1616.0, 1626.5}},
          {"QPSK"},
          25000, 40000, 0, 0, true, false, false, false });

    add({ "Inmarsat AERO", "Satellite",
          {{1525.0, 1559.0}},
          {"QPSK", "BPSK"},
          8000, 15000, 0, 0, false, false, false, false });

    // ── Radar / Misc ──────────────────────────────────────────────────────

    add({ "FMCW Radar 24G", "Radar",
          {{24000.0, 24250.0}},
          {"CSS"},
          0, 0, 0, 0, false, false, false, false });

    add({ "FMCW Radar 77G", "Radar",
          {{76000.0, 77000.0}},
          {"CSS"},
          0, 0, 0, 0, false, false, false, false });

    add({ "Weather Radar", "Radar",
          {{2700.0, 2900.0}},
          {},
          0, 0, 0, 0, true, false, false, false });

    spdlog::debug("ProtocolMapper: built database with {} signatures", db_.size());
}

// ── Constructor ───────────────────────────────────────────────────────────────

ProtocolMapper::ProtocolMapper()
{
    buildDatabase();
}

// ── map ───────────────────────────────────────────────────────────────────────

void ProtocolMapper::map(const SignalFeatures& f, AnalysisResult& r) const
{
    // Score every signature
    std::vector<std::pair<double, const Signature*>> scored;
    scored.reserve(db_.size());
    for (const auto& sig : db_) {
        double s = sig.score(f, r);
        if (s > 0.05)
            scored.emplace_back(s, &sig);
    }

    // Sort descending
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    // Take top 3
    r.hypotheses.clear();
    int limit = std::min(3, static_cast<int>(scored.size()));
    for (int i = 0; i < limit; ++i) {
        const auto& [sc, sig] = scored[i];

        // Build reasoning string
        std::string reason;
        if (!r.digital_modulation.empty())
            reason += r.digital_modulation;
        else if (!r.analog_modulation.empty())
            reason += r.analog_modulation;
        if (r.symbol_rate_sps.in(au::hertz) > 0)
            reason += " " + std::to_string((int)r.symbol_rate_sps.in(au::hertz)) + " baud";
        if (f.center_freq_hz.in(au::hertz) > 0)
            reason += " at " + std::to_string((int)f.center_freq_hz.in(au::mega(au::hertz))) + " MHz";
        if (f.is_burst)
            reason += ", TDMA burst";
        reason += " (score=" + std::to_string((int)(sc * 100)) + "%)";

        Hypothesis h;
        h.system     = sig->name;
        h.category   = sig->category;
        h.confidence = static_cast<float>(sc);
        h.reasoning  = reason;
        r.hypotheses.push_back(std::move(h));
    }

    if (!r.hypotheses.empty()) {
        spdlog::debug("ProtocolMapper: top hypothesis '{}' conf={:.2f}",
                      r.hypotheses[0].system, r.hypotheses[0].confidence);
    }
}

} // namespace analysis
