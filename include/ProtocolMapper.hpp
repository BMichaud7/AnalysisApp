#pragma once
#include "SignalFeatures.hpp"
#include "AnalysisResult.hpp"
#include <vector>

namespace analysis {

class ProtocolMapper {
public:
    ProtocolMapper();  // loads built-in signature database

    // Fills result.hypotheses with top-3 protocol matches.
    void map(const SignalFeatures& f, AnalysisResult& r) const;

private:
    struct Signature {
        std::string name;
        std::string category;
        // Frequency bands (MHz) — empty = any
        std::vector<std::pair<double,double>> freq_bands_mhz;
        // Expected modulations — empty = any
        std::vector<std::string> modulations;
        double bw_min_hz, bw_max_hz;   // 0 = don't care
        double sr_min_sps, sr_max_sps; // 0 = don't care
        bool   burst_required = false;
        bool   burst_forbidden = false;
        bool   ofdm_required  = false;
        bool   fhss_required  = false;
        double score(const SignalFeatures& f, const AnalysisResult& r) const;
    };

    std::vector<Signature> db_;
    void buildDatabase();
};

} // namespace analysis
