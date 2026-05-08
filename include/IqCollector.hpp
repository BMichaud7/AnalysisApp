#pragma once
#include "Config.hpp"
#include <vector>
#include <string>

namespace analysis {

class IqCollector {
public:
    IqCollector(const AmqpConfig& amqp_cfg,
                const CollectorConfig& col_cfg,
                const std::string& local_ip);
    ~IqCollector();

    // Returns CF32 IQ samples (I,Q,I,Q,...) for the requested frequency.
    // Returns empty vector on failure.
    std::vector<float> collect(double center_freq_hz,
                               double bandwidth_hz,
                               const std::string& request_id);

    double lastSampleRate() const { return last_sr_; }

private:
    AmqpConfig      amqp_cfg_;
    CollectorConfig col_cfg_;
    std::string     local_ip_;
    double          last_sr_ = 0;

    void*  connection_ = nullptr;
    void*  session_    = nullptr;
    void*  sender_     = nullptr;
    void*  receiver_   = nullptr;

    bool   connect();
    void   disconnect();
};

} // namespace analysis
