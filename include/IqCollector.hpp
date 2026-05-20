#pragma once
#include "Config.hpp"
#include <memory>
#include <mutex>
#include <vector>
#include <string>

namespace analysis {

class IqTaskChannel;  // persistent AMQP request/response channel — defined in .cpp

class IqCollector {
public:
    IqCollector(const AmqpConfig& amqp_cfg,
                const CollectorConfig& col_cfg,
                const std::string& local_ip,
                int task_rank = 1);
    ~IqCollector();

    // Returns CF32 IQ samples (I,Q,I,Q,...) for the requested frequency.
    // Returns empty vector on failure. Thread-safe: serialises concurrent calls.
    std::vector<float> collect(double center_freq_hz,
                               double bandwidth_hz,
                               const std::string& request_id);

    double lastSampleRate() const { return last_sr_; }

    bool   connect()    { return true; }
    void   disconnect() {}

private:
    AmqpConfig      amqp_cfg_;
    CollectorConfig col_cfg_;
    std::string     local_ip_;
    int             task_rank_ = 1;
    double          last_sr_   = 0;

    // Persistent AMQP channel — connects once, reused for all task submissions.
    std::unique_ptr<IqTaskChannel> ch_;
    // Serialise concurrent collect() calls (SDR is an exclusive resource anyway).
    std::mutex collect_mu_;
};

} // namespace analysis
