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
 * @file IqCollector.hpp
 * @brief IQ sample collector backed by a persistent AMQP task channel.
 *
 * IqCollector submits NARROWBAND task requests to SdrResourceManager, receives
 * CF32 IQ samples over UDP, and returns them to the caller.
 *
 * ## Persistent connection
 * IqTaskChannel (defined in IqCollector.cpp) connects to the Artemis broker
 * once at construction and keeps the connection alive.  This avoids the ~30 s
 * per-connection overhead on new Artemis connections, reducing collect()
 * round-trip from ~30 s to ~80 ms.
 *
 * ## Concurrency
 * IqCollector is shared between AnalysisService worker threads but the SDR
 * is an exclusive resource — only one NARROWBAND task can run at a time.
 * collect_mu_ serialises concurrent calls.
 */
#include "Config.hpp"
#include <au/units/hertz.hh>
#include <memory>
#include <mutex>
#include <vector>
#include <string>

namespace analysis {

class IqTaskChannel;  ///< Persistent AMQP request/response channel (defined in .cpp).

/**
 * @brief Collects IQ samples for a given frequency via SdrResourceManager.
 *
 * Non-copyable.  collect() blocks until samples are received or timeout.
 * Thread-safe: concurrent calls from multiple workers are serialised internally.
 */
class IqCollector {
public:
    /**
     * @brief Construct and connect the persistent AMQP channel.
     * @param amqp_cfg   AMQP connection and queue configuration.
     * @param col_cfg    Collection parameters (sample count, timeout, sample rate).
     * @param local_ip   IP address on which to bind the UDP receive socket.
     * @param task_rank  Task rank submitted to the controller (lower = lower priority).
     * @throws std::runtime_error if AMQP channel fails to connect within 60 s.
     */
    IqCollector(const AmqpConfig& amqp_cfg,
                const CollectorConfig& col_cfg,
                const std::string& local_ip,
                int task_rank = 1);
    ~IqCollector();

    /**
     * @brief Collect IQ samples for the requested frequency.
     *
     * Submits a NARROWBAND task request, binds a UDP socket, receives
     * CollectorConfig::collect_samples samples, sends TASK_STOP, and returns.
     * Concurrent calls from multiple threads are serialised internally.
     *
     * @param center_freq_hz  Centre frequency to tune to (Hz).
     * @param bandwidth_hz    Requested bandwidth (Hz).
     * @param request_id      UUID for the AMQP task request.
     * @return Interleaved float32 IQ samples (I,Q,I,Q,…), or empty on failure.
     */
    std::vector<float> collect(au::QuantityD<au::Hertz> center_freq_hz,
                               au::QuantityD<au::Hertz> bandwidth_hz,
                               const std::string& request_id);

    /// @brief Sample rate of the last successful collect().
    au::QuantityD<au::Hertz> lastSampleRate() const { return last_sr_; }

    bool   connect()    { return true; }  ///< No-op (connection managed internally).
    void   disconnect() {}                ///< No-op (connection managed internally).

private:
    AmqpConfig               amqp_cfg_;
    CollectorConfig          col_cfg_;
    std::string              local_ip_;
    int                      task_rank_ = 1;
    au::QuantityD<au::Hertz> last_sr_{au::hertz(0.0)};

    /// Persistent AMQP channel — connects once at construction, reused forever.
    std::unique_ptr<IqTaskChannel> ch_;
    /// Serialises concurrent collect() calls (SDR is an exclusive resource anyway).
    std::mutex collect_mu_;
};

} // namespace analysis
