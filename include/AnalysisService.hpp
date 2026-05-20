#pragma once
#include "Config.hpp"
#include "AnalysisEngine.hpp"
#include "IqCollector.hpp"
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <vector>

#ifdef ANALYSIS_WITH_DB
#include <pqxx/pqxx>
#endif

// Forward declarations
namespace proton { class container; }

namespace analysis {

// Forward declare the AMQP handler (defined in .cpp)
class ServiceAmqpHandler;

struct Detection {
    std::string scanner_id;
    double      center_freq_hz;
    double      bandwidth_hz;
    double      power_db;
    int64_t     timestamp_ms;

    // 1 024-sample IQ snapshot from the acquiring dwell (interleaved float32 I,Q).
    // Non-empty when AcquisitionApp embeds it in the RF_DETECTION message.
    // Used by the ONNX fast path to classify without re-acquiring the SDR.
    std::vector<float> iq_snapshot;
    double             snapshot_sample_rate_sps{0.0};
};

class AnalysisService {
public:
    explicit AnalysisService(const AppConfig& cfg);
    ~AnalysisService();

    void start();
    void stop();
    bool isRunning() const { return running_.load(); }

private:
    AppConfig       cfg_;
    AnalysisEngine  engine_;
    IqCollector     collector_;
    std::atomic<bool> running_{false};

    // AMQP handler (shared between subscriptionLoop and publishResult)
    std::shared_ptr<ServiceAmqpHandler>      amqp_handler_;
    std::shared_ptr<proton::container>       amqp_container_;

    // Subscription thread
    std::thread sub_thread_;
    void subscriptionLoop();

    // Worker threads — two run slow-path IQ collections concurrently.
    static constexpr int NUM_WORKERS = 2;
    std::vector<std::thread> worker_threads_;
    std::mutex  q_mu_;
    std::condition_variable q_cv_;
    std::queue<Detection> queue_;
    void workerLoop();

    // Frequency-dedup: skip re-analysis of a recently-classified frequency.
    // Key = freq_hz rounded to 100 kHz bucket. Value = last-enqueued epoch ms.
    std::unordered_map<int64_t, int64_t> recent_analyzed_;
    // 5-minute cooldown: each unique signal is analyzed at most once per 5 min.
    static constexpr int64_t REANALYSIS_COOLDOWN_MS = 300'000;
    // Cap the backlog: 6 items / 2 workers × ~3.3s = 10s max drain.
    // With a 3s window, 2 workers can process 2 signals concurrently.
    static constexpr size_t  MAX_QUEUE_DEPTH        = 6;

    void processDetection(const Detection& d);
    void publishResult(const AnalysisResult& r);

#ifdef ANALYSIS_WITH_DB
    // Connection string written once in constructor; each worker thread opens
    // its own pqxx::connection via thread_local in persistResult().
    std::string db_conn_str_;
    void persistResult(const AnalysisResult& r);
#endif
};

} // namespace analysis
