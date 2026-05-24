#pragma once
/**
 * @file AnalysisService.hpp
 * @brief Top-level service: AMQP subscriber + dual worker threads + DB persistence.
 *
 * AnalysisService ties together the AMQP subscription loop and the analysis
 * worker threads:
 *
 * - **subscriptionLoop** — AMQP thread subscribes to rf.detections, decodes
 *   RF_DETECTION messages (including base64 IQ snapshots), and enqueues
 *   detections for the workers.
 *
 * - **workerLoop (×NUM_WORKERS)** — dequeues detections, runs processDetection()
 *   (fast path if snapshot available, slow path otherwise), publishes results
 *   to rf.analysis, and persists to PostgreSQL.
 *
 * - **Frequency dedup** — a cooldown map prevents re-analysing the same
 *   frequency more than once per REANALYSIS_COOLDOWN_MS (5 minutes).
 *
 * - **Queue depth limit** — MAX_QUEUE_DEPTH caps the backlog so IQ collection
 *   windows are not missed while older items drain.
 *
 * @note The single SDR device is an exclusive resource.  Two workers can
 *       collect IQ concurrently if the fast path (ONNX on snapshot) is
 *       available; otherwise IqCollector serialises them.
 */
#include "Config.hpp"
#include "AnalysisEngine.hpp"
#include "AnalysisResult.hpp"
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

namespace proton { class container; }

namespace analysis {

class ServiceAmqpHandler;  ///< Internal AMQP handler (defined in .cpp).

/**
 * @brief Inbound detection from the AMQP rf.detections topic.
 *
 * Populated by the subscription loop from the RF_DETECTION JSON message.
 * @p iq_snapshot is non-empty when AcquisitionApp embedded a snapshot in the
 * message (schema 1.2 @c iq_snapshot_b64 field) — enables the ONNX fast path.
 */
struct Detection {
    std::string scanner_id;
    double      center_freq_hz;
    double      bandwidth_hz;
    double      power_db;
    int64_t     timestamp_ms;

    /// 1 024-sample CF32 IQ snapshot decoded from @c iq_snapshot_b64 (if present).
    /// Empty when not provided; used by AnalysisEngine::analyzeSnapshot().
    std::vector<float> iq_snapshot;
    double             snapshot_sample_rate_sps{0.0};
};

/**
 * @brief Orchestrates the full analysis pipeline as a long-running service.
 *
 * Call start() once; call stop() to drain the queue and shut down cleanly.
 */
class AnalysisService {
public:
    /**
     * @brief Construct the service.
     * @param cfg Application configuration (AMQP, collector, engine, DB).
     */
    explicit AnalysisService(const AppConfig& cfg);
    ~AnalysisService();

    /// @brief Connect to AMQP and launch the subscription + worker threads.
    void start();
    /// @brief Drain the work queue, stop workers, and disconnect.
    void stop();
    /// @brief True after start() returns and before stop() is called.
    bool isRunning() const { return running_.load(); }

private:
    AppConfig       cfg_;
    AnalysisEngine  engine_;
    IqCollector     collector_;
    std::atomic<bool> running_{false};

    std::shared_ptr<ServiceAmqpHandler>      amqp_handler_;
    std::shared_ptr<proton::container>       amqp_container_;

    std::thread sub_thread_;
    void subscriptionLoop();

    /// Number of concurrent worker threads.  Two workers allow parallel fast-path
    /// classification while IqCollector serialises slow-path SDR access.
    static constexpr int NUM_WORKERS = 2;
    std::vector<std::thread> worker_threads_;
    std::mutex  q_mu_;
    std::condition_variable q_cv_;
    std::queue<Detection> queue_;
    void workerLoop();

    /// Per-frequency (100 kHz bucket) dedup: skip re-analysis within the cooldown window.
    std::unordered_map<int64_t, int64_t> recent_analyzed_;
    /// Cooldown: each unique signal is analysed at most once per 5 minutes.
    static constexpr int64_t REANALYSIS_COOLDOWN_MS = 300'000;
    /// Max queue depth: 6 items / 2 workers ≈ 10 s max drain within the 3 s analysis window.
    static constexpr size_t  MAX_QUEUE_DEPTH        = 6;

    void processDetection(const Detection& d);
    void publishResult(const AnalysisResult& r);

    /**
     * @brief Called from the AMQP thread when a REQUEST_DEMOD command arrives.
     *
     * Looks up the most recent AnalysisResult for the requested frequency and
     * forwards it as a DEMOD_REQUEST to the rf.demod.request queue.
     *
     * @param freq_hz    Requested centre frequency in Hz.
     * @param request_id Correlation ID supplied by the caller (may be empty).
     */
    void onDemodCommand(double freq_hz, const std::string& request_id);

    /// @brief Forward a cached result to DemodApp as a DEMOD_REQUEST message.
    void publishDemodRequest(const AnalysisResult& r, const std::string& request_id);

    /// @brief Cache of most-recent AnalysisResult per 100 kHz frequency bucket.
    /// Protected by q_mu_.
    std::unordered_map<int64_t, AnalysisResult> recent_results_;

#ifdef ANALYSIS_WITH_DB
    /// Connection string written once in constructor.
    /// Each worker opens its own pqxx::connection via thread_local in persistResult().
    std::string db_conn_str_;
    void persistResult(const AnalysisResult& r);
#endif
};

} // namespace analysis
