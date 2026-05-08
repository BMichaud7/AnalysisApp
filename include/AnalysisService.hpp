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

    // Worker thread
    std::thread worker_thread_;
    std::mutex  q_mu_;
    std::condition_variable q_cv_;
    std::queue<Detection> queue_;
    void workerLoop();

    void processDetection(const Detection& d);
    void publishResult(const AnalysisResult& r);
};

} // namespace analysis
