#include "AnalysisService.hpp"

#include <sdr/Base64.hpp>
#include <proton/container.hpp>
#include <proton/message.hpp>
#include <proton/messaging_handler.hpp>
#include <proton/connection.hpp>
#include <proton/connection_options.hpp>
#include <proton/reconnect_options.hpp>
#include <proton/sender.hpp>
#include <proton/receiver.hpp>
#include <proton/delivery.hpp>
#include <proton/work_queue.hpp>
#include <proton/transport.hpp>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <sstream>

namespace analysis {

using json = nlohmann::json;

// ── UUID helper (RFC 4122 v4) ─────────────────────────────────────────────────

static std::string generateUuid()
{
    // Use a thread_local Mersenne Twister seeded from random_device
    thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;

    uint64_t hi = dist(rng);
    uint64_t lo = dist(rng);

    // Set version 4 bits
    hi = (hi & 0xFFFFFFFFFFFF0FFFull) | 0x0000000000004000ull;
    // Set variant bits
    lo = (lo & 0x3FFFFFFFFFFFFFFFull) | 0x8000000000000000ull;

    std::ostringstream ss;
    ss << std::hex << std::setfill('0')
       << std::setw(8) << (hi >> 32)
       << '-'
       << std::setw(4) << ((hi >> 16) & 0xFFFF)
       << '-'
       << std::setw(4) << (hi & 0xFFFF)
       << '-'
       << std::setw(4) << (lo >> 48)
       << '-'
       << std::setw(12) << (lo & 0x0000FFFFFFFFFFFFull);
    return ss.str();
}

// ── Combined AMQP handler (sub + pub on separate sessions) ────────────────────

class ServiceAmqpHandler : public proton::messaging_handler {
public:
    ServiceAmqpHandler(const AmqpConfig& cfg,
                       std::function<void(const Detection&)> on_detection)
        : cfg_(cfg)
        , on_detection_(std::move(on_detection))
    {}

    void on_container_start(proton::container& c) override {
        proton::connection_options opts;
        if (!cfg_.username.empty()) {
            opts.sasl_allowed_mechs("PLAIN");
            opts.sasl_allow_insecure_mechs(true);
            opts.user(cfg_.username).password(cfg_.password);
        } else {
            opts.sasl_allowed_mechs("ANONYMOUS");
        }
        proton::reconnect_options ropts;
        ropts.delay(proton::duration(2000));
        ropts.max_delay(proton::duration(30000));
        ropts.max_attempts(0);
        opts.reconnect(ropts);
        c.connect(cfg_.url, opts);
    }

    void on_connection_open(proton::connection& c) override {
        c.open_receiver(cfg_.detections_topic);
        pub_sender_ = c.open_sender(cfg_.analysis_topic);
        spdlog::info("AnalysisService: connected (sub={} pub={})",
                     cfg_.detections_topic, cfg_.analysis_topic);
    }

    void on_sender_open(proton::sender& s) override {
        pub_sender_   = s;
        work_queue_   = &s.work_queue();
        std::lock_guard<std::mutex> lk(pub_ready_mu_);
        pub_ready_ = true;
        pub_ready_cv_.notify_all();
    }

    void on_message(proton::delivery& d, proton::message& m) override {
        try {
            std::string body = proton::get<std::string>(m.body());
            auto j = json::parse(body);

            Detection det{};
            det.scanner_id     = j.value("scanner_id",     "unknown");
            det.center_freq_hz = j.value("center_freq_hz", 0.0);
            det.bandwidth_hz   = j.value("bandwidth_hz",   0.0);
            det.power_db       = j.value("power_db",       0.0);
            det.timestamp_ms   = j.value("timestamp_ms",   (int64_t)0);

            // Parse embedded IQ snapshot.
            // Schema >= 1.2: base64-encoded raw float32 bytes (smaller, faster).
            // Schema  = 1.1: JSON float array (backward compat with old AcquisitionApp).
            if (j.contains("iq_snapshot_b64") && j["iq_snapshot_b64"].is_string()) {
                det.iq_snapshot = sdr::base64::decodeFloats(
                    j["iq_snapshot_b64"].get<std::string>());
                det.snapshot_sample_rate_sps =
                    j.value("snapshot_sample_rate_sps", 0.0);
            } else if (j.contains("iq_snapshot") && j["iq_snapshot"].is_array()) {
                det.iq_snapshot = j["iq_snapshot"].get<std::vector<float>>();
                det.snapshot_sample_rate_sps =
                    j.value("snapshot_sample_rate_sps", 0.0);
            }

            on_detection_(det);
        } catch (const std::exception& ex) {
            spdlog::warn("AnalysisService: failed to parse detection: {}", ex.what());
        }
        d.accept();
    }

    void publish(const std::string& body) {
        std::unique_lock<std::mutex> lk(pub_ready_mu_);
        if (!pub_ready_cv_.wait_for(lk, std::chrono::seconds(5),
                                    [this]{ return pub_ready_; })) {
            spdlog::warn("AnalysisService: publisher not ready, dropping result");
            return;
        }
        if (!work_queue_) return;
        std::string body_copy = body;
        work_queue_->add([this, body_copy]{
            if (pub_sender_ && pub_sender_.credit() > 0) {
                proton::message msg;
                msg.body(body_copy);
                msg.content_type("application/json");
                msg.durable(false);
                pub_sender_.send(msg);
            }
        });
    }

    void close() {
        if (work_queue_)
            work_queue_->add([this]{ pub_sender_.connection().close(); });
    }

    void on_transport_error(proton::transport& t) override {
        spdlog::warn("AnalysisService: transport error: {}", t.error().what());
    }

    void on_connection_error(proton::connection& c) override {
        spdlog::warn("AnalysisService: connection error: {}", c.error().what());
    }

private:
    AmqpConfig cfg_;
    std::function<void(const Detection&)> on_detection_;
    proton::sender      pub_sender_;
    proton::work_queue* work_queue_{nullptr};
    std::mutex          pub_ready_mu_;
    std::condition_variable pub_ready_cv_;
    bool                pub_ready_{false};
};

// ── AnalysisService ───────────────────────────────────────────────────────────

AnalysisService::AnalysisService(const AppConfig& cfg)
    : cfg_(cfg)
    , engine_(cfg.engine)
    , collector_(cfg.amqp, cfg.collector, cfg.streaming_ip, cfg.engine.rank)
{
#ifdef ANALYSIS_WITH_DB
    if (cfg.db.enabled) {
        db_conn_str_ = "host="     + cfg.db.host +
                       " port="    + std::to_string(cfg.db.port) +
                       " dbname="  + cfg.db.dbname +
                       " user="    + cfg.db.user +
                       " password=" + cfg.db.password;
        try {
            db_conn_ = std::make_unique<pqxx::connection>(db_conn_str_);
            spdlog::info("AnalysisService: connected to DB {}:{}/{}",
                         cfg.db.host, cfg.db.port, cfg.db.dbname);
        } catch (const std::exception& ex) {
            spdlog::warn("AnalysisService: DB connection failed (will retry): {}", ex.what());
        }
    }
#endif
}

AnalysisService::~AnalysisService()
{
    stop();
}

void AnalysisService::start()
{
    if (running_.exchange(true)) return;

    // Worker thread starts first (publisher ready before subs fire)
    worker_thread_ = std::thread(&AnalysisService::workerLoop, this);

    // Subscription thread
    sub_thread_ = std::thread(&AnalysisService::subscriptionLoop, this);

    spdlog::info("AnalysisService: started");
}

void AnalysisService::stop()
{
    if (!running_.exchange(false)) return;

    q_cv_.notify_all();

    // Close proton container via shared ptr
    if (amqp_container_) {
        amqp_container_->stop();
    }

    if (sub_thread_.joinable())    sub_thread_.join();
    if (worker_thread_.joinable()) worker_thread_.join();

    spdlog::info("AnalysisService: stopped");
}

void AnalysisService::subscriptionLoop()
{
    while (running_.load()) {
        auto on_det = [this](const Detection& d){
            // Skip re-analysis if this frequency was classified recently.
            // With fast scanning (~8s/sweep) the same signal is detected dozens
            // of times per minute; re-analyzing every hit starves the SCAN task.
            int64_t bucket = static_cast<int64_t>(d.center_freq_hz / 100'000.0);
            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            {
                std::lock_guard<std::mutex> lk(q_mu_);
                auto it = recent_analyzed_.find(bucket);
                if (it != recent_analyzed_.end() &&
                    (now_ms - it->second) < REANALYSIS_COOLDOWN_MS) {
                    return;   // already analyzed this frequency recently
                }
                recent_analyzed_[bucket] = now_ms;
                if (queue_.size() < MAX_QUEUE_DEPTH)
                    queue_.push(d);
                // If the queue is full, drop this detection — the signal will
                // be re-detected and enqueued on the next sweep pass.
            }
            q_cv_.notify_one();
        };

        amqp_handler_ = std::make_shared<ServiceAmqpHandler>(cfg_.amqp, on_det);
        amqp_container_ = std::make_shared<proton::container>(*amqp_handler_);

        try {
            amqp_container_->run();
        } catch (const std::exception& ex) {
            spdlog::error("AnalysisService: AMQP container exception: {}", ex.what());
        }

        if (!running_.load()) break;

        spdlog::info("AnalysisService: reconnecting in 3 seconds…");
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}

void AnalysisService::workerLoop()
{
    while (running_.load()) {
        Detection det;
        {
            std::unique_lock<std::mutex> lk(q_mu_);
            q_cv_.wait_for(lk, std::chrono::milliseconds(200),
                           [this]{ return !queue_.empty() || !running_.load(); });
            if (!running_.load() && queue_.empty()) break;
            if (queue_.empty()) continue;
            det = queue_.front();
            queue_.pop();
        }

        try {
            processDetection(det);
        } catch (const std::exception& ex) {
            spdlog::error("AnalysisService: processDetection exception: {}", ex.what());
        }
    }
}

// Minimum ONNX confidence from the embedded snapshot to skip full IQ collection.
// Above this threshold, the snapshot-only result is published immediately.
// Below it, full collection + feature extraction runs for higher accuracy.
static constexpr float FAST_PATH_THRESHOLD = 0.75f;

void AnalysisService::processDetection(const Detection& d)
{
    std::string req_id = generateUuid();
    spdlog::info("AnalysisService: processing {:.3f} MHz req={} snapshot={}",
                 d.center_freq_hz / 1e6, req_id,
                 d.iq_snapshot.empty() ? "no" : "yes");

    // ── Fast path: ONNX directly from the embedded snapshot ──────────────────
    // The snapshot is 1 024 IQ samples collected during the detection dwell —
    // no SDR re-acquisition needed. If ONNX confidence exceeds the threshold,
    // publish the result immediately (typically < 5 ms from detection receive).
    if (!d.iq_snapshot.empty() && engine_.onnxLoaded()) {
        AnalysisResult fast = engine_.analyzeSnapshot(
            d.iq_snapshot,
            d.snapshot_sample_rate_sps > 0 ? d.snapshot_sample_rate_sps
                                           : cfg_.collector.analysis_sample_rate_sps,
            d.center_freq_hz, req_id, cfg_.scanner_id);
        fast.timestamp_ms = d.timestamp_ms;

        if (fast.onnx_confidence >= FAST_PATH_THRESHOLD) {
            spdlog::info("AnalysisService: fast-path {:.3f} MHz → {} ({:.0f}%) — "
                         "skipping collection",
                         d.center_freq_hz / 1e6,
                         fast.digital_modulation.empty()
                             ? fast.analog_modulation : fast.digital_modulation,
                         fast.onnx_confidence * 100.f);
            fast.fast_path = true;
            publishResult(fast);
            return;
        }
        spdlog::debug("AnalysisService: snapshot ONNX confidence {:.0f}% < {:.0f}% — "
                      "falling back to full collection",
                      fast.onnx_confidence * 100.f, FAST_PATH_THRESHOLD * 100.f);
    }

    // ── Slow path: full IQ collection + feature extraction ───────────────────
    auto iq = collector_.collect(d.center_freq_hz, d.bandwidth_hz, req_id);
    if (iq.empty()) {
        spdlog::warn("AnalysisService: no IQ collected for {:.3f} MHz",
                     d.center_freq_hz / 1e6);
        return;
    }

    double sr = collector_.lastSampleRate();
    if (sr <= 0) sr = cfg_.collector.analysis_sample_rate_sps;

    AnalysisResult result = engine_.analyze(iq, sr, d.center_freq_hz,
                                             req_id, cfg_.scanner_id);
    result.timestamp_ms = d.timestamp_ms;
    publishResult(result);
}

void AnalysisService::publishResult(const AnalysisResult& r)
{
    // Build JSON output
    json j;
    j["msg_type"]       = "ANALYSIS_RESULT";
    j["detection_id"]   = r.detection_id;
    j["scanner_id"]     = r.scanner_id;
    j["timestamp_ms"]   = r.timestamp_ms;
    j["center_freq_hz"] = r.center_freq_hz;
    j["bandwidth_hz"]   = r.bandwidth_hz;
    j["snr_db"]         = r.snr_db;
    j["classified"]     = r.classified;

    if (!r.analog_modulation.empty()) {
        j["modulation"]["analog"]       = r.analog_modulation;
        j["modulation"]["analog_index"] = r.analog_index;
    }
    if (!r.digital_modulation.empty()) {
        j["modulation"]["digital"]          = r.digital_modulation;
        j["modulation"]["symbol_rate_sps"]  = r.symbol_rate_sps;
        j["modulation"]["m_ary"]            = r.m_ary;
        j["modulation"]["is_ofdm"]          = r.is_ofdm;
        j["modulation"]["is_spread"]        = r.is_spread;
    }

    j["channel_structure"]["is_burst"]         = r.is_burst;
    j["channel_structure"]["is_tdma"]          = r.is_tdma;
    j["channel_structure"]["is_fhss"]          = r.is_fhss;
    j["channel_structure"]["is_dsss"]          = r.is_dsss;
    j["channel_structure"]["burst_duty_cycle"] = r.burst_duty_cycle;
    if (r.ofdm_subcarrier_spacing_hz > 0)
        j["channel_structure"]["ofdm_subcarrier_spacing_hz"] = r.ofdm_subcarrier_spacing_hz;

    j["bitstream"]["bit_rate_bps"]     = r.bit_rate_bps;
    j["bitstream"]["line_code"]        = r.line_code;
    j["bitstream"]["fec_detected"]     = r.fec_detected;
    j["bitstream"]["has_sync_pattern"] = r.has_sync_pattern;

    auto& hyp_arr = j["hypotheses"] = json::array();
    for (const auto& h : r.hypotheses) {
        json hj;
        hj["system"]     = h.system;
        hj["category"]   = h.category;
        hj["confidence"] = h.confidence;
        hj["reasoning"]  = h.reasoning;
        hyp_arr.push_back(hj);
    }

    if (!r.classified) j["reject_reason"] = r.reject_reason;

    // Classification path metadata
    j["classification_path"]["rule_confidence"] = r.rule_confidence;
    j["classification_path"]["onnx_used"]       = r.onnx_used;
    j["classification_path"]["fast_path"]       = r.fast_path;
    if (r.onnx_used)
        j["classification_path"]["onnx_confidence"] = r.onnx_confidence;

    std::string body = j.dump();

    if (amqp_handler_) {
        amqp_handler_->publish(body);
    }

#ifdef ANALYSIS_WITH_DB
    persistResult(r);
#endif

    spdlog::info("AnalysisService: published result for {:.3f} MHz → '{}' ({})",
                 r.center_freq_hz / 1e6,
                 r.hypotheses.empty() ? "unclassified" : r.hypotheses[0].system,
                 r.fast_path ? "fast" : "slow");
}

#ifdef ANALYSIS_WITH_DB
void AnalysisService::persistResult(const AnalysisResult& r)
{
    if (db_conn_str_.empty()) return;

    // Reconnect if the connection was lost.
    if (!db_conn_ || !db_conn_->is_open()) {
        try {
            db_conn_ = std::make_unique<pqxx::connection>(db_conn_str_);
        } catch (const std::exception& ex) {
            spdlog::warn("AnalysisService: DB reconnect failed: {}", ex.what());
            db_conn_.reset();
            return;
        }
    }

    auto opt_str = [](const std::string& s) -> std::optional<std::string> {
        return s.empty() ? std::nullopt : std::optional<std::string>{s};
    };

    std::optional<std::string> hyp_sys, hyp_cat;
    std::optional<float> hyp_conf;
    if (!r.hypotheses.empty()) {
        hyp_sys  = r.hypotheses[0].system;
        hyp_cat  = r.hypotheses[0].category;
        hyp_conf = r.hypotheses[0].confidence;
    }

    try {
        pqxx::work tx{*db_conn_};
        tx.exec_params(
            "INSERT INTO analysis_results "
            " (detection_id,scanner_id,center_freq_hz,bandwidth_hz,snr_db,"
            "  classified,analog_modulation,digital_modulation,"
            "  symbol_rate_sps,bit_rate_bps,is_ofdm,is_fhss,is_burst,"
            "  hypothesis_system,hypothesis_category,hypothesis_conf,"
            "  rule_confidence,onnx_used,onnx_confidence,fast_path,reject_reason)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21)",
            r.detection_id,
            r.scanner_id,
            static_cast<int64_t>(r.center_freq_hz),
            r.bandwidth_hz > 0 ? std::optional<int>{static_cast<int>(r.bandwidth_hz)}
                               : std::nullopt,
            r.snr_db,
            r.classified,
            opt_str(r.analog_modulation),
            opt_str(r.digital_modulation),
            r.symbol_rate_sps > 0 ? std::optional<double>{r.symbol_rate_sps} : std::nullopt,
            r.bit_rate_bps   > 0 ? std::optional<double>{r.bit_rate_bps}    : std::nullopt,
            r.is_ofdm,
            r.is_fhss,
            r.is_burst,
            hyp_sys,
            hyp_cat,
            hyp_conf,
            r.rule_confidence,
            r.onnx_used,
            r.onnx_used ? std::optional<float>{r.onnx_confidence} : std::nullopt,
            r.fast_path,
            opt_str(r.reject_reason)
        );
        tx.commit();
    } catch (const std::exception& ex) {
        spdlog::warn("AnalysisService: DB write failed: {}", ex.what());
        db_conn_.reset();   // force reconnect on next call
    }
}
#endif

} // namespace analysis
