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
#include "AnalysisService.hpp"

#include <sdr/Base64.hpp>
#include <proton/container.hpp>
#include <proton/message.hpp>
#include <proton/messaging_handler.hpp>
#include <proton/connection.hpp>
#include <proton/connection_options.hpp>
#include <proton/reconnect_options.hpp>
#include <proton/sender.hpp>
#include <proton/target.hpp>
#include <proton/receiver.hpp>
#include <proton/delivery.hpp>
#include <proton/work_queue.hpp>
#include <proton/transport.hpp>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <au/units/hertz.hh>
#include <au/units/seconds.hh>

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
    using DemodCmdCb = std::function<void(const std::string&, au::QuantityD<au::Hertz>,
                                         const std::string&, const std::string&)>;

    ServiceAmqpHandler(const AmqpConfig& cfg,
                       std::function<void(const Detection&)> on_detection,
                       DemodCmdCb on_demod_cmd)
        : cfg_(cfg)
        , on_detection_(std::move(on_detection))
        , on_demod_cmd_(std::move(on_demod_cmd))
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
        if (!cfg_.demod_commands_queue.empty())
            c.open_receiver(cfg_.demod_commands_queue);
        pub_sender_   = c.open_sender(cfg_.analysis_topic);
        demod_sender_ = c.open_sender(cfg_.demod_request_queue);
        spdlog::info("AnalysisService: connected (sub={} pub={} demod_cmd={})",
                     cfg_.detections_topic, cfg_.analysis_topic,
                     cfg_.demod_commands_queue);
    }

    void on_sender_open(proton::sender& s) override {
        if (s.target().address() == cfg_.analysis_topic) {
            pub_sender_ = s;
            std::lock_guard<std::mutex> lk(pub_ready_mu_);
            pub_ready_ = true;
            pub_ready_cv_.notify_all();
        } else {
            demod_sender_ = s;
        }
        if (!work_queue_) work_queue_ = &s.work_queue();
    }

    void on_message(proton::delivery& d, proton::message& m) override {
        try {
            std::string body = proton::get<std::string>(m.body());
            auto j = json::parse(body);

            // Route demod commands to the command handler.
            static const std::initializer_list<const char*> k_cmd_types = {
                "REQUEST_DEMOD", "START_DEMOD_STREAM", "STOP_DEMOD_STREAM"
            };
            std::string msg_type = j.value("msg_type", "");
            for (auto* t : k_cmd_types) {
                if (msg_type == t) {
                    if (on_demod_cmd_)
                        on_demod_cmd_(msg_type,
                                      au::hertz(j.value("freq_hz", 0.0)),
                                      j.value("stream_id",  std::string("")),
                                      j.value("request_id", std::string("")));
                    d.accept();
                    return;
                }
            }

            Detection det{};
            det.scanner_id     = j.value("scanner_id",     "unknown");
            det.center_freq_hz = au::hertz(j.value("center_freq_hz", 0.0));
            det.bandwidth_hz   = au::hertz(j.value("bandwidth_hz",   0.0));
            det.power_db       = j.value("power_db",       0.0);
            det.timestamp_ms   = au::seconds(j.value("timestamp_ms", 0.0) * 1e-3);

            // Parse embedded IQ snapshot.
            // Schema >= 1.2: base64-encoded raw float32 bytes (smaller, faster).
            // Schema  = 1.1: JSON float array (backward compat with old AcquisitionApp).
            if (j.contains("iq_snapshot_b64") && j["iq_snapshot_b64"].is_string()) {
                det.iq_snapshot = sdr::base64::decodeFloats(
                    j["iq_snapshot_b64"].get<std::string>());
                det.snapshot_sample_rate_sps =
                    au::hertz(j.value("snapshot_sample_rate_sps", 0.0));
            } else if (j.contains("iq_snapshot") && j["iq_snapshot"].is_array()) {
                det.iq_snapshot = j["iq_snapshot"].get<std::vector<float>>();
                det.snapshot_sample_rate_sps =
                    au::hertz(j.value("snapshot_sample_rate_sps", 0.0));
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
            if (pub_sender_) {
                proton::message msg;
                msg.body(body_copy);
                msg.content_type("application/json");
                msg.durable(false);
                pub_sender_.send(msg);
            }
        });
    }

    void publishDemodRequest(const std::string& body) {
        if (!work_queue_) return;
        std::string b = body;
        work_queue_->add([this, b]() mutable {
            if (demod_sender_) {
                proton::message msg;
                msg.body(b);
                msg.content_type("application/json");
                msg.durable(false);
                demod_sender_.send(msg);
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
    DemodCmdCb                            on_demod_cmd_;
    proton::sender      pub_sender_;
    proton::sender      demod_sender_;
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
        // Connection is opened per-worker thread on first use; just log the intent here.
        spdlog::info("AnalysisService: DB persistence enabled ({}:{}/{})",
                     cfg.db.host, cfg.db.port, cfg.db.dbname);
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

    // Worker threads start first (publisher ready before subs fire).
    worker_threads_.reserve(NUM_WORKERS);
    for (int i = 0; i < NUM_WORKERS; ++i)
        worker_threads_.emplace_back(&AnalysisService::workerLoop, this);

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

    if (sub_thread_.joinable()) sub_thread_.join();
    for (auto& t : worker_threads_)
        if (t.joinable()) t.join();

    spdlog::info("AnalysisService: stopped");
}

void AnalysisService::subscriptionLoop()
{
    while (running_.load()) {
        auto on_det = [this](const Detection& d){
            // Skip re-analysis if this frequency was classified recently.
            // With fast scanning (~8s/sweep) the same signal is detected dozens
            // of times per minute; re-analyzing every hit starves the SCAN task.
            int64_t bucket = static_cast<int64_t>(d.center_freq_hz.in(au::hertz) / 100'000.0);
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

        auto on_cmd = [this](const std::string& msg_type, au::QuantityD<au::Hertz> freq_hz,
                             const std::string& stream_id, const std::string& req_id) {
            onDemodCommand(msg_type, freq_hz, stream_id, req_id);
        };
        amqp_handler_ = std::make_shared<ServiceAmqpHandler>(cfg_.amqp, on_det, on_cmd);
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
    const double center_raw = d.center_freq_hz.in(au::hertz);
    spdlog::info("AnalysisService: processing {:.3f} MHz req={} snapshot={}",
                 center_raw / 1e6, req_id,
                 d.iq_snapshot.empty() ? "no" : "yes");

    // ── Fast path: ONNX directly from the embedded snapshot ──────────────────
    // The snapshot is 1 024 IQ samples collected during the detection dwell —
    // no SDR re-acquisition needed. If ONNX confidence exceeds the threshold,
    // publish the result immediately (typically < 5 ms from detection receive).
    if (!d.iq_snapshot.empty() && engine_.onnxLoaded()) {
        au::QuantityD<au::Hertz> snap_sr =
            d.snapshot_sample_rate_sps.in(au::hertz) > 0
                ? d.snapshot_sample_rate_sps
                : cfg_.collector.analysis_sample_rate_sps;
        AnalysisResult fast = engine_.analyzeSnapshot(
            d.iq_snapshot, snap_sr, d.center_freq_hz, req_id, cfg_.scanner_id);
        fast.timestamp_ms = d.timestamp_ms;

        if (fast.onnx_confidence >= FAST_PATH_THRESHOLD) {
            spdlog::info("AnalysisService: fast-path {:.3f} MHz → {} ({:.0f}%) — "
                         "skipping collection",
                         center_raw / 1e6,
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
                     center_raw / 1e6);
        return;
    }

    au::QuantityD<au::Hertz> sr = collector_.lastSampleRate();
    if (sr.in(au::hertz) <= 0)
        sr = cfg_.collector.analysis_sample_rate_sps;

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
    j["timestamp_ms"]   = r.timestamp_ms.in(au::seconds) * 1e3;
    j["center_freq_hz"] = r.center_freq_hz.in(au::hertz);
    j["bandwidth_hz"]   = r.bandwidth_hz.in(au::hertz);
    j["snr_db"]         = r.snr_db;
    j["classified"]     = r.classified;

    if (!r.analog_modulation.empty()) {
        j["modulation"]["analog"]       = r.analog_modulation;
        j["modulation"]["analog_index"] = r.analog_index;
    }
    if (!r.digital_modulation.empty()) {
        j["modulation"]["digital"]          = r.digital_modulation;
        j["modulation"]["symbol_rate_sps"]  = r.symbol_rate_sps.in(au::hertz);
        j["modulation"]["m_ary"]            = r.m_ary;
        j["modulation"]["is_ofdm"]          = r.is_ofdm;
        j["modulation"]["is_spread"]        = r.is_spread;
    }

    j["channel_structure"]["is_burst"]         = r.is_burst;
    j["channel_structure"]["is_tdma"]          = r.is_tdma;
    j["channel_structure"]["is_fhss"]          = r.is_fhss;
    j["channel_structure"]["is_dsss"]          = r.is_dsss;
    j["channel_structure"]["burst_duty_cycle"] = r.burst_duty_cycle;
    if (r.ofdm_subcarrier_spacing_hz.in(au::hertz) > 0)
        j["channel_structure"]["ofdm_subcarrier_spacing_hz"] = r.ofdm_subcarrier_spacing_hz.in(au::hertz);

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

    // Cache result so onDemodCommand can forward it on demand.
    {
        int64_t bucket = static_cast<int64_t>(r.center_freq_hz.in(au::hertz) / 100'000.0);
        std::lock_guard<std::mutex> lk(q_mu_);
        recent_results_[bucket] = r;
    }

    if (amqp_handler_) {
        amqp_handler_->publish(body);
    }

#ifdef ANALYSIS_WITH_DB
    persistResult(r);
#endif

    spdlog::info("AnalysisService: published result for {:.3f} MHz → '{}' ({})",
                 r.center_freq_hz.in(au::hertz) / 1e6,
                 r.hypotheses.empty() ? "unclassified" : r.hypotheses[0].system,
                 r.fast_path ? "fast" : "slow");
}

void AnalysisService::onDemodCommand(const std::string& msg_type,
                                     au::QuantityD<au::Hertz> freq_hz,
                                     const std::string& stream_id,
                                     const std::string& request_id)
{
    // STOP_DEMOD_STREAM: forward directly — no cache lookup required.
    if (msg_type == "STOP_DEMOD_STREAM") {
        if (stream_id.empty()) {
            spdlog::warn("AnalysisService: STOP_DEMOD_STREAM with no stream_id — ignored");
            return;
        }
        json j;
        j["msg_type"]  = "STOP_DEMOD_STREAM";
        j["stream_id"] = stream_id;
        if (amqp_handler_) amqp_handler_->publishDemodRequest(j.dump());
        spdlog::info("AnalysisService: forwarded STOP_DEMOD_STREAM stream={}", stream_id);
        return;
    }

    // REQUEST_DEMOD / START_DEMOD_STREAM: look up cache and forward with signal params.
    const double freq_raw = freq_hz.in(au::hertz);
    if (freq_raw <= 0) {
        spdlog::warn("AnalysisService: {} with invalid freq — ignored", msg_type);
        return;
    }

    int64_t bucket = static_cast<int64_t>(freq_raw / 100'000.0);
    AnalysisResult result;
    {
        std::lock_guard<std::mutex> lk(q_mu_);
        auto it = recent_results_.find(bucket);
        if (it == recent_results_.end()) {
            spdlog::warn("AnalysisService: {} for {:.3f} MHz — no recent result, ignored",
                         msg_type, freq_raw / 1e6);
            return;
        }
        result = it->second;
    }

    spdlog::info("AnalysisService: {} {:.3f} MHz → forwarding to DemodApp",
                 msg_type, freq_raw / 1e6);
    publishDemodRequest(msg_type, result, stream_id, request_id);
}

void AnalysisService::publishDemodRequest(const std::string& msg_type,
                                          const AnalysisResult& r,
                                          const std::string& stream_id,
                                          const std::string& request_id)
{
    std::string modulation = r.digital_modulation.empty()
                           ? r.analog_modulation
                           : r.digital_modulation;
    if (modulation.empty()) {
        spdlog::warn("AnalysisService: publishDemodRequest — no modulation for {:.3f} MHz",
                     r.center_freq_hz.in(au::hertz) / 1e6);
        return;
    }

    json j;
    j["msg_type"]        = msg_type;
    j["request_id"]      = request_id.empty() ? generateUuid() : request_id;
    j["modulation"]      = modulation;
    j["center_freq_hz"]  = r.center_freq_hz.in(au::hertz);
    j["bandwidth_hz"]    = r.bandwidth_hz.in(au::hertz);
    j["symbol_rate_sps"] = r.symbol_rate_sps.in(au::hertz);
    j["confidence"]      = r.onnx_confidence;
    j["timestamp_ms"]    = r.timestamp_ms.in(au::seconds) * 1e3;
    if (!stream_id.empty())
        j["stream_id"]   = stream_id;

    if (amqp_handler_)
        amqp_handler_->publishDemodRequest(j.dump());
}

#ifdef ANALYSIS_WITH_DB
void AnalysisService::persistResult(const AnalysisResult& r)
{
    if (db_conn_str_.empty()) return;

    thread_local std::unique_ptr<pqxx::connection> t_conn;
    if (!t_conn || !t_conn->is_open()) {
        try {
            t_conn = std::make_unique<pqxx::connection>(db_conn_str_);
        } catch (const std::exception& ex) {
            spdlog::warn("AnalysisService: DB connect failed: {}", ex.what());
            t_conn.reset();
            return;
        }
    }

    // Build combined modulation string: prefer digital; append M-ary order.
    std::string mod = r.digital_modulation.empty() ? r.analog_modulation
                                                    : r.digital_modulation;
    if (!r.digital_modulation.empty() && r.m_ary > 1)
        mod += "-" + std::to_string(r.m_ary);

    const std::string mod_class = mod.empty()                      ? "unclassified"
                                  : !r.digital_modulation.empty()  ? "digital"
                                                                    : "analog";

    std::string hyp_sys, hyp_cat;
    float hyp_conf = 0.f;
    if (!r.hypotheses.empty()) {
        hyp_sys  = r.hypotheses[0].system;
        hyp_cat  = r.hypotheses[0].category;
        hyp_conf = r.hypotheses[0].confidence;
    }

    const double cf_raw  = r.center_freq_hz.in(au::hertz);
    const double bw_raw  = r.bandwidth_hz.in(au::hertz);
    const double sr_raw  = r.symbol_rate_sps.in(au::hertz);

    std::optional<double> bw_opt;
    if (bw_raw > 0) bw_opt = bw_raw;
    std::optional<double> sr_opt;
    if (sr_raw > 0) sr_opt = sr_raw;
    std::optional<double> br_opt;
    if (r.bit_rate_bps > 0) br_opt = r.bit_rate_bps;
    std::optional<float> oc_opt;
    if (r.onnx_used) oc_opt = r.onnx_confidence;

    try {
        pqxx::work tx{*t_conn};

        // ── Step 1: find existing row with same (freq ±10 kHz, modulation, BW ±50%) ──
        // Same identity → this signal was seen before; update last_seen.
        pqxx::result existing;
        if (!mod.empty()) {
            existing = tx.exec_params(
                "SELECT id FROM signals "
                "WHERE abs(freq_hz - $1) < 10000 "
                "  AND modulation = $2 "
                "  AND ($3::double precision IS NULL OR bandwidth_hz IS NULL "
                "       OR abs(bandwidth_hz - $3) / GREATEST(bandwidth_hz, 1000.0) < 0.5) "
                "ORDER BY abs(freq_hz - $1) ASC LIMIT 1",
                cf_raw, mod, bw_opt);
        }

        if (!existing.empty()) {
            // Known signal — refresh classification metadata and last_seen.
            tx.exec_params(
                "UPDATE signals SET "
                "  last_seen       = now(), "
                "  snr_db          = $2, "
                "  bandwidth_hz    = COALESCE(NULLIF(bandwidth_hz, 0), $3), "
                "  hypothesis      = $4, "
                "  hyp_category    = $5, "
                "  hyp_confidence  = $6, "
                "  classified      = $7, "
                "  rule_confidence = $8, "
                "  onnx_used       = $9, "
                "  onnx_confidence = $10, "
                "  fast_path       = $11, "
                "  reject_reason   = $12, "
                "  hits            = hits + 1 "
                "WHERE id = $1",
                existing[0][0].as<long long>(),
                r.snr_db, bw_opt,
                hyp_sys, hyp_cat, hyp_conf,
                r.classified, r.rule_confidence,
                r.onnx_used, oc_opt, r.fast_path, r.reject_reason);

        } else {
            // ── Step 2: find unclassified row planted by AcquisitionApp ──
            // Same freq + similar BW, not yet classified → fill in modulation.
            auto unclass = tx.exec_params(
                "SELECT id FROM signals "
                "WHERE abs(freq_hz - $1) < 10000 "
                "  AND classified = false "
                "  AND ($2::double precision IS NULL OR bandwidth_hz IS NULL "
                "       OR abs(bandwidth_hz - $2) / GREATEST(bandwidth_hz, 1000.0) < 0.5) "
                "  AND last_seen > now() - interval '10 minutes' "
                "ORDER BY abs(freq_hz - $1) ASC LIMIT 1",
                cf_raw, bw_opt);

            if (!unclass.empty()) {
                // Promote unclassified row → classified signal.
                tx.exec_params(
                    "UPDATE signals SET "
                    "  last_seen       = now(), "
                    "  freq_hz         = $2, "
                    "  freq_mhz        = $3, "
                    "  bandwidth_hz    = COALESCE(NULLIF(bandwidth_hz, 0), $4), "
                    "  snr_db          = $5, "
                    "  modulation      = $6, "
                    "  mod_class       = $7, "
                    "  is_ofdm         = $8, "
                    "  is_burst        = $9, "
                    "  is_fhss         = $10, "
                    "  symbol_rate_sps = $11, "
                    "  bit_rate_bps    = $12, "
                    "  hypothesis      = $13, "
                    "  hyp_category    = $14, "
                    "  hyp_confidence  = $15, "
                    "  classified      = $16, "
                    "  rule_confidence = $17, "
                    "  onnx_used       = $18, "
                    "  onnx_confidence = $19, "
                    "  fast_path       = $20, "
                    "  reject_reason   = $21, "
                    "  hits            = hits + 1 "
                    "WHERE id = $1",
                    unclass[0][0].as<long long>(),
                    cf_raw, cf_raw / 1e6, bw_opt,
                    r.snr_db, mod, mod_class,
                    r.is_ofdm, r.is_burst, r.is_fhss,
                    sr_opt, br_opt,
                    hyp_sys, hyp_cat, hyp_conf,
                    r.classified, r.rule_confidence,
                    r.onnx_used, oc_opt, r.fast_path, r.reject_reason);

            } else {
                // ── Step 3: entirely new signal — insert classified row ──
                // Happens when AnalysisApp classifies a freq without a prior
                // AcquisitionApp detection (e.g. replay), or when the unclassified
                // row has expired.
                tx.exec_params(
                    "INSERT INTO signals "
                    "(first_seen, last_seen, freq_hz, freq_mhz, bandwidth_hz, snr_db, "
                    " scanner_id, modulation, mod_class, is_ofdm, is_burst, is_fhss, "
                    " symbol_rate_sps, bit_rate_bps, "
                    " hypothesis, hyp_category, hyp_confidence, "
                    " classified, rule_confidence, onnx_used, onnx_confidence, "
                    " fast_path, reject_reason) "
                    "VALUES (now(), now(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, "
                    "        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)",
                    cf_raw, cf_raw / 1e6, bw_opt,
                    r.snr_db, r.scanner_id,
                    mod, mod_class,
                    r.is_ofdm, r.is_burst, r.is_fhss,
                    sr_opt, br_opt,
                    hyp_sys, hyp_cat, hyp_conf,
                    r.classified, r.rule_confidence,
                    r.onnx_used, oc_opt, r.fast_path, r.reject_reason);
            }
        }

        tx.commit();
        spdlog::debug("AnalysisService: persisted {:.3f} MHz mod={}",
                      cf_raw / 1e6, mod.empty() ? "(none)" : mod);

    } catch (const std::exception& ex) {
        spdlog::warn("AnalysisService: DB write failed: {}", ex.what());
        t_conn.reset();
    }
}
#endif

} // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
