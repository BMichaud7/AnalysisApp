#include "IqCollector.hpp"
#include <sdr/Types.hpp>

#include <proton/container.hpp>
#include <proton/message.hpp>
#include <proton/messaging_handler.hpp>
#include <proton/connection.hpp>
#include <proton/connection_options.hpp>
#include <proton/sender.hpp>
#include <proton/sender_options.hpp>
#include <proton/receiver.hpp>
#include <proton/receiver_options.hpp>
#include <proton/source_options.hpp>
#include <proton/target_options.hpp>
#include <proton/delivery.hpp>
#include <proton/work_queue.hpp>
#include <proton/transport.hpp>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/select.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stdexcept>

namespace analysis {

using json = nlohmann::json;
using namespace std::chrono;

// ── JSON encoding for task request ────────────────────────────────────────────
// dest_ip only — controller allocates a port from its pool and returns it in
// streams[0].udp_port of the TASK_RESPONSE.  We bind AFTER getting that port.

static std::string buildTaskRequestJson(const std::string& request_id,
                                         double center_freq_hz,
                                         double bandwidth_hz,
                                         double sample_rate_sps,
                                         int64_t duration_ms,
                                         const std::string& dest_ip,
                                         int rank)
{
    auto ts = duration_cast<milliseconds>(
                  system_clock::now().time_since_epoch()).count();

    json j = {
        {"msg_type",        "TASK_REQUEST"},
        {"schema_version",  sdr::SCHEMA_VERSION},
        {"request_id",      request_id},
        {"timestamp_ms",    ts},
        {"task_type",       "NARROWBAND"},
        {"rank",            rank},
        {"schedule", {
            {"mode",         "IMMEDIATE"},
            {"duration_ms",  duration_ms}
        }},
        {"rf", {
            {"center_freq_hz",  center_freq_hz},
            {"bandwidth_hz",    bandwidth_hz},
            {"sample_rate_sps", sample_rate_sps},
            {"rx_count",        1}
        }},
        {"streaming", {{"dest_ip", dest_ip}}}
    };
    return j.dump();
}

static std::string buildTaskStopJson(const std::string& request_id,
                                      const std::string& task_id)
{
    auto ts = duration_cast<milliseconds>(
                  system_clock::now().time_since_epoch()).count();
    json j = {
        {"msg_type",    "TASK_STOP"},
        {"request_id",  request_id},
        {"task_id",     task_id},
        {"timestamp_ms", ts},
        {"reason",      "collection complete"}
    };
    return j.dump();
}

// ── Proton one-shot request/response handler ──────────────────────────────────

class RpcHandler : public proton::messaging_handler {
public:
    struct BrokerCfg {
        std::string url, username, password;
        std::string request_queue, response_queue;
    };

    RpcHandler(BrokerCfg cfg, std::string request_body,
               std::string request_id,
               std::function<void(const std::string&)> on_response,
               std::function<void(const std::string&)> on_error)
        : cfg_(std::move(cfg))
        , request_body_(std::move(request_body))
        , request_id_(std::move(request_id))
        , on_response_(std::move(on_response))
        , on_error_(std::move(on_error))
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
        c.connect(cfg_.url, opts);
    }

    void on_connection_open(proton::connection& c) override {
        // Use ANYCAST capabilities so Artemis creates proper queue addresses
        // (MULTICAST default stalls credit propagation by ~15s per connection).
        proton::sender_options sopts;
        sopts.target(proton::target_options().capabilities(
            {proton::symbol("queue")}));
        sender_ = c.open_sender(cfg_.request_queue, sopts);
        if (!cfg_.response_queue.empty()) {
            proton::receiver_options ropts;
            ropts.source(proton::source_options().capabilities(
                {proton::symbol("queue")}));
            receiver_ = c.open_receiver(cfg_.response_queue, ropts);
        }
    }

    void on_sender_open(proton::sender& s) override {
        proton::message msg;
        msg.body(request_body_);
        msg.content_type("application/json");
        msg.durable(false);
        s.send(msg);
        spdlog::debug("IqCollector: sent request ({} bytes)", request_body_.size());
        // Fire-and-forget (e.g. TASK_STOP): close after sending, no reply needed
        if (cfg_.response_queue.empty())
            s.connection().close();
    }

    void on_message(proton::delivery& d, proton::message& m) override {
        d.accept();
        try {
            std::string body = proton::get<std::string>(m.body());
            // Match by request_id — discard stale responses from other apps
            // that share the same sdr.task.response queue (e.g. AcquisitionApp)
            auto j = json::parse(body);
            if (!request_id_.empty() &&
                j.value("request_id", "") != request_id_) {
                spdlog::debug("IqCollector: discarding stale response for req={}",
                              j.value("request_id", "?"));
                return;  // stay connected, wait for our response
            }
            on_response_(body);
        } catch (const std::exception& ex) {
            on_error_(std::string("on_message parse error: ") + ex.what());
        }
        sender_.connection().close();
    }

    void on_transport_error(proton::transport& t) override { on_error_(t.error().what()); }
    void on_connection_error(proton::connection& c) override { on_error_(c.error().what()); }
    void on_error(const proton::error_condition& e) override { on_error_(e.what()); }

private:
    BrokerCfg    cfg_;
    std::string  request_body_;
    std::string  request_id_;
    proton::sender   sender_;
    proton::receiver receiver_;
    std::function<void(const std::string&)> on_response_;
    std::function<void(const std::string&)> on_error_;
};

// ── UDP helpers ───────────────────────────────────────────────────────────────

// Bind to the specific port the controller allocated — called AFTER parsing
// streams[0].udp_port from the TASK_RESPONSE, never before.
static int openBoundUdpSocket(int port)
{
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return -1;

    int rcvbuf = 16 * 1024 * 1024;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = INADDR_ANY;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

static std::vector<float> receiveIq(int udp_fd, int target_samples, int timeout_ms)
{
    std::vector<float> iq;
    iq.reserve(static_cast<size_t>(target_samples) * 2);

    auto deadline = steady_clock::now() + milliseconds(timeout_ms);
    constexpr int BUF = 65536;
    std::vector<uint8_t> buf(BUF);

    while (static_cast<int>(iq.size()) / 2 < target_samples) {
        auto now = steady_clock::now();
        if (now >= deadline) {
            spdlog::warn("IqCollector: UDP timeout: collected {}/{} samples",
                         static_cast<int>(iq.size()) / 2, target_samples);
            break;
        }
        long remaining_ms = duration_cast<milliseconds>(deadline - now).count();

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(udp_fd, &fds);
        timeval tv;
        tv.tv_sec  = remaining_ms / 1000;
        tv.tv_usec = (remaining_ms % 1000) * 1000;

        int ret = ::select(udp_fd + 1, &fds, nullptr, nullptr, &tv);
        if (ret <= 0) break;

        ssize_t n = ::recv(udp_fd, buf.data(), BUF, 0);
        if (n < static_cast<ssize_t>(sdr::IQ_PACKET_HEADER_SIZE)) continue;

        const auto* hdr = reinterpret_cast<const sdr::IqPacketHeader*>(buf.data());
        if (hdr->magic != sdr::IQ_PACKET_MAGIC) {
            spdlog::debug("IqCollector: bad magic in UDP packet");
            continue;
        }

        int n_samp = hdr->num_samples;
        int payload_bytes = static_cast<int>(n) - sdr::IQ_PACKET_HEADER_SIZE;
        int max_samp = payload_bytes / static_cast<int>(sizeof(float)) / 2;
        n_samp = std::min(n_samp, max_samp);

        const float* samples = reinterpret_cast<const float*>(
                                   buf.data() + sdr::IQ_PACKET_HEADER_SIZE);
        for (int i = 0; i < n_samp * 2; ++i)
            iq.push_back(samples[i]);
    }

    return iq;
}

// ── IqCollector implementation ────────────────────────────────────────────────

IqCollector::IqCollector(const AmqpConfig& amqp_cfg,
                          const CollectorConfig& col_cfg,
                          const std::string& local_ip,
                          int task_rank)
    : amqp_cfg_(amqp_cfg)
    , col_cfg_(col_cfg)
    , local_ip_(local_ip)
    , task_rank_(task_rank)
{}

IqCollector::~IqCollector() = default;
bool IqCollector::connect()    { return true; }
void IqCollector::disconnect() {}

std::vector<float> IqCollector::collect(double center_freq_hz,
                                         double bandwidth_hz,
                                         const std::string& request_id)
{
    double bw     = std::max(bandwidth_hz, 25000.0);
    double sr     = col_cfg_.analysis_sample_rate_sps;
    int64_t dur_ms = static_cast<int64_t>(
                         (double)col_cfg_.collect_samples / sr * 1000.0) + 500;

    // Step 1: submit task — no port specified, controller allocates from pool
    std::string req_json = buildTaskRequestJson(
        request_id, center_freq_hz, bw, sr, dur_ms, local_ip_, task_rank_);

    RpcHandler::BrokerCfg bcfg{
        amqp_cfg_.url, amqp_cfg_.username, amqp_cfg_.password,
        amqp_cfg_.task_request_queue, amqp_cfg_.task_response_queue
    };

    std::mutex mu;
    std::condition_variable cv;
    std::string response_body, error_body;
    bool done = false;

    auto on_resp = [&](const std::string& body) {
        std::lock_guard<std::mutex> lk(mu);
        response_body = body; done = true; cv.notify_all();
    };
    auto on_err = [&](const std::string& err) {
        std::lock_guard<std::mutex> lk(mu);
        error_body = err; done = true; cv.notify_all();
    };

    RpcHandler handler(bcfg, req_json, request_id, on_resp, on_err);
    proton::container container(handler);
    std::thread amqp_thread([&]{ container.run(); });

    bool timed_out = false;
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, milliseconds(col_cfg_.analysis_timeout_ms),
                    [&]{ return done; });
        timed_out = !done;
    }
    // Only stop the container on timeout — on success the connection closes
    // itself gracefully in on_message(); an unconditional stop() races with
    // that close and triggers a spurious connection-aborted error.
    if (timed_out) container.stop();
    if (amqp_thread.joinable()) amqp_thread.join();

    if (!error_body.empty()) {
        spdlog::error("IqCollector: AMQP error: {}", error_body);
        return {};
    }

    // Step 2: parse TASK_RESPONSE, get controller-allocated UDP port
    std::string accepted_task_id;
    int udp_port = 0;
    try {
        auto j = json::parse(response_body);
        bool accepted = (j.value("status", "") == "ACCEPTED");
        if (!accepted) {
            spdlog::warn("IqCollector: task rejected: {}",
                         j.value("reject_reason", "unknown"));
            return {};
        }
        accepted_task_id = j.value("task_id", "");

        if (j.contains("streams") && j["streams"].is_array() && !j["streams"].empty()) {
            udp_port = j["streams"][0].value("udp_port", 0);
            last_sr_ = j["streams"][0].value("sample_rate_sps", sr);
        } else {
            last_sr_ = sr;
        }

        if (udp_port == 0) {
            spdlog::error("IqCollector: ACCEPTED response missing streams[0].udp_port");
            return {};
        }
        spdlog::debug("IqCollector: task accepted id='{}' udp_port={} sr={:.0f} Hz",
                      accepted_task_id, udp_port, last_sr_);
    } catch (const std::exception& ex) {
        spdlog::error("IqCollector: failed to parse TASK_RESPONSE: {}", ex.what());
        return {};
    }

    // Step 3: bind to the controller-allocated port
    int udp_fd = openBoundUdpSocket(udp_port);
    if (udp_fd < 0) {
        spdlog::error("IqCollector: failed to bind UDP socket on port {}: {}",
                      udp_port, strerror(errno));
        return {};
    }
    spdlog::info("IqCollector: UDP bound on port {}", udp_port);

    // Step 4: receive IQ
    auto iq = receiveIq(udp_fd, col_cfg_.collect_samples,
                         col_cfg_.analysis_timeout_ms);
    ::close(udp_fd);

    spdlog::info("IqCollector: collected {} samples (sr={:.0f} Hz)",
                 static_cast<int>(iq.size()) / 2, last_sr_);

    // Step 5: fire-and-forget TASK_STOP (no response expected)
    if (!accepted_task_id.empty()) {
        std::string stop_json = buildTaskStopJson(request_id + "_stop",
                                                   accepted_task_id);
        auto noop = [](const std::string&){};
        // Empty response_queue → fire-and-forget; on_sender_open closes immediately
        RpcHandler::BrokerCfg stop_bcfg{
            amqp_cfg_.url, amqp_cfg_.username, amqp_cfg_.password,
            amqp_cfg_.task_request_queue, ""
        };
        RpcHandler stop_handler(stop_bcfg, stop_json, "", noop, noop);
        proton::container stop_c(stop_handler);
        std::thread stop_th([&]{ stop_c.run(); });
        if (stop_th.joinable()) stop_th.join();
    }

    return iq;
}

} // namespace analysis
