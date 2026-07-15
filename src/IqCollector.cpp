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
#include "IqCollector.hpp"
#include <sdr/Types.hpp>
#include <au/units/seconds.hh>

#include <au/prefix.hh>
#include <proton/container.hpp>
#include <proton/message.hpp>
#include <proton/messaging_handler.hpp>
#include <proton/connection.hpp>
#include <proton/connection_options.hpp>
#include <proton/reconnect_options.hpp>
#include <proton/sender.hpp>
#include <proton/sender_options.hpp>
#include <proton/receiver.hpp>
#include <proton/receiver_options.hpp>
#include <proton/source_options.hpp>
#include <proton/target_options.hpp>
#include <proton/delivery.hpp>
#include <proton/work_queue.hpp>
#include <proton/transport.hpp>
#include <proton/symbol.hpp>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/select.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace analysis {

using json = nlohmann::json;
using namespace std::chrono;

// ── Persistent AMQP channel for task request/response ────────────────────────
// Mirrors TaskAmqpChannel in AcquisitionApp: connects once at construction,
// uses a dynamic (unique) receiver so no other subscriber steals responses,
// and reuses the connection across all collect() calls — paying the ~30s
// Artemis settlement cost once instead of on every analysis.

class IqTaskChannel : public proton::messaging_handler {
public:
    IqTaskChannel(std::string url, std::string user, std::string pass,
                  std::string req_queue)
        : url_(std::move(url)), user_(std::move(user)), pass_(std::move(pass))
        , req_queue_(std::move(req_queue))
    {}

    ~IqTaskChannel() { stop(); }

    void start(int connect_timeout_ms = 60000) {
        container_ = std::make_unique<proton::container>(*this);
        thread_ = std::thread([this]{ container_->run(); });
        std::unique_lock<std::mutex> lk(mu_);
        ready_cv_.wait_for(lk, milliseconds(connect_timeout_ms),
                           [this]{ return ready_; });
        if (!ready_)
            spdlog::warn("[IqTaskChannel] not ready after {}ms", connect_timeout_ms);
    }

    void stop() {
        if (container_) {
            proton::work_queue* wq = nullptr;
            { std::lock_guard<std::mutex> lk(mu_); wq = wq_; }
            if (wq)
                wq->add([this]{ sender_.connection().close(); });
            // Always stop the container: if wq_ is stale (transport error,
            // reconnect in progress) the close() above is silently dropped
            // and thread_.join() would hang indefinitely without this.
            container_->stop();
            if (thread_.joinable()) thread_.join();
            { std::lock_guard<std::mutex> lk(mu_); wq_ = nullptr; }
            container_.reset();
        }
    }

    // Send msg_body and block until the correlated response arrives.
    // Returns {} if timed out. Only one exchange() at a time (enforced by
    // collect_mu_ in IqCollector — do not call concurrently).
    std::string exchange(const std::string& msg_body,
                         const std::string& corr_id,
                         int timeout_ms) {
        proton::work_queue* wq = nullptr;
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (!ready_cv_.wait_for(lk, milliseconds(timeout_ms),
                                    [this]{ return ready_; })) {
                spdlog::warn("[IqTaskChannel] not ready");
                return {};
            }
            pending_corr_  = corr_id;
            pending_body_.clear();
            pending_done_  = false;
            wq = wq_;
        }
        wq->add([this, msg_body]() mutable {
            proton::message msg;
            msg.body(msg_body);
            msg.content_type("application/json");
            msg.reply_to(reply_addr_);
            // Do NOT check credit() — proton queues when credit arrives.
            if (sender_) sender_.send(msg);
        });
        std::unique_lock<std::mutex> lk(mu_);
        result_cv_.wait_for(lk, milliseconds(timeout_ms),
                            [this]{ return pending_done_; });
        return pending_body_;
    }

    // Fire-and-forget (TASK_STOP).
    void send(const std::string& msg_body) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!ready_) return;
        wq_->add([this, msg_body]() mutable {
            proton::message msg;
            msg.body(msg_body);
            msg.content_type("application/json");
            if (sender_) sender_.send(msg);
        });
    }

    // proton callbacks ─────────────────────────────────────────────────────────
    void on_container_start(proton::container& c) override {
        proton::connection_options opts;
        if (!user_.empty()) {
            opts.sasl_allowed_mechs("PLAIN");
            opts.sasl_allow_insecure_mechs(true);
            opts.user(user_).password(pass_);
        } else {
            opts.sasl_allowed_mechs("ANONYMOUS");
        }
        proton::reconnect_options ropts;
        ropts.delay(proton::duration(2000));
        ropts.max_delay(proton::duration(30000));
        ropts.max_attempts(0);
        opts.reconnect(ropts);
        c.connect(url_, opts);
    }

    void on_connection_open(proton::connection& conn) override {
        proton::sender_options sopts;
        sopts.target(proton::target_options().capabilities({proton::symbol("queue")}));
        sender_ = conn.open_sender(req_queue_, sopts);

        proton::receiver_options ropts;
        ropts.source(proton::source_options().dynamic(true));
        conn.open_receiver("", ropts);
    }

    void on_receiver_open(proton::receiver& r) override {
        reply_addr_ = r.source().address();
        spdlog::info("[IqTaskChannel] connected, reply_to={}", reply_addr_);
        std::lock_guard<std::mutex> lk(mu_);
        wq_ = &r.work_queue();
        ready_ = true;
        ready_cv_.notify_all();
    }

    void on_message(proton::delivery& d, proton::message& msg) override {
        d.accept();
        try {
            std::string b = proton::get<std::string>(msg.body());
            auto j = json::parse(b);
            std::lock_guard<std::mutex> lk(mu_);
            if (!pending_corr_.empty() &&
                (j.value("request_id", "") == pending_corr_ ||
                 j.value("correlation_id", "") == pending_corr_)) {
                pending_body_ = b;
                pending_done_ = true;
                pending_corr_.clear();
                result_cv_.notify_all();
            }
        } catch (...) {}
    }

    void on_transport_error(proton::transport& t) override {
        spdlog::warn("[IqTaskChannel] transport error: {}", t.error().what());
        std::lock_guard<std::mutex> lk(mu_);
        ready_ = false;
        reply_addr_.clear();
    }
    void on_connection_error(proton::connection& c) override {
        spdlog::warn("[IqTaskChannel] connection error: {}", c.error().what());
    }

private:
    std::string url_, user_, pass_, req_queue_;

    std::unique_ptr<proton::container> container_;
    std::thread  thread_;

    proton::sender      sender_;
    proton::work_queue* wq_{nullptr};
    std::string         reply_addr_;

    std::mutex              mu_;
    std::condition_variable ready_cv_;
    std::condition_variable result_cv_;
    bool        ready_{false};
    std::string pending_corr_;
    std::string pending_body_;
    bool        pending_done_{false};
};

// ── JSON helpers ──────────────────────────────────────────────────────────────

static std::string buildTaskRequestJson(const std::string& request_id,
                                         double center_freq_hz,
                                         double bandwidth_hz,
                                         double sample_rate_sps,
                                         int64_t duration_ms,
                                         const std::string& dest_ip,
                                         int rank,
                                         uint16_t prebound_port = 0)
{
    auto ts = duration_cast<milliseconds>(
                  system_clock::now().time_since_epoch()).count();
    json streaming = {{"dest_ip", dest_ip}};
    if (prebound_port > 0)
        streaming["dest_ports"] = json::array({static_cast<int>(prebound_port)});

    return json{
        {"msg_type",        "TASK_REQUEST"},
        {"schema_version",  sdr::SCHEMA_VERSION},
        {"request_id",      request_id},
        {"timestamp_ms",    ts},
        {"task_type",       "NARROWBAND"},
        {"rank",            rank},
        {"schedule", {{"mode", "IMMEDIATE"}, {"duration_ms", duration_ms}}},
        {"rf", {
            {"center_freq_hz",  center_freq_hz},
            {"bandwidth_hz",    bandwidth_hz},
            {"sample_rate_sps", sample_rate_sps},
            {"rx_count",        1}
        }},
        {"streaming", streaming}
    }.dump();
}

static std::string buildTaskStopJson(const std::string& request_id,
                                      const std::string& task_id)
{
    return json{
        {"msg_type",     "TASK_STOP"},
        {"request_id",   request_id},
        {"task_id",      task_id},
        {"timestamp_ms", duration_cast<milliseconds>(
                             system_clock::now().time_since_epoch()).count()},
        {"reason",       "collection complete"}
    }.dump();
}

// ── UDP helpers ───────────────────────────────────────────────────────────────

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
        ::close(fd); return -1;
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
        fd_set fds; FD_ZERO(&fds); FD_SET(udp_fd, &fds);
        timeval tv{remaining_ms / 1000, (remaining_ms % 1000) * 1000};
        if (::select(udp_fd + 1, &fds, nullptr, nullptr, &tv) <= 0) break;

        ssize_t n = ::recv(udp_fd, buf.data(), BUF, 0);
        if (n < static_cast<ssize_t>(sdr::IQ_PACKET_HEADER_SIZE)) continue;
        const auto* hdr = reinterpret_cast<const sdr::IqPacketHeader*>(buf.data());
        if (hdr->magic != sdr::IQ_PACKET_MAGIC) continue;

        int n_samp = std::min(hdr->num_samples,
                              (uint16_t)((static_cast<int>(n) - sdr::IQ_PACKET_HEADER_SIZE)
                                         / (int)sizeof(float) / 2));
        const float* samples = reinterpret_cast<const float*>(
                                   buf.data() + sdr::IQ_PACKET_HEADER_SIZE);
        iq.insert(iq.end(), samples, samples + n_samp * 2);
    }
    return iq;
}

// ── IqCollector ───────────────────────────────────────────────────────────────

IqCollector::IqCollector(const AmqpConfig& amqp_cfg,
                          const CollectorConfig& col_cfg,
                          const std::string& local_ip,
                          int task_rank)
    : amqp_cfg_(amqp_cfg)
    , col_cfg_(col_cfg)
    , local_ip_(local_ip)
    , task_rank_(task_rank)
{
    ch_ = std::make_unique<IqTaskChannel>(
        amqp_cfg_.url, amqp_cfg_.username, amqp_cfg_.password,
        amqp_cfg_.task_request_queue);
    ch_->start(60000);
}

IqCollector::~IqCollector() = default;

std::vector<float> IqCollector::collect(au::QuantityD<au::Hertz> center_freq_hz,
                                         au::QuantityD<au::Hertz> bandwidth_hz,
                                         const std::string& request_id)
{
    // Serialize: only one NARROWBAND task at a time (SDR is an exclusive resource).
    std::lock_guard<std::mutex> collect_guard(collect_mu_);

    const double bw = std::max(bandwidth_hz.in(au::hertz), 25000.0);
    const double sr = col_cfg_.analysis_sample_rate_sps.in(au::hertz);
    int64_t dur_ms = static_cast<int64_t>((double)col_cfg_.collect_samples / sr * 1000.0) + 500;

    // Pre-bind UDP socket before submitting the task — controller streams
    // to the ready socket immediately without the 30s AMQP-latency data-loss race.
    int      prebound_fd   = openBoundUdpSocket(0);
    uint16_t prebound_port = 0;
    if (prebound_fd >= 0) {
        struct sockaddr_in sa{}; socklen_t sl = sizeof(sa);
        if (getsockname(prebound_fd, reinterpret_cast<sockaddr*>(&sa), &sl) == 0)
            prebound_port = ntohs(sa.sin_port);
    }

    auto closePreboundFd = [&]{
        if (prebound_fd >= 0) { ::close(prebound_fd); prebound_fd = -1; }
    };

    // Step 1: submit NARROWBAND task via persistent channel.
    std::string req_json = buildTaskRequestJson(
        request_id, center_freq_hz.in(au::hertz), bw, sr, dur_ms,
        local_ip_, task_rank_, prebound_port);

    spdlog::info("IqCollector: submitting NARROWBAND {:.3f} MHz req={}",
                 center_freq_hz.in(au::mega(au::hertz)), request_id);

    const int timeout_ms = static_cast<int>(
        col_cfg_.analysis_timeout_ms.in(au::milli(au::seconds)));
    std::string resp_body = ch_->exchange(req_json, request_id, timeout_ms);
    if (resp_body.empty()) {
        spdlog::error("IqCollector: no response for req={}", request_id);
        closePreboundFd();
        return {};
    }

    // Step 2: parse TASK_ACCEPTED.
    std::string accepted_task_id;
    int udp_port = 0;
    try {
        auto j = json::parse(resp_body);
        if (j.value("status", "") != "ACCEPTED") {
            spdlog::warn("IqCollector: task rejected: {}",
                         j.value("reject_reason", "unknown"));
            closePreboundFd();
            return {};
        }
        accepted_task_id = j.value("task_id", "");
        if (j.contains("streams") && !j["streams"].empty()) {
            udp_port = j["streams"][0].value("udp_port", 0);
            last_sr_ = au::hertz(j["streams"][0].value("sample_rate_sps", sr));
        } else {
            last_sr_ = au::hertz(sr);
        }
        if (udp_port == 0) {
            spdlog::error("IqCollector: ACCEPTED missing streams[0].udp_port");
            closePreboundFd();
            return {};
        }
        spdlog::debug("IqCollector: accepted task_id={} port={} sr={:.0f}",
                      accepted_task_id, udp_port, last_sr_.in(au::hertz));
    } catch (const std::exception& ex) {
        spdlog::error("IqCollector: parse error: {}", ex.what());
        closePreboundFd();
        return {};
    }

    // Step 3: use pre-bound socket if controller honoured our port, else re-bind.
    int udp_fd;
    if (prebound_fd >= 0 && prebound_port > 0 && udp_port == prebound_port) {
        udp_fd = prebound_fd; prebound_fd = -1;
    } else {
        closePreboundFd();
        udp_fd = openBoundUdpSocket(udp_port);
        if (udp_fd < 0) {
            spdlog::error("IqCollector: bind port {} failed: {}",
                          udp_port, strerror(errno));
            return {};
        }
    }

    // Step 4: receive IQ.
    auto iq = receiveIq(udp_fd, col_cfg_.collect_samples, timeout_ms);
    ::close(udp_fd);
    spdlog::info("IqCollector: collected {} samples (sr={:.0f} Hz)",
                 (int)iq.size() / 2, last_sr_.in(au::hertz));

    // Step 5: fire-and-forget TASK_STOP.
    if (!accepted_task_id.empty())
        ch_->send(buildTaskStopJson(request_id + "_stop", accepted_task_id));

    return iq;
}



} // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
