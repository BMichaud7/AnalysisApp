#pragma once
#include <string>
#include <cstdint>

namespace analysis {

struct AmqpConfig {
    std::string url                  = "amqp://localhost:5672";
    std::string username             = "sdr_ctrl";
    std::string password             = "password";
    std::string detections_topic     = "rf.detections";
    std::string analysis_topic       = "rf.analysis";
    std::string task_request_queue   = "sdr.task.request";
    std::string task_response_queue  = "sdr.task.response";
};

struct CollectorConfig {
    double      analysis_sample_rate_sps = 2'000'000.0; // SR used for IQ collection
    int         collect_samples          = 1'000'000;   // samples per analysis window
    int         analysis_timeout_ms      = 5000;        // max wait for IQ
};

struct EngineConfig {
    double      fft_size             = 4096;
    double      guard_band_fraction  = 0.1;    // fraction of BW to ignore at edges
    double      snr_threshold_db     = 5.0;    // min SNR to attempt classification
    int         rank                 = 1;      // rank for IQ collection tasks
};

struct AppConfig {
    AmqpConfig      amqp;
    CollectorConfig collector;
    EngineConfig    engine;
    std::string     scanner_id       = "analysis-0";
    std::string     streaming_ip     = "127.0.0.1";
};

AppConfig parseConfig(const std::string& xml_path);

} // namespace analysis
