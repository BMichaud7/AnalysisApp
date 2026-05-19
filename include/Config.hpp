#pragma once
#include "OnnxClassifier.hpp"
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
    // 65 536 samples at 2 MSPS = 32 ms — enough for cumulants, symbol-rate
    // estimation, and OFDM/FHSS detection down to ~4 800 baud.  Increase to
    // 131 072 if you need reliable FHSS detection at hop rates < 100 Hz.
    int         collect_samples          = 65'536;
    int         analysis_timeout_ms      = 5000;        // max wait for IQ
};

struct EngineConfig {
    double      fft_size             = 4096;
    double      guard_band_fraction  = 0.1;
    double      snr_threshold_db     = 5.0;
    int         rank                 = 1;
    OnnxConfig  onnx;               // optional ML fallback path
};

struct DbConfig {
    std::string host     = "localhost";
    int         port     = 5432;
    std::string dbname   = "sdr_scanner";
    std::string user     = "sdr";
    std::string password = "";
    bool        enabled  = false;   // set true when <database> block present in XML
};

struct AppConfig {
    AmqpConfig      amqp;
    CollectorConfig collector;
    EngineConfig    engine;
    DbConfig        db;
    std::string     scanner_id       = "analysis-0";
    std::string     streaming_ip     = "127.0.0.1";
};

AppConfig parseConfig(const std::string& xml_path);

} // namespace analysis
