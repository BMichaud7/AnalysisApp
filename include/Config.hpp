#pragma once
/**
 * @file Config.hpp
 * @brief Configuration structs for AnalysisApp, loaded from XML.
 *
 * parseConfig() reads analysis.xml and populates an AppConfig.
 * Sub-structs group related settings:
 * - AmqpConfig — broker connection and topic names
 * - CollectorConfig — IQ collection parameters
 * - EngineConfig — analysis pipeline tuning
 * - DbConfig — PostgreSQL connection
 * - OnnxConfig (in OnnxClassifier.hpp) — model path and GPU settings
 */
#include "OnnxClassifier.hpp"
#include <string>
#include <cstdint>

namespace analysis {

/// @brief AMQP broker connection and topic configuration.
struct AmqpConfig {
    std::string url                 = "amqp://localhost:5672"; ///< Broker URL.
    std::string username            = "sdr_ctrl";              ///< AMQP username.
    std::string password            = "password";              ///< AMQP password.
    std::string detections_topic    = "rf.detections";         ///< Topic to subscribe for detections.
    std::string analysis_topic      = "rf.analysis";           ///< Topic to publish analysis results.
    std::string task_request_queue  = "sdr.task.request";      ///< Queue for outbound task requests.
    std::string task_response_queue = "sdr.task.response";     ///< Queue for inbound task responses.
};

/// @brief IQ collection parameters for the slow path.
struct CollectorConfig {
    double analysis_sample_rate_sps = 2'000'000.0; ///< Sample rate for IQ collection (samples/s).
    /// Samples to collect per analysis.  65 536 samples at 2 MSPS = 32 ms — enough for
    /// cumulants, symbol-rate estimation, and OFDM/FHSS detection down to ~4 800 baud.
    int    collect_samples          = 65'536;
    int    analysis_timeout_ms      = 5000;         ///< Maximum wait for IQ task response (ms).
};

/// @brief Analysis pipeline configuration.
struct EngineConfig {
    double     fft_size            = 4096; ///< FFT size for feature extraction.
    double     guard_band_fraction = 0.1;  ///< Fraction of bandwidth treated as roll-off guard.
    double     snr_threshold_db    = 5.0;  ///< Minimum SNR to attempt classification (dB).
    int        rank                = 2;    ///< Task rank: 2=Ana (preempts Acq=1, preempted by DF=3).
    OnnxConfig onnx;                       ///< Optional ONNX classifier configuration.
};

/// @brief PostgreSQL persistence configuration.
struct DbConfig {
    std::string host     = "localhost";   ///< Database host.
    int         port     = 5432;          ///< Database port.
    std::string dbname   = "sdr_scanner"; ///< Database name.
    std::string user     = "sdr";         ///< Database user.
    std::string password = "";            ///< Database password.
    bool        enabled  = false;         ///< True when a <database> block is present in XML.
};

/// @brief Top-level configuration for AnalysisApp.
struct AppConfig {
    AmqpConfig      amqp;
    CollectorConfig collector;
    EngineConfig    engine;
    DbConfig        db;
    std::string     scanner_id   = "analysis-0"; ///< Identifier used in published analysis messages.
    std::string     streaming_ip = "127.0.0.1";  ///< Local IP for UDP IQ receive sockets.
};

/**
 * @brief Parse an XML config file into an AppConfig.
 * @param xml_path Path to the analysis.xml configuration file.
 * @return Populated AppConfig.
 * @throws std::runtime_error if the file cannot be opened or is malformed.
 */
AppConfig parseConfig(const std::string& xml_path);

} // namespace analysis
