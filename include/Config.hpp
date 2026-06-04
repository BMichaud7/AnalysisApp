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
#pragma once
/**
 * @file Config.hpp
 * @note Frequency/rate parameters use au::QuantityD<au::Hertz>;
 *       duration parameters use au::QuantityD<au::Seconds>.
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
#include <au/units/hertz.hh>
#include <au/units/seconds.hh>
#include <au/prefix.hh>
#include <string>
#include <cstdint>

namespace analysis {

/**
 * @brief AMQP broker connection and topic configuration.
 *
 * Loaded from the @c \<amqp\> block of analysis.xml.  All fields have safe
 * defaults so individual elements can be omitted.
 */
struct AmqpConfig {
    std::string url                  = "amqp://localhost:5672"; ///< Broker URL. Supports amqp:// and amqps://.
    std::string username             = "sdr_ctrl";              ///< AMQP username (empty = anonymous).
    std::string password             = "password";              ///< AMQP password.
    std::string detections_topic     = "rf.detections";         ///< Topic to subscribe for RF_DETECTION messages from AcquisitionApp.
    std::string analysis_topic       = "rf.analysis";           ///< Topic to publish ANALYSIS_RESULT messages (consumed by DemodApp).
    std::string task_request_queue   = "sdr.task.request";      ///< Queue for outbound IQ task requests to SdrResourceManager.
    std::string task_response_queue  = "sdr.task.response";     ///< Queue for inbound task ACCEPTED/REJECTED responses.
    std::string demod_commands_queue = "sdr.demod.commands";    ///< Subscribe: on-demand REQUEST_DEMOD commands from user apps.
    std::string demod_request_queue  = "rf.demod.request";      ///< Publish: DEMOD_REQUEST messages to DemodApp.
};

/**
 * @brief IQ collection parameters for the slow-path analysis.
 *
 * The slow path is triggered when the fast-path ONNX snapshot confidence falls
 * below the configured threshold.  A fresh NARROWBAND IQ capture is requested
 * from SdrResourceManager for deeper feature extraction.
 */
struct CollectorConfig {
    /// Sample rate for slow-path IQ collection (samples/s).
    /// 2 MSPS provides sufficient bandwidth for cumulant estimation,
    /// symbol-rate detection, and OFDM/FHSS identification ≥ 4 800 baud.
    au::QuantityD<au::Hertz>   analysis_sample_rate_sps{au::hertz(2'000'000.0)};

    /// IQ samples to collect per slow-path analysis window.
    /// 65 536 samples @ 2 MSPS = 32 ms — enough for 4th-order cumulants,
    /// FM deviation measurement, and cyclic-prefix OFDM detection.
    int                        collect_samples{65'536};

    /// Maximum time (ms) to wait for the IQ task ACCEPTED response from
    /// SdrResourceManager.  Increase if the SDR is heavily contended.
    au::QuantityD<au::Seconds> analysis_timeout_ms{au::milli(au::seconds)(5000)};
};

/**
 * @brief Analysis engine configuration — feature extraction and ML classifier.
 *
 * Loaded from the @c \<engine\> block of analysis.xml.
 * Controls both the rule-based feature engine and the two ONNX model paths.
 */
struct EngineConfig {
    double     fft_size            = 4096; ///< FFT size for spectral features (power of 2, ≥ 64).
    double     guard_band_fraction = 0.1;  ///< Roll-off guard fraction — bins at band edges are excluded.
    double     snr_threshold_db    = 5.0;  ///< Minimum SNR (dB) to attempt ONNX classification. Signals below this use rule engine only.
    int        rank                = 2;    ///< IQ-fetch priority: 1=Acq (lowest), 2=Analysis, 3=DF (highest).
    OnnxConfig onnx;                       ///< Primary (high-SNR) ONNX classifier. 47-class RadioResNet, val_acc=0.748.
    OnnxConfig onnx_low_snr;               ///< Optional low-SNR classifier (DAE denoised). Used when SNR < snr_model_split_db.
    double     snr_model_split_db  = 8.0;  ///< SNR (dB) below which onnx_low_snr is preferred over onnx.
};

/**
 * @brief PostgreSQL persistence configuration.
 *
 * Loaded from the optional @c \<database\> block of analysis.xml.
 * When absent, @c enabled remains false and no database connection is opened.
 * Classification results are written to the @c signals table (same schema as AcquisitionApp).
 */
struct DbConfig {
    std::string host     = "localhost";   ///< Database host.
    int         port     = 5432;          ///< Database port.
    std::string dbname   = "sdr_scanner"; ///< Database name.
    std::string user     = "sdr";         ///< Database user.
    std::string password = "";            ///< Database password.
    bool        enabled  = false;         ///< Automatically set to true when a @c \<database\> block is present.
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

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
