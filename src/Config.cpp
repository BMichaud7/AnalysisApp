#include "Config.hpp"
#include <tinyxml2.h>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace analysis {

namespace {

// Helper: safely read a child element's text as a string, returning default if absent.
static std::string xmlText(const tinyxml2::XMLElement* parent,
                            const char* child_name,
                            const std::string& default_val = "")
{
    if (!parent) return default_val;
    const auto* el = parent->FirstChildElement(child_name);
    if (!el || !el->GetText()) return default_val;
    return std::string(el->GetText());
}

static double xmlDouble(const tinyxml2::XMLElement* parent,
                         const char* child_name,
                         double default_val = 0.0)
{
    std::string s = xmlText(parent, child_name, "");
    if (s.empty()) return default_val;
    try { return std::stod(s); }
    catch (...) { return default_val; }
}

static int xmlInt(const tinyxml2::XMLElement* parent,
                   const char* child_name,
                   int default_val = 0)
{
    std::string s = xmlText(parent, child_name, "");
    if (s.empty()) return default_val;
    try { return std::stoi(s); }
    catch (...) { return default_val; }
}

} // anonymous namespace

AppConfig parseConfig(const std::string& xml_path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(xml_path.c_str()) != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("Failed to load config file: " + xml_path +
                                 " — " + doc.ErrorStr());
    }

    const auto* root = doc.FirstChildElement("sdr_analysis");
    if (!root)
        throw std::runtime_error("Config XML missing root element <sdr_analysis>");

    AppConfig cfg;

    // Top-level fields
    cfg.scanner_id   = xmlText(root, "scanner_id",   cfg.scanner_id);
    cfg.streaming_ip = xmlText(root, "streaming_ip",  cfg.streaming_ip);

    // <amqp>
    const auto* amqp_el = root->FirstChildElement("amqp");
    if (amqp_el) {
        cfg.amqp.url                 = xmlText(amqp_el, "url",                  cfg.amqp.url);
        cfg.amqp.username            = xmlText(amqp_el, "username",             cfg.amqp.username);
        cfg.amqp.password            = xmlText(amqp_el, "password",             cfg.amqp.password);
        cfg.amqp.detections_topic    = xmlText(amqp_el, "detections_topic",     cfg.amqp.detections_topic);
        cfg.amqp.analysis_topic      = xmlText(amqp_el, "analysis_topic",       cfg.amqp.analysis_topic);
        cfg.amqp.task_request_queue  = xmlText(amqp_el, "task_request_queue",   cfg.amqp.task_request_queue);
        cfg.amqp.task_response_queue = xmlText(amqp_el, "task_response_queue",  cfg.amqp.task_response_queue);
    }

    // <collector>
    const auto* col_el = root->FirstChildElement("collector");
    if (col_el) {
        cfg.collector.analysis_sample_rate_sps =
            xmlDouble(col_el, "analysis_sample_rate_sps",
                      cfg.collector.analysis_sample_rate_sps);
        cfg.collector.collect_samples =
            xmlInt(col_el, "collect_samples", cfg.collector.collect_samples);
        cfg.collector.analysis_timeout_ms =
            xmlInt(col_el, "analysis_timeout_ms", cfg.collector.analysis_timeout_ms);
    }

    // <engine>
    const auto* eng_el = root->FirstChildElement("engine");
    if (eng_el) {
        cfg.engine.fft_size            = xmlDouble(eng_el, "fft_size",            cfg.engine.fft_size);
        cfg.engine.snr_threshold_db    = xmlDouble(eng_el, "snr_threshold_db",    cfg.engine.snr_threshold_db);
        cfg.engine.guard_band_fraction = xmlDouble(eng_el, "guard_band_fraction", cfg.engine.guard_band_fraction);
        cfg.engine.rank                = xmlInt   (eng_el, "rank",                cfg.engine.rank);

        // <engine><onnx> — optional ML fallback
        const auto* onnx_el = eng_el->FirstChildElement("onnx");
        if (onnx_el) {
            cfg.engine.onnx.model_path         = xmlText  (onnx_el, "model_path",  "");
            cfg.engine.onnx.classes_path       = xmlText  (onnx_el, "classes_path","");
            cfg.engine.onnx.use_gpu            = xmlText  (onnx_el, "use_gpu",     "true") != "false";
            cfg.engine.onnx.use_tensorrt       = xmlText  (onnx_el, "use_tensorrt","true") != "false";
            cfg.engine.onnx.tensorrt_fp16      = xmlText  (onnx_el, "tensorrt_fp16","true") != "false";
            cfg.engine.onnx.tensorrt_cache_mb  = xmlInt   (onnx_el, "tensorrt_cache_mb", 128);
            cfg.engine.onnx.input_len          = xmlInt   (onnx_el, "input_len",   1024);
            cfg.engine.onnx.max_batch          = xmlInt   (onnx_el, "max_batch",   8);
            cfg.engine.onnx.fallback_confidence =
                xmlDouble(onnx_el, "fallback_confidence_threshold", 0.60);
            cfg.engine.onnx.fallback_on_unknown =
                xmlText(onnx_el, "fallback_on_unknown", "true") != "false";
            cfg.engine.onnx.enabled = !cfg.engine.onnx.model_path.empty();
        }
    }

    // <database> — optional PostgreSQL result persistence
    const auto* db_el = root->FirstChildElement("database");
    if (db_el) {
        cfg.db.host     = xmlText(db_el, "host",     cfg.db.host);
        cfg.db.port     = xmlInt (db_el, "port",     cfg.db.port);
        cfg.db.dbname   = xmlText(db_el, "name",     cfg.db.dbname);
        cfg.db.user     = xmlText(db_el, "user",     cfg.db.user);
        cfg.db.password = xmlText(db_el, "password", cfg.db.password);
        cfg.db.enabled  = true;
    }

    spdlog::info("Config loaded from '{}': scanner={} amqp={} db={}",
                 xml_path, cfg.scanner_id, cfg.amqp.url,
                 cfg.db.enabled ? cfg.db.host + "/" + cfg.db.dbname : "disabled");

    return cfg;
}

} // namespace analysis
