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
#include "Config.hpp"
#include "AnalysisService.hpp"

#include <spdlog/spdlog.h>
#include <csignal>
#include <atomic>
#include <cstring>

static std::atomic<bool> g_shutdown{false};

static void signalHandler(int sig)
{
    spdlog::info("Signal {} received, shutting down…", sig);
    g_shutdown.store(true);
}

int main(int argc, char** argv)
{
    const char* log_level_env = std::getenv("SDR_LOG_LEVEL");
    spdlog::set_level(
        (log_level_env && std::string(log_level_env) == "debug")
            ? spdlog::level::debug : spdlog::level::info
    );
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

    if (argc < 2) {
        spdlog::error("Usage: sdr_analysis <config.xml>");
        return 1;
    }

    // Install signal handlers
    struct sigaction sa{};
    sa.sa_handler = signalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    ::sigaction(SIGINT,  &sa, nullptr);
    ::sigaction(SIGTERM, &sa, nullptr);

    analysis::AppConfig cfg;
    try {
        cfg = analysis::parseConfig(argv[1]);
    } catch (const std::exception& ex) {
        spdlog::error("Failed to parse config: {}", ex.what());
        return 1;
    }

    spdlog::info("Starting AnalysisApp v2.3.0 scanner_id={}",
                 cfg.scanner_id);

    analysis::AnalysisService svc(cfg);
    svc.start();

    // Block until signal
    sigset_t mask;
    sigemptyset(&mask);
    while (!g_shutdown.load()) {
        sigsuspend(&mask);
    }

    spdlog::info("Stopping…");
    svc.stop();

    return 0;
}
