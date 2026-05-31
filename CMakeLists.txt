cmake_minimum_required(VERSION 3.20)
project(AnalysisApp VERSION 2.5.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── PkgConfig ─────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# ── Au (Aurora units library) ─────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(Au
    GIT_REPOSITORY https://github.com/aurora-opensource/au.git
    GIT_TAG        0.5.1
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(Au)

# ── SdrSdk (provides sdr_task_api types via its subdirectory) ─────────────────
# Expected as a sibling directory: ../SdrSdk  (same parent as this repo).
# SdrSdk itself expects ../SdrTaskApi as its own sibling.
get_filename_component(PARENT "${CMAKE_SOURCE_DIR}" DIRECTORY)
if(EXISTS "${PARENT}/SdrSdk/CMakeLists.txt")
    message(STATUS "[AnalysisApp] using sibling SdrSdk at ${PARENT}/SdrSdk")
    set(SDRSDK_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDRSDK_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    add_subdirectory("${PARENT}/SdrSdk" "${CMAKE_BINARY_DIR}/SdrSdk")
elseif(EXISTS "${PARENT}/SdrTaskApi/CMakeLists.txt")
    message(STATUS "[AnalysisApp] using sibling SdrTaskApi at ${PARENT}/SdrTaskApi")
    add_subdirectory("${PARENT}/SdrTaskApi" "${CMAKE_BINARY_DIR}/SdrTaskApi")
else()
    find_package(sdr_task_api QUIET)
    if(NOT sdr_task_api_FOUND)
        message(FATAL_ERROR "[AnalysisApp] SdrSdk not found.\n  Clone as sibling directories:\n    git clone https://github.com/BMichaud7/SdrSdk.git ../SdrSdk\n    git clone https://github.com/BMichaud7/SdrTaskApi.git ../SdrTaskApi")
    endif()
    message(STATUS "[AnalysisApp] using installed sdr_task_api ${sdr_task_api_VERSION}")
endif()

# ── FFTW3 (single precision) ──────────────────────────────────────────────────
pkg_check_modules(FFTW3F QUIET IMPORTED_TARGET fftw3f)
if(NOT FFTW3F_FOUND)
    message(FATAL_ERROR "[AnalysisApp] fftw3f not found.\n  Ubuntu: apt-get install libfftw3-dev\n  CentOS: dnf install fftw-devel")
endif()

# ── qpid-proton C++ ───────────────────────────────────────────────────────────
find_package(ProtonCpp QUIET)
if(NOT ProtonCpp_FOUND)
    pkg_check_modules(PROTON QUIET IMPORTED_TARGET libqpid-proton-cpp)
    if(NOT PROTON_FOUND)
        message(FATAL_ERROR "[AnalysisApp] qpid-proton-cpp not found.\n  Ubuntu: apt-get install libqpid-proton-cpp12-dev\n  CentOS: dnf install qpid-proton-cpp-devel")
    endif()
    message(STATUS "[AnalysisApp] qpid-proton-cpp: found via pkg-config ${PROTON_VERSION}")
endif()

# ── nlohmann/json ─────────────────────────────────────────────────────────────
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "[AnalysisApp] nlohmann_json not on system; fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ── spdlog ────────────────────────────────────────────────────────────────────
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    pkg_check_modules(SPDLOG QUIET IMPORTED_TARGET spdlog)
    if(SPDLOG_FOUND)
        add_library(spdlog::spdlog INTERFACE IMPORTED)
        target_include_directories(spdlog::spdlog INTERFACE ${SPDLOG_INCLUDE_DIRS})
        target_link_libraries(spdlog::spdlog INTERFACE ${SPDLOG_LIBRARIES})
        target_compile_definitions(spdlog::spdlog INTERFACE SPDLOG_FMT_EXTERNAL)
        message(STATUS "[AnalysisApp] spdlog: found via pkg-config ${SPDLOG_VERSION}")
    else()
        message(STATUS "[AnalysisApp] spdlog not found; fetching from GitHub")
        include(FetchContent)
        FetchContent_Declare(spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG        v1.13.0)
        FetchContent_MakeAvailable(spdlog)
    endif()
endif()

# ── tinyxml2 ──────────────────────────────────────────────────────────────────
pkg_check_modules(TINYXML2 QUIET IMPORTED_TARGET tinyxml2)
if(NOT TINYXML2_FOUND)
    message(FATAL_ERROR "[AnalysisApp] tinyxml2 not found.\n  Ubuntu: apt-get install libtinyxml2-dev\n  CentOS: dnf install tinyxml2-devel")
endif()

# ── libuuid (optional — only needed if using uuid/uuid.h directly) ────────────
# The service uses its own UUIDv4 generator so this is informational only.
find_library(UUID_LIB uuid)
find_path(UUID_INCLUDE_DIR uuid/uuid.h)
if(UUID_LIB AND UUID_INCLUDE_DIR)
    message(STATUS "[AnalysisApp] libuuid: found (${UUID_LIB})")
else()
    message(STATUS "[AnalysisApp] libuuid headers not found — using built-in UUIDv4 generator")
    set(UUID_LIB "")
    set(UUID_INCLUDE_DIR "")
endif()

# ── ONNX Runtime (optional — enables ML-based AMR via OnnxClassifier) ────────
# Download from https://github.com/microsoft/onnxruntime/releases
# and set ONNXRUNTIME_ROOT to the extracted directory, or install system-wide.
#
#   CentOS/RHEL:
#     curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/\
#          onnxruntime-linux-x64-1.17.3.tgz | tar xz -C /opt
#     cmake ... -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.17.3
#
#   CUDA variant:
#     cmake ... -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-gpu-1.17.3
option(WITH_ONNX "Enable ONNX Runtime for ML-based classification" OFF)
if(WITH_ONNX)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to extracted onnxruntime directory")
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" /usr/local/lib /usr/lib)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" /usr/local/include /usr/include)
    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE_DIR)
        message(STATUS "[AnalysisApp] ONNX Runtime: found (${ONNXRUNTIME_LIB})")
        add_library(onnxruntime::onnxruntime INTERFACE IMPORTED)
        target_include_directories(onnxruntime::onnxruntime INTERFACE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(onnxruntime::onnxruntime INTERFACE ${ONNXRUNTIME_LIB})
    else()
        message(FATAL_ERROR "[AnalysisApp] ONNX Runtime not found.\n"
            "  Download: https://github.com/microsoft/onnxruntime/releases\n"
            "  Then: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ...")
    endif()
endif()

# ── libpqxx (optional — persists analysis_results to PostgreSQL) ──────────────
# Falls back to building libpqxx 7.9 from source when the system package is
# unavailable (e.g. CentOS Stream 10 / RHEL 10 — EPEL 10 has no libpqxx-devel).
# Prerequisite in all cases: libpq-devel (PostgreSQL C client).
option(WITH_DB "Enable PostgreSQL result persistence via libpqxx" OFF)
if(WITH_DB)
    pkg_check_modules(PQXX QUIET IMPORTED_TARGET libpqxx)
    if(PQXX_FOUND)
        message(STATUS "[AnalysisApp] libpqxx: system package ${PQXX_VERSION}")
    else()
        find_library(PQXX_LIB pqxx)
        find_path(PQXX_INCLUDE_DIR pqxx/pqxx)
        if(PQXX_LIB AND PQXX_INCLUDE_DIR)
            add_library(PkgConfig::PQXX INTERFACE IMPORTED)
            target_include_directories(PkgConfig::PQXX INTERFACE ${PQXX_INCLUDE_DIR})
            target_link_libraries(PkgConfig::PQXX INTERFACE ${PQXX_LIB})
            set(PQXX_FOUND TRUE)
            message(STATUS "[AnalysisApp] libpqxx: found without pkg-config")
        else()
            # Build libpqxx from source; requires libpq-devel
            find_library(LIBPQ_LIB pq)
            if(NOT LIBPQ_LIB)
                message(FATAL_ERROR "[AnalysisApp] libpq not found — required to build libpqxx.\n"
                    "  Ubuntu: apt-get install libpq-dev\n"
                    "  CentOS/RHEL: dnf install libpq-devel")
            endif()
            message(STATUS "[AnalysisApp] libpqxx not found — fetching 7.9.2 from GitHub")
            include(FetchContent)
            FetchContent_Declare(pqxx
                GIT_REPOSITORY https://github.com/jtv/libpqxx.git
                GIT_TAG        7.9.2
                GIT_SHALLOW    TRUE)
            set(PQXX_LIBRARIES_ONLY ON  CACHE BOOL "" FORCE)
            set(BUILD_DOC          OFF CACHE BOOL "" FORCE)
            set(BUILD_TESTING      OFF CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(pqxx)
            add_library(PkgConfig::PQXX ALIAS pqxx)
            set(PQXX_FOUND TRUE)
        endif()
    endif()
endif()

# ── Library target ────────────────────────────────────────────────────────────
set(ANALYSIS_SOURCES
    src/Config.cpp
    src/FeatureExtractor.cpp
    src/ModulationClassifier.cpp
    src/ProtocolMapper.cpp
    src/AnalysisEngine.cpp
    src/OnnxClassifier.cpp
    src/IqCollector.cpp
    src/AnalysisService.cpp
)

add_library(analysis_lib STATIC ${ANALYSIS_SOURCES})
target_include_directories(analysis_lib
    PUBLIC
        include
)
if(UUID_INCLUDE_DIR)
    target_include_directories(analysis_lib PRIVATE ${UUID_INCLUDE_DIR})
endif()

target_link_libraries(analysis_lib PUBLIC
    sdr_task_api
    Au::au
    PkgConfig::FFTW3F
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    PkgConfig::TINYXML2
    Threads::Threads
)
if(UUID_LIB)
    target_link_libraries(analysis_lib PUBLIC ${UUID_LIB})
endif()
if(WITH_ONNX)
    target_link_libraries(analysis_lib PUBLIC onnxruntime::onnxruntime)
    target_compile_definitions(analysis_lib PUBLIC ANALYSIS_WITH_ONNX=1)
endif()
if(WITH_DB)
    target_link_libraries(analysis_lib PUBLIC PkgConfig::PQXX)
    target_compile_definitions(analysis_lib PUBLIC ANALYSIS_WITH_DB=1)
endif()

# Proton linkage (cmake package path vs pkg-config path)
if(ProtonCpp_FOUND)
    target_link_libraries(analysis_lib PUBLIC Proton::cpp)
else()
    target_link_libraries(analysis_lib PUBLIC PkgConfig::PROTON)
endif()

target_compile_options(analysis_lib PRIVATE -O2 -march=native)
target_compile_definitions(analysis_lib PRIVATE
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
)

# ── Main binary ───────────────────────────────────────────────────────────────
add_executable(sdr_analysis src/main.cpp)
target_link_libraries(sdr_analysis PRIVATE analysis_lib)

# ── Install ───────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS sdr_analysis
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES config/analysis.xml
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/sdr-analysis)

# ── Tests ─────────────────────────────────────────────────────────────────────
include(CTest)
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    add_executable(analysis_tests
        tests/unit/test_feature_extractor.cpp
        tests/unit/test_modulation_classifier.cpp
        tests/unit/test_protocol_mapper.cpp
        tests/unit/test_analysis_engine.cpp
    )
    target_link_libraries(analysis_tests PRIVATE
        analysis_lib
        GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(analysis_tests)
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "┌─ AnalysisApp ${PROJECT_VERSION} ─────────────────────────────┐")
message(STATUS "│  C++ standard  : C++20                                        │")
message(STATUS "│  Build type    : ${CMAKE_BUILD_TYPE}                          │")
message(STATUS "│  FFTW3f        : ${FFTW3F_VERSION}                            │")
message(STATUS "│  qpid-proton   : ${PROTON_VERSION}                            │")
message(STATUS "│  ONNX Runtime  : ${WITH_ONNX}                                 │")
message(STATUS "│  PostgreSQL DB : ${WITH_DB}                                   │")
message(STATUS "└───────────────────────────────────────────────────────────────┘")
