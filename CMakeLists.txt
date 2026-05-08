cmake_minimum_required(VERSION 3.20)
project(AnalysisApp VERSION 2.3.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── PkgConfig ─────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# ── SdrTaskApi sibling ────────────────────────────────────────────────────────
get_filename_component(PARENT "${CMAKE_SOURCE_DIR}" DIRECTORY)
if(EXISTS "${PARENT}/SdrTaskApi/CMakeLists.txt")
    message(STATUS "[AnalysisApp] SdrTaskApi: using sibling source at ${PARENT}/SdrTaskApi")
    add_subdirectory("${PARENT}/SdrTaskApi" "${CMAKE_BINARY_DIR}/SdrTaskApi")
else()
    find_package(sdr_task_api QUIET)
    if(NOT sdr_task_api_FOUND)
        message(FATAL_ERROR "[AnalysisApp] SdrTaskApi not found.\n  Clone as sibling:\n    git clone https://github.com/BMichaud7/SdrTaskApi.git ../SdrTaskApi")
    endif()
    message(STATUS "[AnalysisApp] SdrTaskApi: using installed package ${sdr_task_api_VERSION}")
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

# ── Library target ────────────────────────────────────────────────────────────
set(ANALYSIS_SOURCES
    src/Config.cpp
    src/FeatureExtractor.cpp
    src/ModulationClassifier.cpp
    src/ProtocolMapper.cpp
    src/AnalysisEngine.cpp
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
    PkgConfig::FFTW3F
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    PkgConfig::TINYXML2
    Threads::Threads
)
if(UUID_LIB)
    target_link_libraries(analysis_lib PUBLIC ${UUID_LIB})
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

# ── Tests ─────────────────────────────────────────────────────────────────────
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
)
target_link_libraries(analysis_tests PRIVATE
    analysis_lib
    GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(analysis_tests)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "┌─ AnalysisApp ${PROJECT_VERSION} ─────────────────────────────┐")
message(STATUS "│  C++ standard  : C++20                                        │")
message(STATUS "│  Build type    : ${CMAKE_BUILD_TYPE}                          │")
message(STATUS "│  FFTW3f        : ${FFTW3F_VERSION}                            │")
message(STATUS "│  qpid-proton   : ${PROTON_VERSION}                            │")
message(STATUS "└───────────────────────────────────────────────────────────────┘")
