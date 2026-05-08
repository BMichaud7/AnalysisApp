# ════════════════════════════════════════════════════════════════════════
#  Containerfile — AnalysisApp RF Signal Auto-Identification Service
#  Multi-stage: builder → runtime
#  Base: CentOS Stream 10 (el10) — compatible with RHEL 10 / k3s nodes
#
#  Build:
#    podman build -t sdr-analysis:2.3.0 .
#    docker build -t sdr-analysis:2.3.0 .
#
#  Run:
#    podman run --rm \
#      -e SDR_CONFIG_PATH=/etc/sdr-analysis/analysis.xml \
#      -v ./config/analysis.xml:/etc/sdr-analysis/analysis.xml:ro \
#      -p 20000-20099:20000-20099/udp \
#      sdr-analysis:2.3.0
# ════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM quay.io/centos/centos:stream10 AS builder

ENV LANG=C.UTF-8

# Enable EPEL (provides qpid-proton-cpp-devel, tinyxml2-devel, fmt-devel, fftw-devel)
RUN dnf install -y epel-release && dnf clean all

# Build tools + all library dependencies
RUN dnf install -y \
        cmake \
        make \
        gcc-c++ \
        pkg-config \
        git \
        ca-certificates \
        # Signal processing
        fftw-devel \
        # AMQP 1.0 messaging
        qpid-proton-cpp-devel \
        # XML config parsing
        tinyxml2-devel \
        # UUID generation
        libuuid-devel \
        # fmt (needed by spdlog when using pkg-config path)
        fmt-devel \
    && dnf clean all

# Clone SdrTaskApi (shared types/codec — sibling dependency)
RUN git clone --depth 1 https://github.com/BMichaud7/SdrTaskApi.git /workspace/SdrTaskApi

# Copy AnalysisApp source
WORKDIR /workspace/AnalysisApp
COPY CMakeLists.txt     .
COPY include/           include/
COPY src/               src/
COPY tests/             tests/
COPY config/            config/
COPY build.sh           .

# Build release — nlohmann/json and spdlog fetched by FetchContent if absent
RUN cmake -B build \
        -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/install \
        -DFETCHCONTENT_QUIET=OFF \
    && cmake --build build --parallel "$(nproc)" \
    && cmake --install build


# ── Stage 2: Runtime image ────────────────────────────────────────────────
FROM quay.io/centos/centos:stream10 AS runtime

ENV LANG=C.UTF-8

# Runtime shared libraries only (no -devel headers)
RUN dnf install -y epel-release && \
    dnf install -y \
        fftw \
        qpid-proton-cpp \
        tinyxml2 \
        libuuid \
        fmt \
        cyrus-sasl-plain \
    && dnf clean all

# tini is not in EPEL 10 — fetch static binary directly
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini-static-amd64 /usr/local/bin/tini
RUN chmod +x /usr/local/bin/tini

# Non-root service user
RUN groupadd -r sdranalysis && \
    useradd -r -g sdranalysis -s /sbin/nologin sdranalysis

# Config volume mount point (ConfigMap mounts here in Kubernetes)
RUN mkdir -p /etc/sdr-analysis && chown sdranalysis:sdranalysis /etc/sdr-analysis

# Copy built binary and default config
COPY --from=builder /install/bin/sdr_analysis /usr/local/bin/sdr_analysis
COPY --from=builder /workspace/AnalysisApp/config/analysis.xml /etc/sdr-analysis/analysis.xml

USER sdranalysis

# Override config path and log level via environment (ConfigMap / -e flag)
ENV SDR_CONFIG_PATH=/etc/sdr-analysis/analysis.xml
ENV SDR_LOG_LEVEL=info

EXPOSE 20000-20099/udp

ENTRYPOINT ["/usr/local/bin/tini", "--"]
CMD ["/usr/local/bin/sdr_analysis", "--config", "/etc/sdr-analysis/analysis.xml"]
