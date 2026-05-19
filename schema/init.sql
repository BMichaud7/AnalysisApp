-- SDR Analysis — PostgreSQL schema
-- Run once before starting sdr_analysis:
--   psql -U sdr -d sdr_scanner -f schema/init.sql
--
-- This schema stores signal classification results produced by AnalysisApp.
-- Raw detections are stored by AcquisitionApp in a separate 'detections' table
-- (see AcquisitionApp/schema/init.sql).

CREATE TABLE IF NOT EXISTS analysis_results (
    id                  BIGSERIAL       PRIMARY KEY,
    analyzed_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),
    detection_id        TEXT            NOT NULL,  -- UUID from AnalysisResult
    scanner_id          TEXT            NOT NULL,
    center_freq_hz      BIGINT          NOT NULL,
    bandwidth_hz        INTEGER,
    snr_db              REAL,

    -- Classification outputs
    classified          BOOLEAN         NOT NULL DEFAULT false,
    analog_modulation   TEXT,           -- e.g. 'FM_WB', 'AM_DSB_LC', 'CW'
    digital_modulation  TEXT,           -- e.g. 'BPSK', 'QPSK', 'OFDM'
    symbol_rate_sps     DOUBLE PRECISION,
    bit_rate_bps        DOUBLE PRECISION,
    is_ofdm             BOOLEAN         DEFAULT false,
    is_fhss             BOOLEAN         DEFAULT false,
    is_burst            BOOLEAN         DEFAULT false,

    -- Top protocol hypothesis
    hypothesis_system   TEXT,           -- e.g. 'GSM', 'FM Broadcast', 'AIS'
    hypothesis_category TEXT,           -- e.g. 'Cellular', 'Broadcast', 'Marine'
    hypothesis_conf     REAL,           -- 0.0–1.0

    -- Classification path metadata
    rule_confidence     REAL,           -- certainty from rule-based classifier
    onnx_used           BOOLEAN         DEFAULT false,
    onnx_confidence     REAL,           -- softmax probability (if ONNX ran)
    fast_path           BOOLEAN         DEFAULT false,  -- true = classified from IQ snapshot

    -- Reject reason when classified=false
    reject_reason       TEXT
);

CREATE INDEX IF NOT EXISTS idx_analysis_time  ON analysis_results (analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_freq  ON analysis_results (center_freq_hz);
CREATE INDEX IF NOT EXISTS idx_analysis_mod   ON analysis_results (digital_modulation, analog_modulation);
CREATE INDEX IF NOT EXISTS idx_analysis_hyp   ON analysis_results (hypothesis_system);

-- ── Useful views ──────────────────────────────────────────────────────────────

-- Recent classifications with combined modulation column
CREATE OR REPLACE VIEW recent_classifications AS
SELECT
    analyzed_at,
    round(center_freq_hz / 1e6, 3)         AS freq_mhz,
    round(bandwidth_hz   / 1e3, 1)         AS bw_khz,
    round(snr_db::numeric, 1)              AS snr_db,
    coalesce(digital_modulation,
             analog_modulation, 'UNKNOWN') AS modulation,
    hypothesis_system,
    round(hypothesis_conf::numeric, 2)     AS hyp_conf,
    round(rule_confidence::numeric, 2)     AS rule_conf,
    onnx_used,
    fast_path,
    scanner_id
FROM analysis_results
WHERE analyzed_at > now() - interval '60 seconds'
  AND classified = true
ORDER BY analyzed_at DESC;

-- Per-frequency activity: most-seen modulations per band
CREATE OR REPLACE VIEW freq_classification_summary AS
SELECT
    round(center_freq_hz / 1e6, 2)         AS freq_mhz,
    count(*)                                AS classification_count,
    mode() WITHIN GROUP (ORDER BY
        coalesce(digital_modulation, analog_modulation)) AS dominant_modulation,
    mode() WITHIN GROUP (ORDER BY hypothesis_system)     AS dominant_system,
    round(avg(snr_db)::numeric, 1)          AS avg_snr_db,
    max(analyzed_at)                        AS last_seen
FROM analysis_results
WHERE classified = true
GROUP BY round(center_freq_hz / 1e6, 2)
ORDER BY classification_count DESC;

-- Fast-path vs slow-path breakdown (useful for tuning FAST_PATH_THRESHOLD)
CREATE OR REPLACE VIEW classification_path_stats AS
SELECT
    fast_path,
    count(*)                               AS total,
    round(avg(onnx_confidence)::numeric, 3) AS avg_onnx_conf,
    round(avg(rule_confidence)::numeric, 3) AS avg_rule_conf,
    count(*) FILTER (WHERE classified)    AS classified_count,
    round(100.0 * count(*) FILTER (WHERE classified)
          / nullif(count(*), 0), 1)       AS classified_pct
FROM analysis_results
GROUP BY fast_path;
