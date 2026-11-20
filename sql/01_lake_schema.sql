-- Gold layer schema (PostgreSQL 16 + pgvector).
-- Runs automatically the first time the postgres container starts.

CREATE EXTENSION IF NOT EXISTS vector;

-- Where each batch came from and how far it got through the pipeline.
CREATE TABLE IF NOT EXISTS dim_batch (
    batch_id        TEXT PRIMARY KEY,
    role            TEXT NOT NULL CHECK (role IN ('reference', 'vendor')),
    source          TEXT NOT NULL,          -- e.g. fma_small, librispeech_dev_clean, vendor name
    licence         TEXT,
    landed_at       TIMESTAMPTZ,
    ingested_at     TIMESTAMPTZ,
    screened_at     TIMESTAMPTZ,
    file_count      INT,
    valid_count     INT,
    status          TEXT NOT NULL DEFAULT 'ingested'
);

-- Reference library: audio we know the owner/licence of. Incoming data is
-- screened against this.
CREATE TABLE IF NOT EXISTS ref_track (
    track_id        BIGSERIAL PRIMARY KEY,
    file_id         TEXT UNIQUE NOT NULL,   -- same id as Bronze/Silver
    batch_id        TEXT NOT NULL REFERENCES dim_batch(batch_id),
    rel_path        TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    title           TEXT,
    artist          TEXT,
    licence         TEXT,
    duration_s      REAL,
    n_hashes        INT,
    embedding       vector(72)
);

-- Landmark hashes (Shazam-style peak pairs). One reference file -> a few thousand rows.
CREATE TABLE IF NOT EXISTS ref_landmark (
    hash            INT NOT NULL,
    track_id        BIGINT NOT NULL,
    t_offset        INT NOT NULL            -- frame index of the anchor peak
);

-- One row per screened incoming file. This is the main fact table for reporting.
CREATE TABLE IF NOT EXISTS fact_screening (
    batch_id        TEXT NOT NULL REFERENCES dim_batch(batch_id),
    file_id         TEXT NOT NULL,
    rel_path        TEXT NOT NULL,
    sha256          TEXT,
    qc_status       TEXT NOT NULL,          -- pass / fail
    qc_reasons      TEXT,
    best_track_id   BIGINT,
    hash_votes      INT,
    n_query_hashes  INT,
    match_offset_s  REAL,
    embed_track_id  BIGINT,
    embed_similarity REAL,
    decision        TEXT NOT NULL,          -- CLEAR / REVIEW / FLAGGED / REJECTED_QC
    extract_ms      REAL,
    match_ms        REAL,
    run_id          TEXT,
    screened_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (batch_id, file_id)
);

CREATE TABLE IF NOT EXISTS etl_run_log (
    run_id          TEXT PRIMARY KEY,
    stage           TEXT NOT NULL,
    batch_id        TEXT,
    status          TEXT NOT NULL,
    rows_in         INT,
    rows_out        INT,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    error           TEXT,
    details         JSONB
);

-- Batch level compliance summary used by the legal/procurement report.
CREATE OR REPLACE VIEW vw_batch_compliance AS
SELECT
    b.batch_id,
    b.source                                         AS vendor,
    b.landed_at,
    b.screened_at,
    count(f.*)                                       AS files_screened,
    count(*) FILTER (WHERE f.decision = 'CLEAR')     AS clear,
    count(*) FILTER (WHERE f.decision = 'REVIEW')    AS review,
    count(*) FILTER (WHERE f.decision = 'FLAGGED')   AS flagged,
    count(*) FILTER (WHERE f.decision = 'REJECTED_QC') AS rejected_qc,
    round(100.0 * count(*) FILTER (WHERE f.decision = 'CLEAR') / nullif(count(f.*), 0), 2)
                                                     AS pct_clear,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY f.extract_ms + f.match_ms)
                                                     AS p95_latency_ms
FROM dim_batch b
LEFT JOIN fact_screening f USING (batch_id)
WHERE b.role = 'vendor'
GROUP BY b.batch_id, b.source, b.landed_at, b.screened_at;

-- Which reference works show up most often in vendor deliveries.
CREATE OR REPLACE VIEW vw_top_matched_works AS
SELECT r.track_id, r.title, r.artist, r.licence,
       count(*) AS times_matched,
       count(DISTINCT f.batch_id) AS batches
FROM fact_screening f
JOIN ref_track r ON r.track_id = f.best_track_id
WHERE f.decision IN ('FLAGGED', 'REVIEW')
GROUP BY r.track_id, r.title, r.artist, r.licence;
