-- Schema sugerido para BigQuery (ejecutar en consola o bq CLI)
-- Dataset: sugarcane

CREATE TABLE IF NOT EXISTS `sugarcane.predictions` (
  predicted_at TIMESTAMP,
  class_name STRING,
  confidence FLOAT64,
  session_id STRING,
  demo_mode BOOL,
  model_id STRING,
  source STRING
);

-- Scheduled query diaria (18:00 America/Bogota) — resumen últimas 24 h
CREATE OR REPLACE TABLE `sugarcane.daily_summary` AS
SELECT
  DATE(predicted_at) AS day,
  class_name,
  COUNT(*) AS n,
  AVG(confidence) AS avg_conf
FROM `sugarcane.predictions`
WHERE predicted_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY day, class_name;
