-- DuckDB: clase_ternaria por continuidad mensual
CREATE OR REPLACE MACRO clase_ternaria(t) AS TABLE (
WITH base AS (
  SELECT
    *,
    -- Índice mensual consecutivo: yyyymm → (yyyy * 12 + mm)
    (CAST(foto_mes / 100 AS INTEGER) * 12 + (foto_mes % 100)) AS period_idx 
  FROM t
),
seq AS (
  SELECT
    *,
    LEAD(period_idx, 1) OVER (PARTITION BY numero_de_cliente ORDER BY period_idx) AS p1,
    LEAD(period_idx, 2) OVER (PARTITION BY numero_de_cliente ORDER BY period_idx) AS p2
  FROM base
),
bounds AS (
  SELECT MAX(period_idx) AS maxp FROM seq
)
SELECT
  * EXCLUDE (period_idx, p1, p2),
  CASE
    -- BAJA+1: falta el mes siguiente (o no vuelve a aparecer)
    WHEN period_idx < (SELECT maxp FROM bounds)
         AND (p1 IS NULL OR p1 > period_idx + 1)
    THEN 'BAJA+1'

    -- BAJA+2: aparece el mes siguiente, pero falta el segundo siguiente
    WHEN period_idx < (SELECT maxp FROM bounds) - 1
         AND p1 = period_idx + 1
         AND (p2 IS NULL OR p2 > period_idx + 2)
    THEN 'BAJA+2'

    -- CONTINUA: meses “antiguos” respecto del borde (al menos 2 meses antes del máximo)
    WHEN period_idx <= (SELECT maxp FROM bounds) - 2
    THEN 'CONTINUA'

    -- Borde (último/anteúltimo mes global): no etiquetamos
    ELSE NULL
  END AS clase_ternaria
FROM seq
);
