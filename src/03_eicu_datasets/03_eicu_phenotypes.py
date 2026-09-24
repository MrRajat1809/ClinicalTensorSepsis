"""Annotate retrospective comorbidity and alternative-diagnosis flags."""

import importlib
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_VERSION = json.loads(Path(__file__).with_name("dataset_config.json").read_text(encoding="utf-8"))["dataset_version"]
io = importlib.import_module("01_eicu_base_cohort")
RAW_DIR = io.RAW
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu" / "work"
METRICS_DIR = BASE_DIR / "outputs" / "eicu"
CHARLSON_MAPPING_VERSION = "2.0.0"
CHARLSON_PATTERNS = {
    "mi": (
        "^(410|412)",
        "^(I21|I22|I252)",
    ),
    "chf": (
        "^(39891|40201|40211|40291|40401|40403|40411|40413|40491|40493|425[4-9]|428)",
        "^(I099|I110|I130|I132|I255|I420|I42[5-9]|I43|I50|P290)",
    ),
    "pvd": (
        "^(0930|4373|440|441|443[1-9]|4471|5571|5579|V434)",
        "^(I70|I71|I731|I738|I739|I771|I790|I792|K551|K558|K559|Z958|Z959)",
    ),
    "cevd": (
        "^(36234|43[0-8])",
        "^(G45|G46|H340|I6[0-9])",
    ),
    "dementia": (
        "^(290|2941|3312)",
        "^(F0[0-3]|F051|G30|G311)",
    ),
    "cpd": (
        "^(4168|4169|49[0-9]|50[0-5]|5064|5081|5088)",
        "^(I278|I279|J4[0-7]|J6[0-7]|J684|J701|J703)",
    ),
    "rheum": (
        "^(4465|710[0-4]|714[0-2]|7148|725)",
        "^(M05|M06|M315|M32|M33|M34|M351|M353|M360)",
    ),
    "pud": (
        "^(53[1-4])",
        "^(K2[5-8])",
    ),
    "mild_liver": (
        "^(07022|07023|07032|07033|07044|07054|0706|0709|570|571|5733|5734|5738|5739|V427)",
        "^(B18|K70[0-3]|K709|K71[3-5]|K717|K73|K74|K760|K76[2-4]|K768|K769|Z944)",
    ),
    "diab": (
        "^(250[0-3]|2508|2509)",
        "^(E10[01689]|E11[01689]|E12[01689]|E13[01689]|E14[01689])",
    ),
    "diab_comp": (
        "^(250[4-7])",
        "^(E10[2-57]|E11[2-57]|E12[2-57]|E13[2-57]|E14[2-57])",
    ),
    "paraplegia": (
        "^(3341|342|343|344[0-6]|3449)",
        "^(G041|G114|G80[12]|G81|G82|G83[0-4]|G839)",
    ),
    "renal": (
        "^(40301|40311|40391|40402|40403|40412|40413|40492|40493|582|583[0-7]|585|586|5880|V420|V451|V56)",
        "^(I120|I131|N03[2-7]|N05[2-7]|N18|N19|N250|Z49[0-2]|Z940|Z992)",
    ),
    "cancer": (
        "^(14[0-9]|15[0-9]|16[0-9]|17[0-2]|17[4-9][0-9]|18[0-9][0-9]|19[0-4][0-9]|195[0-8]|20[0-8]|2386)",
        "^(C0[0-9]|C1[0-9]|C2[0-6]|C3[0-4]|C3[7-9]|C4[013]|C4[5-9]|C5[0-8]|C6[0-9]|C7[0-6]|C8[1-58]|C9[0-7])",
    ),
    "sev_liver": (
        "^(456[0-2]|572[2-8])",
        "^(I850|I859|I864|I982|K704|K711|K721|K729|K76[5-7])",
    ),
    "mets": (
        "^(19[6-9])",
        "^(C7[7-9]|C80)",
    ),
    "hiv": (
        "^(04[2-4])",
        "^(B2[0-2]|B24)",
    ),
}
ALTERNATIVE_DIAGNOSES = {
    "ami": "(icd_version = 9 AND icd_code LIKE '410%') OR "
           "(icd_version = 10 AND (icd_code LIKE 'I21%' OR icd_code LIKE 'I22%'))",
    "pulmonary_embolism": "(icd_version = 9 AND icd_code LIKE '4151%') OR "
                         "(icd_version = 10 AND icd_code LIKE 'I26%')",
    "pancreatitis": "(icd_version = 9 AND icd_code LIKE '5770%') OR "
                    "(icd_version = 10 AND icd_code LIKE 'K85%')",
    "trauma_burns": "(icd_version = 9 AND "
                    "TRY_CAST(SUBSTRING(icd_code, 1, 3) AS INTEGER) BETWEEN 800 AND 959) OR "
                    "(icd_version = 10 AND (icd_code LIKE 'S%' OR "
                    "(icd_code LIKE 'T%' AND "
                    "TRY_CAST(SUBSTRING(icd_code, 2, 2) AS INTEGER) BETWEEN 0 AND 32)))",
}


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def build_phenotype_cohort(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, metrics_dir=METRICS_DIR):
    started = time.perf_counter()
    raw_dir, processed_dir, metrics_dir = map(Path, (raw_dir, processed_dir, metrics_dir))
    infection_file = processed_dir / "infection_cohort.parquet"
    diagnoses_file = io.source_path(raw_dir, "diagnosis")
    for path in (infection_file, diagnoses_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    component_sql = ",\n".join(
        f"CASE WHEN code_status = 'usable_format' AND ((icd_version = 9 AND "
        f"REGEXP_MATCHES(icd_code, {sql_string(pattern9)})) OR (icd_version = 10 AND "
        f"REGEXP_MATCHES(icd_code, {sql_string(pattern10)}))) THEN 1 ELSE 0 END AS cci_{name}"
        for name, (pattern9, pattern10) in CHARLSON_PATTERNS.items()
    )
    alternative_sql = ",\n".join(
        f"CASE WHEN code_status = 'usable_format' AND seq_num BETWEEN 1 AND 2 "
        f"AND ({predicate}) THEN TRUE ELSE FALSE END AS alternative_{name}"
        for name, predicate in ALTERNATIVE_DIAGNOSES.items()
    )
    flag_names = [f"cci_{name}" for name in CHARLSON_PATTERNS]
    flag_names += [f"alternative_{name}" for name in ALTERNATIVE_DIAGNOSES]
    aggregate_sql = ", ".join(f"MAX({name}) AS {name}" for name in flag_names)
    fill_sql = ", ".join(
        f"COALESCE(summary.{name}, {'FALSE' if name.startswith('alternative_') else '0'}) AS {name}"
        for name in flag_names
    )
    any_alternative_sql = " OR ".join(f"alternative_{name}" for name in ALTERNATIVE_DIAGNOSES)

    with duckdb.connect(":memory:") as connection:
        connection.execute(f"""
            CREATE TEMP TABLE infection AS
            SELECT * FROM read_parquet({sql_string(infection_file.as_posix())})
        """)
        incoming, subjects, admissions, stays = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT hadm_id), "
            "COUNT(DISTINCT stay_id) FROM infection"
        ).fetchone()
        if incoming != subjects or incoming != admissions or incoming != stays:
            raise ValueError("Infection input must have one non-null patient/admission/stay per row")
        connection.execute(f"""
            CREATE TEMP TABLE normalized_diagnoses AS
            WITH tokens AS (
                SELECT infection.stay_id, infection.subject_id, infection.hadm_id,
                    CASE UPPER(TRIM(diagnosispriority)) WHEN 'PRIMARY' THEN 1 WHEN 'MAJOR' THEN 2 ELSE 3 END AS seq_num,
                    TRIM(UNNEST(STRING_SPLIT(COALESCE(icd9code,''),','))) AS source_icd_code
                FROM ({io.csv_query(raw_dir,'diagnosis',dict(patientunitstayid='BIGINT',diagnosispriority='VARCHAR',icd9code='VARCHAR'))}) d
                JOIN infection ON infection.stay_id=d.patientunitstayid
            ) SELECT *, UPPER(REPLACE(source_icd_code,'.','')) AS icd_code,
                CASE WHEN REGEXP_MATCHES(source_icd_code,'^[0-9]') THEN 9
                    WHEN REGEXP_FULL_MATCH(UPPER(source_icd_code),'E[0-9]{{3}}([.][0-9])?') THEN 9
                    WHEN REGEXP_MATCHES(UPPER(source_icd_code),'^E[0-9]{{2}}[.]') THEN 10
                    WHEN REGEXP_MATCHES(UPPER(source_icd_code),'^[A-DF-UW-Z][0-9]') THEN 10
                    ELSE NULL END AS icd_version
            FROM tokens
        """)
        connection.execute(f"""
            CREATE TEMP TABLE diagnosis_evidence AS
            WITH classified AS (
                SELECT *, CASE
                    WHEN icd_version IS NULL OR icd_version NOT IN (9, 10) THEN 'unsupported_version'
                    WHEN icd_code IS NULL OR icd_code = '' THEN 'missing_code'
                    WHEN (icd_version = 9 AND REGEXP_FULL_MATCH(icd_code,
                          '([0-9]{{3,5}}|V[0-9]{{2,4}}|E[0-9]{{3,4}})'))
                      OR (icd_version = 10 AND REGEXP_FULL_MATCH(icd_code,
                          '[A-Z][0-9][A-Z0-9]{{1,5}}')) THEN 'usable_format'
                    ELSE 'invalid_format'
                END AS code_status
                FROM normalized_diagnoses
            )
            SELECT *, {component_sql}, {alternative_sql} FROM classified
        """)
        connection.execute(f"""
            CREATE TEMP TABLE phenotype AS
            WITH diagnosis_summary AS (
                SELECT stay_id, COUNT(*) AS diagnosis_rows,
                    COUNT(*) FILTER (WHERE code_status = 'usable_format') AS usable_diagnosis_rows,
                    COUNT(*) FILTER (WHERE code_status != 'usable_format') AS unusable_diagnosis_rows,
                    {aggregate_sql}
                FROM diagnosis_evidence GROUP BY stay_id
            ), joined AS (
                SELECT infection.*,
                    COALESCE(summary.diagnosis_rows, 0) AS diagnosis_rows,
                    COALESCE(summary.usable_diagnosis_rows, 0) AS usable_diagnosis_rows,
                    COALESCE(summary.unusable_diagnosis_rows, 0) AS unusable_diagnosis_rows,
                    COALESCE(summary.usable_diagnosis_rows, 0) > 0 AS has_usable_diagnoses,
                    {fill_sql}
                FROM infection LEFT JOIN diagnosis_summary summary USING (stay_id)
            ), scored AS (
                SELECT *,
                    cci_mi + cci_chf + cci_pvd + cci_cevd + cci_dementia + cci_cpd +
                    cci_rheum + cci_pud + GREATEST(cci_mild_liver, 3 * cci_sev_liver) +
                    GREATEST(cci_diab, 2 * cci_diab_comp) + 2 * cci_paraplegia +
                    2 * cci_renal + GREATEST(2 * cci_cancer, 6 * cci_mets) +
                    6 * cci_hiv AS charlson_comorbidity_index,
                    ({any_alternative_sql}) AS has_alternative_diagnosis,
                    EPOCH(icu_outtime - suspected_infection_time) / 3600.0 AS icu_hours_after_sit,
                    icu_outtime >= suspected_infection_time + INTERVAL 24 HOUR AS icu_ge_24h_after_sit,
                    TRUE AS cci_is_retrospective
                FROM joined
            )
            SELECT *, NOT has_alternative_diagnosis AND icu_ge_24h_after_sit
                AS restricted_phenotype_eligible
            FROM scored
        """)
        count, unique_stays, invalid_cci = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT stay_id), COUNT(*) FILTER "
            "(WHERE charlson_comorbidity_index IS NULL OR charlson_comorbidity_index NOT BETWEEN 0 AND 29) "
            "FROM phenotype"
        ).fetchone()
        if count != incoming or count != unique_stays or invalid_cci:
            raise ValueError("Phenotype row preservation, stay uniqueness, or CCI range check failed")
        summary_predicates = {
            "stays_with_alternative_diagnosis": "has_alternative_diagnosis",
            "stays_under_24h_after_sit": "icu_ge_24h_after_sit = FALSE",
            "stays_with_unknown_post_sit_duration": "icu_ge_24h_after_sit IS NULL",
            "stays_without_diagnosis_records": "diagnosis_rows = 0",
            "stays_without_usable_diagnoses": "NOT has_usable_diagnoses",
            "stays_with_unusable_diagnosis_rows": "unusable_diagnosis_rows > 0",
            "stays_with_zero_cci": "charlson_comorbidity_index = 0",
            "stays_with_cci_hierarchy_overlap": "(cci_diab = 1 AND cci_diab_comp = 1) OR "
                "(cci_mild_liver = 1 AND cci_sev_liver = 1) OR (cci_cancer = 1 AND cci_mets = 1)",
            "restricted_phenotype_stays": "restricted_phenotype_eligible",
        }
        summary = {
            name: connection.execute(f"SELECT COUNT(*) FROM phenotype WHERE {predicate}").fetchone()[0]
            for name, predicate in summary_predicates.items()
        }
        per_rule = [
            {"rule": name, "n_matched": connection.execute(
                f"SELECT COUNT(*) FROM phenotype WHERE alternative_{name}"
            ).fetchone()[0]}
            for name in ALTERNATIVE_DIAGNOSES
        ]
        code_status_counts = dict(connection.execute(
            "SELECT code_status, COUNT(*) FROM diagnosis_evidence GROUP BY code_status"
        ).fetchall())
        distribution = dict(connection.execute(
            "SELECT charlson_comorbidity_index, COUNT(*) FROM phenotype "
            "GROUP BY charlson_comorbidity_index ORDER BY charlson_comorbidity_index"
        ).fetchall())
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_file = processed_dir / "phenotypes.parquet"
        evidence_file = processed_dir / "diagnosis_evidence.parquet"
        connection.execute(
            f"COPY (SELECT * FROM phenotype ORDER BY stay_id) TO {sql_string(out_file.as_posix())} (FORMAT PARQUET)"
        )
        connection.execute(
            f"COPY (SELECT * FROM diagnosis_evidence ORDER BY stay_id, seq_num, icd_version, source_icd_code) "
            f"TO {sql_string(evidence_file.as_posix())} (FORMAT PARQUET)"
        )
    report = {
        "common_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE_DIR / "src/common").glob("*.json")},
        "dataset_config_sha256": hashlib.sha256(Path(__file__).with_name("dataset_config.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_version": DATASET_VERSION,
        "source_database": "eICU-CRD", "source_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__, "charlson_mapping_version": CHARLSON_MAPPING_VERSION,
        "policy": {
            "phenotype_exclusions": False, "minimum_post_sit_icu_duration_hours": None,
            "alternative_diagnosis_priorities": ["Primary", "Major"],
            "cci_source": "retrospective ICU diagnosis codes; ambiguous ICD9/10 tokens withheld", "cci_age_adjusted": False,
            "cci_hierarchies": ["diabetes", "liver disease", "malignancy"],
            "zero_cci": "no mapped condition in available codes; not proof of no comorbidity",
            "code_validation": "format only; not validation against an ICD dictionary",
            "restricted_phenotype": "no flagged alternative diagnosis and exact >=24h ICU after SIT; flag only",
            "candidate_handling": "All infection candidates remain available for Sepsis-3 adjudication",
        },
        "infection_cohort_in": incoming, "phenotype_cohort_out": count, "removed_total": 0,
        "rules_overlap": True, "per_rule": per_rule, "summary": summary,
        "code_status_counts": code_status_counts, "cci_distribution": distribution,
    }
    report_path = metrics_dir / "03_phenotypes.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[03] Complete v{DATASET_VERSION}; {time.perf_counter() - started:.1f}s; report: {metrics_dir}/03_phenotypes.json", flush=True)
    return report


if __name__ == "__main__":
    print(f"[03] Annotate retrospective comorbidity and alternative-diagnosis flags. v{DATASET_VERSION}", flush=True)
    build_phenotype_cohort()
