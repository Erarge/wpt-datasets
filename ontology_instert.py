import hashlib
from pathlib import Path

import pandas as pd
import requests


# --------------------------
# SPARQL config (adjust)
# --------------------------
SPARQL_UPDATE_URL = "http://localhost:8080/sparql"  # SPARQL Update endpoint
CHARGER_ID = "WC01"


PREFIXES = """
PREFIX eauto: <https://cloud.erarge.com.tr/ontologies/eauto#>
PREFIX sosa:  <http://www.w3.org/ns/sosa/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
"""

# --------------------------
# SPARQL helpers
# --------------------------
def sparql_update(query: str) -> None:
    # Send as raw text/plain (as requested)
    r = requests.post(
        SPARQL_UPDATE_URL,
        data=query.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(
            f"SPARQL update failed ({r.status_code})\n"
            f"Response:\n{r.text}\n"
            f"Query sent:\n{query}"
        )
    r.raise_for_status()


def iri_safe_session_key(source_file: str) -> str:
    return hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:12]


def esc_lit(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def fmt_double(v) -> str:
    # returns a SPARQL numeric literal (unquoted) or None if invalid
    if pd.isna(v):
        return None
    try:
        return str(float(v))
    except Exception:
        return None


def fmt_datetime_iso(v) -> str:
    """
    Ensure ISO-8601 dateTime for Virtuoso:
      - If already ISO string, keep
      - If pandas Timestamp, convert
    """
    if pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        # keep timezone info if present; otherwise treat as naive
        return v.isoformat()
    return str(v).strip()


# --------------------------
# Step 1: Session UPSERT
# --------------------------
def build_upsert_session_metadata(session_key: str, meta: dict, charger_id: str) -> str:
    # Note: numeric literals are unquoted here (cleaner SPARQL)
    soc = fmt_double(meta["soc"])
    distance = fmt_double(meta["distance"])
    duration = fmt_double(meta["duration"])
    if soc is None or distance is None or duration is None:
        raise ValueError(f"Bad session numeric meta: soc={meta['soc']} distance={meta['distance']} duration={meta['duration']}")

    return PREFIXES + f"""
INSERT {{
  eauto:wptSession_{session_key}
      a eauto:ChargingSession , sosa:FeatureOfInterest ;
      rdfs:label "WPT session {esc_lit(meta['source_file'])}"@en ;
      eauto:wpt_sourceFile "{esc_lit(meta['source_file'])}" ;
      eauto:wpt_location "{esc_lit(meta['location'])}" ;
      eauto:wpt_timeOfDay "{esc_lit(meta['time_of_day'])}" ;
      eauto:wpt_socPercent {soc} ;
      eauto:wpt_distanceMm {distance} ;
      eauto:wpt_coilType "{esc_lit(meta['coil_type'])}" ;
      eauto:wpt_durationMin {duration} ;
      eauto:usesCharger eauto:ic_{charger_id} .
}}
WHERE {{
  FILTER NOT EXISTS {{
    eauto:wptSession_{session_key} a eauto:ChargingSession .
  }}
}}
"""


# --------------------------
# CSV loading + preprocessing
# --------------------------
def iter_csv_files(folder_path: Path):
    for file in folder_path.rglob("*.csv"):
        if file.name.lower() == "session_parameters.csv":
            continue
        yield file


def load_one_csv(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["source_file"] = file.name
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep numeric values numeric (no '%' or 'mm').
    """
    df = df.copy()

    df["soc"] = (
        df["soc"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df["soc"] = pd.to_numeric(df["soc"], errors="coerce")

    df["distance"] = (
        df["distance"].astype(str)
        .str.replace("mm", "", regex=False)
        .str.strip()
    )
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # Ensure core numeric columns
    for c in ["v_pri_v", "a_pri_a", "v_sin_v", "a_sin_a", "v_sout_v", "a_sout_a", "t_coil_c", "t_ambiant_c"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived fields
    df["p_pri"] = df["v_pri_v"] * df["a_pri_a"]
    df["p_sin"] = df["v_sin_v"] * df["a_sin_a"]
    df["efficiency"] = df["p_sin"] / df["p_pri"]

    return df


def extract_session_metadata(df: pd.DataFrame) -> dict:
    r0 = df.iloc[0]
    return {
        "source_file": r0["source_file"],
        "location": r0["location"],
        "time_of_day": r0["time_of_day"],
        "soc": r0["soc"],
        "distance": r0["distance"],
        "coil_type": r0["coil_type"],
        "duration": r0["duration"],
    }


# --------------------------
# Step 2: Observation INSERTS (per row)
# --------------------------
# Column -> (sensor_suffix, observable_property_suffix, xsd_type)
FIELD_SPECS = {
    "v_pri_v":    ("s_v_pri_v",    "op_v_pri_v",    "xsd:double"),
    "a_pri_a":    ("s_a_pri_a",    "op_a_pri_a",    "xsd:double"),
    "v_sin_v":    ("s_v_sin_v",    "op_v_sin_v",    "xsd:double"),
    "a_sin_a":    ("s_a_sin_a",    "op_a_sin_a",    "xsd:double"),
    "v_sout_v":   ("s_v_sout_v",   "op_v_sout_v",   "xsd:double"),
    "a_sout_a":   ("s_a_sout_a",   "op_a_sout_a",   "xsd:double"),
    "t_coil_c":   ("s_t_coil_c",   "op_t_coil_c",   "xsd:double"),
    "t_ambiant_c":("s_t_ambiant_c","op_t_ambiant_c","xsd:double"),

    # derived metrics (assuming you created these sensors & properties)
    "p_pri":      ("s_p_pri",      "op_p_pri",      "xsd:double"),
    "p_sin":      ("s_p_sin",      "op_p_sin",      "xsd:double"),
    "efficiency": ("s_efficiency", "op_efficiency", "xsd:double"),
}


def build_insert_observations_for_row(
    session_key: str,
    charger_id: str,
    row: pd.Series,
) -> str:
    """
    One SPARQL update per CSV row inserting observations for all fields.
    Idempotent by using deterministic obs IRIs + FILTER NOT EXISTS per obs.
    """
    # Row ID for IRIs: prefer 'index' column; fallback to dataframe index
    row_id = row.get("index", None)
    if pd.isna(row_id):
        row_id = row.name
    row_id = str(int(row_id)) if str(row_id).isdigit() or isinstance(row_id, (int, float)) else str(row_id)

    ts = fmt_datetime_iso(row.get("timestamp"))
    if ts is None:
        raise ValueError(f"Missing timestamp for row {row_id}")

    blocks = []
    for col, (sensor_suffix, op_suffix, xsd_type) in FIELD_SPECS.items():
        val = row.get(col, None)
        num = fmt_double(val)
        if num is None:
            # Skip missing/invalid values rather than failing entire ingestion
            continue

        obs_iri = f"eauto:obs_{session_key}_{row_id}_{col}"
        sensor_iri = f"eauto:{sensor_suffix}_{charger_id}"
        op_iri = f"eauto:{op_suffix}"
        session_iri = f"eauto:wptSession_{session_key}"

        blocks.append(f"""
  {obs_iri} a sosa:Observation ;
    sosa:hasFeatureOfInterest {session_iri} ;
    sosa:madeBySensor {sensor_iri} ;
    sosa:observedProperty {op_iri} ;
    sosa:resultTime "{esc_lit(ts)}"^^xsd:dateTime ;
    sosa:hasSimpleResult {num} .
""")

    if not blocks:
        return None

    # Use WHERE + FILTER NOT EXISTS to avoid duplicates on re-run
    # We guard insertion by checking one representative observation existence per row,
    # OR you can keep it per-block; here we do per-block for correctness.
    where_filters = "\n".join(
        [f"  FILTER NOT EXISTS {{ eauto:obs_{session_key}_{row_id}_{col} a sosa:Observation . }}"
         for col in FIELD_SPECS.keys()]
    )

    return PREFIXES + f"""
INSERT {{
{''.join(blocks)}
}}
WHERE {{
{where_filters}
}}
"""


def insert_observations_for_file_df(df: pd.DataFrame, charger_id: str) -> None:
    source_file = str(df["source_file"].iloc[0])
    session_key = iri_safe_session_key(source_file)

    for _, row in df.iterrows():
        q = build_insert_observations_for_row(session_key, charger_id, row)
        if q is None:
            continue
        sparql_update(q)


# --------------------------
# Orchestrator: per-file session + observations
# --------------------------
def ingest_folder(input_dir: Path, charger_id: str) -> pd.DataFrame:
    dfs = []

    for file in iter_csv_files(input_dir):
        try:
            df = load_one_csv(file)
            df = preprocess_dataframe(df)

            # Step 1: UPSERT session metadata once per file
            meta = extract_session_metadata(df)
            session_key = iri_safe_session_key(str(meta["source_file"]))
            sparql_update(build_upsert_session_metadata(session_key, meta, charger_id))
            print(f"[OK] Session upserted: {meta['source_file']} -> {session_key}")

            # Step 2: Insert observations per row
            insert_observations_for_file_df(df, charger_id)
            print(f"[OK] Observations inserted for: {meta['source_file']} (rows={len(df)})")

            dfs.append(df)

        except Exception as e:
            print(f"[ERROR] Ingestion failed for {file}: {type(e).__name__}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    input_dir = Path("./data")
    df = ingest_folder(input_dir, CHARGER_ID)
    print("[INFO] Columns:", list(df.columns))
    print("[INFO] Total rows loaded:", len(df))


if __name__ == "__main__":
    main()
