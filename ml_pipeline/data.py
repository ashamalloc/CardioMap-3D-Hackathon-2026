from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# ============================================================
# Configuration
# ============================================================

TARGETS = ["CAD", "LAD", "LCX", "RCA"]

# In the Extension of Z-Alizadeh Sani dataset:
# - CAD/overall disease status is represented by Cath
# - LAD, LCX and RCA are vessel-level targets
#
# All of these must be excluded from model INPUT features.
LEAKAGE_COLUMNS = {
    "cad",
    "lad",
    "lcx",
    "rca",
    "cath",
}

IDENTIFIER_PATTERNS = (
    "id",
    "patientid",
    "recordid",
    "name",
    "ssn",
    "socialsecurity",
)

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "extention_of_z_alizadeh_sani_dataset.xlsx"
)


# ============================================================
# Column utilities
# ============================================================

def normalize_column(name: object) -> str:
    """Normalize dataframe column names."""
    text = str(name).strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def canonical(name: object) -> str:
    """Return a simplified lowercase version of a column name."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_column(
    columns: Iterable[object],
    aliases: set[str],
) -> str | None:
    """Find a column using canonicalized aliases."""
    lookup = {
        canonical(column): str(column)
        for column in columns
    }

    for alias in aliases:
        if alias in lookup:
            return lookup[alias]

    return None


# ============================================================
# Dataset loading
# ============================================================

def load_local_excel(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> pd.DataFrame:
    """
    Load the supplied Extension of Z-Alizadeh Sani Excel file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n\n"
            "Expected dataset:\n"
            f"{DEFAULT_DATASET_PATH}"
        )

    if not path.is_file():
        raise ValueError(
            f"Dataset path is not a file:\n{path}"
        )

    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(
            f"Unsupported file format: {path.suffix}"
        )

    print(f"Loading dataset: {path}")

    workbook = pd.ExcelFile(path)

    usable_frames: list[pd.DataFrame] = []

    for sheet in workbook.sheet_names:
        frame = pd.read_excel(
            path,
            sheet_name=sheet,
        )

        if frame.shape[0] > 0 and frame.shape[1] > 5:
            usable_frames.append(frame)

    if not usable_frames:
        raise ValueError(
            "No usable tabular sheet was found in the Excel workbook."
        )

    df = usable_frames[0].copy()

    print(
        f"Loaded {df.shape[0]} rows × {df.shape[1]} columns"
    )

    return df


def fetch_uci() -> pd.DataFrame:
    """
    Compatibility wrapper.

    The project uses the locally downloaded Excel dataset.
    No ucimlrepo API call is made.
    """
    return load_local_excel(DEFAULT_DATASET_PATH)


# ============================================================
# Target resolution
# ============================================================

def resolve_targets(
    df: pd.DataFrame,
) -> dict[str, str]:
    """
    Resolve the four required prediction targets.

    IMPORTANT:
    The supplied 59-column Extension dataset does not contain
    a literal 'CAD' column.

    Instead:
        Cath -> overall CAD classification

    Therefore:
        CAD target = Cath
        LAD target = LAD
        LCX target = LCX
        RCA target = RCA
    """

    mapping: dict[str, str] = {}

    # --------------------------------------------------------
    # Overall CAD
    # --------------------------------------------------------
    cad_column = _find_column(
        df.columns,
        {"cad"},
    )

    cath_column = _find_column(
        df.columns,
        {"cath"},
    )

    if cad_column is not None:
        mapping["CAD"] = cad_column
    elif cath_column is not None:
        # This is the expected structure of the supplied dataset.
        mapping["CAD"] = cath_column

    # --------------------------------------------------------
    # Vessel targets
    # --------------------------------------------------------
    for target in ["LAD", "LCX", "RCA"]:
        column = _find_column(
            df.columns,
            {canonical(target)},
        )

        if column is not None:
            mapping[target] = column

    missing = [
        target
        for target in TARGETS
        if target not in mapping
    ]

    if missing:
        raise ValueError(
            "Could not identify required target columns: "
            + ", ".join(missing)
            + "\n\nAvailable columns:\n"
            + ", ".join(map(str, df.columns))
        )

    print("\nResolved targets:")

    for target in TARGETS:
        print(
            f"  {target} -> {mapping[target]}"
        )

    return mapping


# ============================================================
# Binary target normalization
# ============================================================

def normalize_binary_series(
    series: pd.Series,
    target_name: str,
) -> pd.Series:
    """
    Convert a binary target into integer labels:
        0 = negative / normal
        1 = positive / disease / stenosis
    """

    series = series.copy()

    # --------------------------------------------------------
    # Numeric targets
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        values = pd.to_numeric(
            series,
            errors="coerce",
        )

        unique_values = sorted(
            set(
                values.dropna()
                .unique()
                .tolist()
            )
        )

        if set(unique_values).issubset({0, 1}):
            return values.astype(int)

        if len(unique_values) == 2:
            low, high = unique_values

            return (
                values
                .map({
                    low: 0,
                    high: 1,
                })
                .astype(int)
            )

    # --------------------------------------------------------
    # Text targets
    # --------------------------------------------------------

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    positive = {
        "1",
        "yes",
        "y",
        "true",
        "cad",
        "disease",
        "abnormal",
        "stenosis",
        "stenotic",
        "positive",
        "present",
        "severe",
        "significant",
        ">=50",
        ">50",
        ">=50%",
        ">50%",
        "50",
        "50%",
    }

    negative = {
        "0",
        "no",
        "n",
        "false",
        "normal",
        "healthy",
        "non-stenotic",
        "nonstenotic",
        "negative",
        "absent",
        "none",
        "<50",
        "<50%",
    }

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="Int64",
    )

    result[text.isin(positive)] = 1
    result[text.isin(negative)] = 0

    # --------------------------------------------------------
    # Fallback for two unknown categorical levels
    # --------------------------------------------------------

    if result.isna().any():

        levels = sorted(
            text.dropna()
            .unique()
            .tolist()
        )

        if len(levels) == 2:

            result = (
                text
                .map({
                    levels[0]: 0,
                    levels[1]: 1,
                })
                .astype("Int64")
            )

    # --------------------------------------------------------
    # Fail if labels cannot be understood
    # --------------------------------------------------------

    if result.isna().any():

        invalid_values = sorted(
            text[result.isna()]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unrecognized labels in target "
            f"'{target_name}': {invalid_values}"
        )

    return result.astype(int)


# ============================================================
# Leakage / identifier detection
# ============================================================

def is_identifier_column(
    column: object,
) -> bool:
    """
    Identify columns that must not be used as input features.
    """

    c = canonical(column)

    # Explicit leakage/target columns.
    if c in LEAKAGE_COLUMNS:
        return True

    # Patient identifiers.
    if any(
        pattern in c
        for pattern in IDENTIFIER_PATTERNS
    ):
        return True

    # Excel-generated index columns.
    if c.startswith("unnamed"):
        return True

    if c in {
        "index",
        "serialno",
        "srno",
    }:
        return True

    return False


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.Series],
    dict[str, str],
]:
    """
    Prepare leakage-safe features and targets.

    Returns
    -------
    X:
        Clinical/demographic/ECG/lab/echo features.

    y_all:
        Four target vectors:
            CAD
            LAD
            LCX
            RCA

    target_columns:
        Mapping of logical targets to source dataframe columns.
    """

    df = df.copy()

    # Normalize column names.
    df.columns = [
        normalize_column(column)
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Resolve target columns
    # --------------------------------------------------------

    target_columns = resolve_targets(df)

    # --------------------------------------------------------
    # Build target vectors
    # --------------------------------------------------------

    y_all: dict[str, pd.Series] = {}

    for target in TARGETS:

        source_column = target_columns[target]

        y_all[target] = normalize_binary_series(
            df[source_column],
            target,
        )

    # --------------------------------------------------------
    # Build feature list
    # --------------------------------------------------------

    feature_columns: list[str] = []

    for column in df.columns:

        # Never allow any resolved target into X.
        if column in target_columns.values():
            continue

        # Remove CAD/LAD/LCX/RCA/Cath and IDs.
        if is_identifier_column(column):
            continue

        feature_columns.append(column)

    if not feature_columns:
        raise ValueError(
            "No usable clinical features remain after "
            "leakage/identifier removal."
        )

    # --------------------------------------------------------
    # Feature dataframe
    # --------------------------------------------------------

    X = df[feature_columns].copy()

    # --------------------------------------------------------
    # Keep rows with valid labels for every target
    # --------------------------------------------------------

    valid_mask = pd.Series(
        True,
        index=df.index,
    )

    for target in TARGETS:
        valid_mask &= y_all[target].notna()

    X = X.loc[valid_mask]

    for target in TARGETS:
        y_all[target] = y_all[target].loc[
            valid_mask
        ]

    if X.empty:
        raise ValueError(
            "No valid patient records remain after "
            "target preprocessing."
        )

    # --------------------------------------------------------
    # Final leakage safety check
    # --------------------------------------------------------

    forbidden = {
        canonical(column)
        for column in LEAKAGE_COLUMNS
    }

    actual_features = {
        canonical(column)
        for column in X.columns
    }

    leakage_detected = (
        forbidden & actual_features
    )

    if leakage_detected:
        raise RuntimeError(
            "TARGET LEAKAGE DETECTED!\n"
            "Forbidden columns found in model features:\n"
            + ", ".join(
                sorted(leakage_detected)
            )
        )

    # --------------------------------------------------------
    # Print useful training information
    # --------------------------------------------------------

    print(
        f"\nPrepared dataset:"
        f"\n  Patients : {X.shape[0]}"
        f"\n  Features : {X.shape[1]}"
    )

    print(
        "\nExcluded from model inputs:"
    )

    print(
        "  CAD / Cath"
        "\n  LAD"
        "\n  LCX"
        "\n  RCA"
        "\n  Patient identifiers / index columns"
    )

    print(
        "\nTarget mapping:"
    )

    print(
        f"  CAD -> {target_columns['CAD']}"
    )

    print(
        f"  LAD -> {target_columns['LAD']}"
    )

    print(
        f"  LCX -> {target_columns['LCX']}"
    )

    print(
        f"  RCA -> {target_columns['RCA']}"
    )

    return (
        X,
        y_all,
        target_columns,
    )