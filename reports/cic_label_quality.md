# CIC-IDS2017 Label Quality Report

## Issue identified

File: `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

The CSV is valid UTF-8, but the raw file contains 2,180 occurrences of the UTF-8 byte sequence `EF BF BD`, representing the Unicode replacement character `�`.

The affected labels are:

| Original label in local CSV | Count |
|---|---:|
| `Web Attack � Brute Force` | 1,507 |
| `Web Attack � XSS` | 652 |
| `Web Attack � Sql Injection` | 21 |

## Planned handling

During preprocessing, normalize only these three label values to the following project-standard names:

| Original label | Normalized label |
|---|---|
| `Web Attack � Brute Force` | `Web Attack - Brute Force` |
| `Web Attack � XSS` | `Web Attack - XSS` |
| `Web Attack � Sql Injection` | `Web Attack - SQL Injection` |

The ASCII hyphen is a project-standard separator, not a claim about the exact original punctuation.

## Data integrity rules

- Preserve all raw CSV files unchanged.
- Do not merge the three attack categories.
- Do not delete records solely because their labels contain the replacement character.
- Verify that the affected counts remain unchanged after normalization.
- Record the normalization in the preprocessing code and experiment documentation.

## Status

Encoding verified as valid UTF-8. Label corruption identified. Normalization is planned but has not yet been applied.
