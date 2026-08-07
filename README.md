# BOAMP Raw Data Acquisition

Download the historical BOAMP raw corpus with:

```bash
python3 scripts/download_boamp.py
```

This retrieves one JSONL file per complete calendar year for:

```text
2015-01-01 <= dateparution < 2026-01-01
```

Optional current-year partial data is kept separate:

```bash
python3 scripts/download_boamp.py --include-2026
```

Force a redownload of already completed years:

```bash
python3 scripts/download_boamp.py --force
```

Inspect API metadata/counts without downloading raw records:

```bash
python3 scripts/download_boamp.py --inspect-only
```

Raw BOAMP files and logs are ignored by git. Metadata is written to
`data/metadata/`.
