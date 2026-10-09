# EQUILYM - Automated Insight Generation Engine

A general-purpose auto-analytics engine for **district-level monthly healthcare data**. Upload a CSV and EQUILYM
automatically finds significant month-over-month **trends**, statistical **outliers**, **correlations** between indicators
and optional **threshold breaches**, then turns them into human-readable, severity-ranked insights with charts and
downloadable results. Python only (Pandas, NumPy, SciPy, Streamlit, Plotly); no API key, cloud service or internet needed for analysis.

## 1. Problem statement (Assignment 4)
Build an Auto-Analytics Engine that ingests a small district-level healthcare CSV and identifies trends, outliers and
correlations, producing human-readable insights - with **no hardcoded district-specific narratives**. Templated text is fine,
but every number, district, period and coefficient must come from the data.

## 2. Features
- CSV upload (or bundled sample), `head()`, `info()` equivalent, per-column missing counts, overview metrics.
- Validation with actionable messages; nothing is silently deleted, clipped or aggregated.
- Trend detection (configurable threshold, default 10 %), IQR and Z-score outliers, Pearson matrix with flagged pairs.
- Insight engine with four types (`trend`, `outlier`, `correlation`, `threshold_breach`) and Low/Medium/High severity from documented, configurable rules.
- Live sidebar filters (district, period range, indicator) and sliders for every threshold; one-click reset.
- Insight table (sortable, filterable), severity bar chart, correlation heatmap, per-district line chart, outlier plot, trend bar chart.
- Exports: insights CSV (all / displayed), insights JSON, correlation matrix CSV, trends CSV, outliers CSV, data-quality JSON.

## 3. Architecture
```
EQUILYM/
├── app.py                    # Streamlit UI only: widgets, caching, display
├── requirements.txt
├── README.md
├── data/sample_healthcare_data.csv
├── outputs/sample_insights.json   # real output of the pipeline on the sample (default settings)
├── src/
│   ├── data_loader.py        # CSV reading (all-text), info()/missing-value helpers
│   ├── validation.py         # schema checks, month/number normalisation, duplicates, ranges
│   ├── trend_detection.py    # consecutive-period % change
│   ├── outlier_detection.py  # IQR / Z-score
│   ├── correlation_analysis.py  # Pearson matrix, flagged pairs, small-sample warnings
│   ├── threshold_rules.py    # optional absolute limits -> threshold_breach
│   ├── insight_generator.py  # insight records, severity, templated explanations
│   ├── pipeline.py           # filtered scope -> engines -> insights (also used by tests)
│   ├── charts.py             # Plotly figure builders
│   ├── export_utils.py       # CSV/JSON export, formula-injection-safe
│   └── utils.py              # SeverityConfig, formatting, month helpers
├── tests/                    # pytest suite (7 module files + pipeline integration)
└── screenshots/              # empty - see "Known limitations"
```
Data flow: CSV upload -> validation -> normalised dataset -> filtered scope -> trend / outlier / correlation / breach
calculations -> insights -> dashboard and exports. Results are cached per (data, filters, thresholds), so changing any
control recomputes and stale results are never shown.

## 4. Setup (Windows-friendly)
Prerequisites: Python 3.10+ (3.12 used for development).
```bat
cd EQUILYM
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
macOS/Linux: `source .venv/bin/activate` instead of the activate line above.

**Launch:** `streamlit run app.py`  **Tests:** `pytest -v`

## 5. Dataset schema
| Column | Meaning |
|---|---|
| `month` | Reporting month (`2026-07`; also accepts `2026-07-01`, `Jul 2026`, `July 2026`, `07/2026`, `2026/07`, `202607`) |
| `district` | District name/identifier |
| any other numeric column | Detected automatically as an indicator (e.g. `anc_coverage`, `institutional_delivery`, `immunization` as %, `high_risk_cases` as a count) |

One row per `(district, month)`. Non-numeric extra columns (e.g. `region`) are kept but not analysed. A column counts as numeric
when >= 80 % of its non-missing values parse as numbers; ID-like numeric columns (e.g. a numeric district code) would be treated
as indicators - deselect them in the sidebar Indicators filter. Column names are trimmed, lower-cased and spaces/hyphens become `_`.
`data/sample_healthcare_data.csv` is the 12-row, 6-district, 2-month example from the assignment; it is for demonstration and testing only.

## 6. Data validation rules
| Situation | Behaviour |
|---|---|
| Missing `month`/`district` column, empty file, header only, malformed rows, duplicate column names, no numeric indicator | **Error**, analysis blocked, message says what to fix |
| Missing district or unparsable month | Row **excluded from analysis but listed** in the excluded-rows table |
| Numbers as text (`"85"`, `85%`, `1,200`) | Converted |
| Non-numeric text in a numeric column | Treated as missing, **warning with CSV line numbers** |
| Percent indicator (name contains coverage/percent/pct/share/proportion/immuniz/delivery) outside 0-100, or any other indicator below 0 | **Warning**; value **kept, not clipped** (optional "treat as missing" switch; "may be negative" list for signed indicators) |
| Duplicate `(district, month)` | **Error** until you pick a policy (keep first / keep last / mean); the applied policy is reported |
| Missing months, single-observation districts, small dataset | Warning/info; trend comparisons across a gap are labelled |
| District names differing only by case | Warning (not merged) |

Uploaded files are parsed as data only; exported text cells starting with `= + - @` are prefixed with `'` to prevent spreadsheet formula injection.

## 7. Methods
**Trend** (per district and indicator, consecutive *available* periods, chronological):
`pct_change = (current - previous) / |previous| * 100`, `change = current - previous`; significant if `|pct_change| >= threshold`
(default 10). Previous value 0 -> percentage change is *undefined* (never divided); a 0 -> non-zero move is flagged as a
zero-baseline change for review. A comparison that skips calendar months records `gap_periods` and `missing_periods`.
Example from the sample: Ahmedabad ANC coverage 85 -> 69, change -16, pct_change -18.8 %.

**Outliers** (population = see Filtering; quantiles use linear interpolation):
- IQR: `Q1 - k*IQR` / `Q3 + k*IQR`, `k` default 1.5; outlier if outside the bounds. Needs >= 4 values.
- Z-score: `z = (x - mean) / sd` (sample sd, `ddof=1`); flagged when `|z| >= threshold` (default 3). Needs >= 3 values and `sd > 0`; constant columns get an explanation instead of a score. Note that with `n` values `|z|` can never exceed `(n-1)/sqrt(n)` (3.17 for n = 12), so a threshold of 3 is nearly unreachable on tiny datasets - the app says so.
- The Z-score column is shown for both methods when defined; for the Z method the displayed bounds are `mean +/- threshold*sd`.

**Correlation:** Pearson `r` on pairwise-complete observations (`DataFrame.corr()`), pairs flagged when `|r| >= 0.70` (configurable), unique pairs only,
direction recorded, p-value from `scipy.stats.pearsonr`. Undefined values (constant column, < 3 paired points) stay empty/`n/a` - never 0 - and are never flagged.

**Insights and severity** (all boundaries adjustable in the sidebar; severity is a *review priority*, not a clinical judgement):
| Type | Severity basis | Low | Medium | High |
|---|---|---|---|---|
| trend | `|pct_change| / trend threshold` | 1.0x - <1.5x | 1.5x - <2.0x | >= 2.0x |
| outlier (IQR) | distance beyond bound / IQR | < 1.0 | 1.0 - <2.0 | >= 2.0 |
| outlier (Z) | `|z| / z threshold` | < 1.25x | 1.25x - <1.5x | >= 1.5x |
| correlation | `|r|` | < 0.80 | 0.80 - <0.90 | >= 0.90 (sample size reported separately) |
| threshold_breach | % beyond the configured limit | < 10 % | 10 - <25 % | >= 25 % |
Zero-baseline trends get a fixed `Medium`. Each insight stores its basis in `supporting_data.severity_basis`.
Note: the assignment's example output labels Ahmedabad's -18.8 % as HIGH, but the documented rule above (1.88x the threshold) gives **Medium**;
lower "Trend: High from" to 1.8 in the sidebar to make it High.

Insight fields: `insight_id` (`INS-0001`..., deterministic order: type, severity, entity, indicator, period), `type`, `indicator`
(`a:b` for correlations), `entity` (district, or "N districts (pooled)" for correlations), `period` (current period for trends,
`start..end` for correlations), `metric`/`metric_name`, `value`, `prev_value`, `change`, `change_pct`, `severity`, `explanation`,
`supporting_data`. Fields that do not apply are `null` (empty in CSV). A `threshold_breach` is produced only by the optional absolute-limit rules,
never merely because a trend threshold was crossed. If nothing meets the thresholds the app says so and shows no insights.

## 8. Filtering behaviour
- District and period-range filters define the **filtered rows**; trends, threshold rules, displayed outliers and district charts use them.
- Outlier bounds and the correlation matrix use the population selected in the sidebar: **Full validated dataset (global)** (default - filters only choose which outliers are shown) or **Current filters only** (statistics recomputed on the filtered rows). The Overview tab lists the row counts of the dataset, filtered rows and statistics population, and each tab repeats which population it uses.
- The Indicator filter chooses which indicators are analysed everywhere. Insight-table filters (type, severity, district, indicator, period) affect only that table and its severity chart; IDs never change when you filter.

## 9. Exports
Download buttons in the **Export** tab: `insights_all.csv` (every insight in scope, columns `insight_id,type,indicator,entity,period,value,prev_value,change,change_pct,severity,explanation`),
`insights_filtered.csv` (only what the insight table currently displays), `insights_all.json`, `correlation_matrix.csv`,
`trends_all_comparisons.csv`, `outliers.csv`, `data_quality_report.json`. CSVs are UTF-8 with BOM (Excel-friendly), numbers rounded to 4 decimals.
`outputs/sample_insights.json` is the real JSON produced from the sample dataset with default settings.

## 10. Testing
`pytest -v` runs unit tests (trends, outliers, correlations, insights, exports, loader, validation) and integration tests on the 12-row sample (confirms the Ahmedabad ANC -18.8 % result under a 10 % threshold).

## 11. Why correlations from 2 months x 6 districts are fragile
The sample has 12 rows, but they are 6 districts observed twice, so there are effectively 6 independent units, and repeated months of one district are not independent. With so few points a single district can create or destroy a correlation (in the sample, ANC coverage vs high-risk cases has r = -0.93; dropping Mehsana alone moves it to -0.78). The 95 % confidence interval for an `r` from n = 12 is very wide, and the assignment recommends >= 10 districts and >= 3 months. EQUILYM therefore shows the matrix as computed, adds a "Limited sample" warning to the dashboard, to each flagged pair and to its insight text, and never claims causation.

## 12. Known limitations
- Percent-vs-count detection is name-based (see section 6); other indicators are only checked for negatives unless marked signed.
- Outlier bounds pool all districts and months, so seasonal or district-specific baselines are not modelled.
- Severity boundaries are analytical defaults, not clinically validated.
- Pearson `r` assumes linear relationships and is sensitive to outliers.
- Screenshots are not included (they must be captured from a running app); `outputs/sample_insights.json` is provided instead.

## 13. Troubleshooting
- `streamlit` not found: activate the virtual environment, then `pip install -r requirements.txt`.
- Port busy: `streamlit run app.py --server.port 8502`.
- "The data cannot be analysed yet": read the red messages in the Data quality tab (duplicates, missing columns, ...).
- Unexpected zero results: check the sidebar filters and press **Clear / reset filters and thresholds**.
- Non-UTF-8 CSV: it is decoded as Windows-1252 with a notice; re-save as UTF-8 if accented names look wrong.
