# %% Imports
"""
Temporal Analysis of 5G Coverage and Performance
Analyzes how coverage and performance metrics change over time across multiple sub-campaigns
at the same locations in Rome (2023 dataset).
Includes trend analysis, temporal comparisons, and early vs late campaign analysis.
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from scikit_posthocs import posthoc_dunn
from datetime import datetime
from warnings import filterwarnings
filterwarnings('ignore')

def print_heading(title: str):
    print("="*80)
    print(title)
    print("="*80)

def print_subheading(title: str):
    print("-"*80)
    print(title)
    print("-"*80)

def mann_kendall_test(data):
    """
    Mann-Kendall trend test for monotonic trends.
    Returns: (tau, p_value, trend_direction)
    trend_direction: 'increasing', 'decreasing', or 'no trend'
    """
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])

    # Calculate variance
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Calculate z-score
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Kendall's tau
    tau = s / (0.5 * n * (n - 1))

    # Determine trend direction
    if p_value < 0.05:
        trend = 'increasing' if tau > 0 else 'decreasing'
    else:
        trend = 'no trend'

    return tau, p_value, trend

# %% Configuration
DATA_DIR = Path("5G_2023_passive")
OUTPUT_DIR = Path("Output")
LOCATIONS = {
    4: "Urban (Rome City Center)",
    7: "Suburban",
    9: "Suburban"
}
ALPHA = 0.05  # Significance level
RSRP_THRESHOLD = -110  # dBm - Coverage quality threshold

# %% Load Data
print_heading("LOADING DATA")

all_data: list[pl.DataFrame] = []

for loc_num, loc_type in LOCATIONS.items():
    # Find all outdoor files for this location
    pattern = f"location_{loc_num}_od_*.csv"
    files = list(DATA_DIR.glob(pattern))

    print(f"Location {loc_num} ({loc_type}): Found {len(files)} files")

    for file in files:
        try:
            df = pl.read_csv(
                file,
                null_values=["NA", ""],
                infer_schema_length=10000,
                ignore_errors=True
            )
            df = df.with_columns([
                pl.lit(loc_num).alias('location_num'),
                pl.lit(loc_type).alias('location_type'),
                pl.lit(file.name).alias('source_file')
            ])
            all_data.append(df)
        except Exception as e:
            print(f"  Error reading {file.name}: {e}")

# Combine all data
df = pl.concat(all_data)
print(f"Total records loaded: {len(df):,}")

# %% Clean Operator Field
# MNC field contains values like: """Op""[1]", """Op""[2]", etc.
df = df.with_columns(
    pl.col('MNC').str.extract(r'\[(\d+)\]', 1).alias('operator')
)

# %% Parse Dates
print_heading("PARSING TEMPORAL DATA")

# Parse Date field from DD.MM.YYYY format
df = df.with_columns([
    pl.col('Date').str.strip_chars('"').str.to_date('%d.%m.%Y').alias('date_parsed')
])

# Show date range per location
print("Date ranges by location:")
date_ranges = df.group_by('location_num').agg([
    pl.col('date_parsed').min().alias('first_date'),
    pl.col('date_parsed').max().alias('last_date'),
    pl.col('date_parsed').n_unique().alias('num_dates')
]).sort('location_num')
print(date_ranges)

# Get all unique dates per location
print("\nUnique dates per location:")
for loc in sorted(LOCATIONS.keys()):
    dates = df.filter(pl.col('location_num') == loc)['date_parsed'].unique().sort()
    dates_str = [d.strftime('%Y-%m-%d') for d in dates if d is not None]
    print(f"  Location {loc}: {dates_str}")

# %% Select Operator
print_heading("OPERATOR SELECTION")

# Count records per operator per location
operator_counts = df.group_by(['location_num', 'operator']).agg(pl.len().alias('count'))
operator_counts_pivot = operator_counts.pivot(
    index='location_num',
    on='operator',
    values='count'
).fill_null(0)
print("Records per operator per location:")
print(operator_counts_pivot)

# Find operator present in all locations with most total data
operator_cols = [col for col in operator_counts_pivot.columns if col != 'location_num']
present_in_all = operator_counts_pivot.select(
    [(pl.col(col) > 0).all().alias(col) for col in operator_cols]
)
operators_in_all_list = [col for col in operator_cols if present_in_all[col][0]]

if len(operators_in_all_list) == 0:
    print("No operator found in all locations. Using operator with most total data.")
    selected_operator = df['operator'].value_counts().sort('count', descending=True)['operator'][0]
else:
    # Get total counts for operators present in all locations
    totals = operator_counts_pivot.select(
        [pl.col(col).sum().alias(col) for col in operators_in_all_list]
    )
    selected_operator = max(operators_in_all_list, key=lambda col: totals[col][0])

print(f"\nSelected Operator: Op[{selected_operator}]")

# Filter data for selected operator
df_filtered = df.filter(pl.col('operator') == selected_operator)

print(f"Records after filtering: {len(df_filtered):,}")
print("\nRecords per location:")
for loc in sorted(df_filtered['location_num'].unique()):
    count = len(df_filtered.filter(pl.col('location_num') == loc))
    loc_name = LOCATIONS[loc]
    print(f"  Location {loc} ({loc_name}): {count:,}")

# %% Prepare Temporal Metrics
print_heading("TEMPORAL METRIC CALCULATION")

# Select relevant columns and remove nulls
df_temporal = df_filtered.select([
    'location_num',
    'location_type',
    'date_parsed',
    'SSS_RSRP',
    'SSS-SINR'
]).drop_nulls(subset=['date_parsed', 'SSS_RSRP', 'SSS-SINR'])

print(f"Records with valid date, RSRP, and SINR: {len(df_temporal):,}")

# Calculate daily statistics per location
daily_stats = df_temporal.group_by(['location_num', 'location_type', 'date_parsed']).agg([
    pl.col('SSS_RSRP').count().alias('sample_count'),
    pl.col('SSS_RSRP').mean().alias('mean_rsrp'),
    pl.col('SSS_RSRP').median().alias('median_rsrp'),
    pl.col('SSS_RSRP').std().alias('std_rsrp'),
    pl.col('SSS_RSRP').quantile(0.25).alias('q25_rsrp'),
    pl.col('SSS_RSRP').quantile(0.75).alias('q75_rsrp'),
    pl.col('SSS-SINR').mean().alias('mean_sinr'),
    pl.col('SSS-SINR').median().alias('median_sinr'),
    pl.col('SSS-SINR').std().alias('std_sinr'),
    pl.col('SSS-SINR').quantile(0.25).alias('q25_sinr'),
    pl.col('SSS-SINR').quantile(0.75).alias('q75_sinr'),
    # Coverage quality: percentage above threshold
    ((pl.col('SSS_RSRP') > RSRP_THRESHOLD).sum() / pl.col('SSS_RSRP').count() * 100).alias('coverage_quality_pct')
]).sort(['location_num', 'date_parsed'])

print("\nDaily statistics summary:")
print(daily_stats)

# %% RSRP Statistics Over Time
print_subheading("RSRP Statistics Over Time by Location (dBm)")
for loc in sorted(LOCATIONS.keys()):
    loc_stats = daily_stats.filter(pl.col('location_num') == loc)
    print(f"\nLocation {loc} ({LOCATIONS[loc]}):")
    print(loc_stats.select(['date_parsed', 'sample_count', 'mean_rsrp', 'std_rsrp', 'median_rsrp']))

# %% SINR Statistics Over Time
print_subheading("SINR Statistics Over Time by Location (dB)")
for loc in sorted(LOCATIONS.keys()):
    loc_stats = daily_stats.filter(pl.col('location_num') == loc)
    print(f"\nLocation {loc} ({LOCATIONS[loc]}):")
    print(loc_stats.select(['date_parsed', 'sample_count', 'mean_sinr', 'std_sinr', 'median_sinr']))

# %% Coverage Quality Over Time
print_subheading("Coverage Quality Over Time (% RSRP > -110 dBm)")
for loc in sorted(LOCATIONS.keys()):
    loc_stats = daily_stats.filter(pl.col('location_num') == loc)
    print(f"\nLocation {loc} ({LOCATIONS[loc]}):")
    print(loc_stats.select(['date_parsed', 'coverage_quality_pct']))

# %% Statistical Tests - Mann-Kendall Trend Analysis
print_heading("TREND ANALYSIS - MANN-KENDALL TEST")

print("Tests for monotonic trends over time (H0: No monotonic trend)")
print(f"Significance level (α): {ALPHA}\n")

# RSRP Trends
print_subheading("RSRP Trends Over Time")
for loc in sorted(LOCATIONS.keys()):
    loc_stats = daily_stats.filter(pl.col('location_num') == loc).sort('date_parsed')
    rsrp_values = loc_stats['mean_rsrp'].to_numpy()

    if len(rsrp_values) >= 3:  # Need at least 3 points for trend test
        tau, p_value, trend = mann_kendall_test(rsrp_values)
        sig_text = "SIGNIFICANT" if p_value < ALPHA else "NOT SIGNIFICANT"

        print(f"\nLocation {loc} ({LOCATIONS[loc]}):")
        print(f"  Kendall's tau: {tau:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Trend: {trend.upper()} [{sig_text}]")

        if p_value < ALPHA:
            direction = "improving" if trend == 'increasing' else "degrading"
            print(f"  Interpretation: RSRP is {direction} over time")
    else:
        print(f"\nLocation {loc} ({LOCATIONS[loc]}): Insufficient data points for trend test")

# SINR Trends
print_subheading("SINR Trends Over Time")
for loc in sorted(LOCATIONS.keys()):
    loc_stats = daily_stats.filter(pl.col('location_num') == loc).sort('date_parsed')
    sinr_values = loc_stats['mean_sinr'].to_numpy()

    if len(sinr_values) >= 3:
        tau, p_value, trend = mann_kendall_test(sinr_values)
        sig_text = "SIGNIFICANT" if p_value < ALPHA else "NOT SIGNIFICANT"

        print(f"\nLocation {loc} ({LOCATIONS[loc]}):")
        print(f"  Kendall's tau: {tau:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Trend: {trend.upper()} [{sig_text}]")

        if p_value < ALPHA:
            direction = "improving" if trend == 'increasing' else "degrading"
            print(f"  Interpretation: SINR is {direction} over time")
    else:
        print(f"\nLocation {loc} ({LOCATIONS[loc]}): Insufficient data points for trend test")

# Coverage Quality Trends
print_subheading("Coverage Quality Trends Over Time")
for loc in sorted(LOCATIONS.keys()):
    loc_stats = daily_stats.filter(pl.col('location_num') == loc).sort('date_parsed')
    coverage_values = loc_stats['coverage_quality_pct'].to_numpy()

    if len(coverage_values) >= 3:
        tau, p_value, trend = mann_kendall_test(coverage_values)
        sig_text = "SIGNIFICANT" if p_value < ALPHA else "NOT SIGNIFICANT"

        print(f"\nLocation {loc} ({LOCATIONS[loc]}):")
        print(f"  Kendall's tau: {tau:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Trend: {trend.upper()} [{sig_text}]")

        if p_value < ALPHA:
            direction = "improving" if trend == 'increasing' else "degrading"
            print(f"  Interpretation: Coverage quality is {direction} over time")
    else:
        print(f"\nLocation {loc} ({LOCATIONS[loc]}): Insufficient data points for trend test")

# %% Early vs Late Campaign Comparison
print_heading("EARLY VS LATE CAMPAIGN COMPARISON")

# Split data into early and late campaigns (by median date per location)
df_temporal_with_period = df_temporal.join(
    df_temporal.group_by('location_num').agg(
        pl.col('date_parsed').median().alias('median_date')
    ),
    on='location_num'
).with_columns([
    pl.when(pl.col('date_parsed') <= pl.col('median_date'))
    .then(pl.lit('Early'))
    .otherwise(pl.lit('Late'))
    .alias('campaign_period')
])

print("Campaign period definitions (split by median date):")
period_info = df_temporal_with_period.group_by(['location_num', 'campaign_period']).agg([
    pl.col('date_parsed').min().alias('start_date'),
    pl.col('date_parsed').max().alias('end_date'),
    pl.len().alias('sample_count')
]).sort(['location_num', 'campaign_period'])
print(period_info)

# %% Statistical Tests - Early vs Late (RSRP)
print_subheading("WILCOXON RANK-SUM TEST - EARLY VS LATE RSRP")

for loc in sorted(LOCATIONS.keys()):
    loc_data = df_temporal_with_period.filter(pl.col('location_num') == loc)
    early_rsrp = loc_data.filter(pl.col('campaign_period') == 'Early')['SSS_RSRP'].to_numpy()
    late_rsrp = loc_data.filter(pl.col('campaign_period') == 'Late')['SSS_RSRP'].to_numpy()

    if len(early_rsrp) > 0 and len(late_rsrp) > 0:
        stat, p_value = stats.ranksums(early_rsrp, late_rsrp)

        early_mean = np.mean(early_rsrp)
        late_mean = np.mean(late_rsrp)
        diff = late_mean - early_mean

        print(f"\nLocation {loc} ({LOCATIONS[loc]}):")
        print(f"  H0: Early and Late RSRP distributions are equal")
        print(f"  Early RSRP mean: {early_mean:.2f} dBm (n={len(early_rsrp):,})")
        print(f"  Late RSRP mean: {late_mean:.2f} dBm (n={len(late_rsrp):,})")
        print(f"  Difference: {diff:+.2f} dBm")
        print(f"  Test Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.6f}")

        if p_value < ALPHA:
            print(f"  Result: REJECT H0 (p < {ALPHA}) [SIGNIFICANT]")
            if diff > 0:
                print(f"  Interpretation: RSRP improved significantly from early to late campaigns")
            else:
                print(f"  Interpretation: RSRP degraded significantly from early to late campaigns")
        else:
            print(f"  Result: FAIL TO REJECT H0 (p >= {ALPHA}) [NOT SIGNIFICANT]")
            print(f"  Interpretation: No significant change in RSRP between campaign periods")

# %% Statistical Tests - Early vs Late (SINR)
print_subheading("WILCOXON RANK-SUM TEST - EARLY VS LATE SINR")

for loc in sorted(LOCATIONS.keys()):
    loc_data = df_temporal_with_period.filter(pl.col('location_num') == loc)
    early_sinr = loc_data.filter(pl.col('campaign_period') == 'Early')['SSS-SINR'].to_numpy()
    late_sinr = loc_data.filter(pl.col('campaign_period') == 'Late')['SSS-SINR'].to_numpy()

    if len(early_sinr) > 0 and len(late_sinr) > 0:
        stat, p_value = stats.ranksums(early_sinr, late_sinr)

        early_mean = np.mean(early_sinr)
        late_mean = np.mean(late_sinr)
        diff = late_mean - early_mean

        print(f"\nLocation {loc} ({LOCATIONS[loc]}):")
        print(f"  H0: Early and Late SINR distributions are equal")
        print(f"  Early SINR mean: {early_mean:.2f} dB (n={len(early_sinr):,})")
        print(f"  Late SINR mean: {late_mean:.2f} dB (n={len(late_sinr):,})")
        print(f"  Difference: {diff:+.2f} dB")
        print(f"  Test Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.6f}")

        if p_value < ALPHA:
            print(f"  Result: REJECT H0 (p < {ALPHA}) [SIGNIFICANT]")
            if diff > 0:
                print(f"  Interpretation: SINR improved significantly from early to late campaigns")
            else:
                print(f"  Interpretation: SINR degraded significantly from early to late campaigns")
        else:
            print(f"  Result: FAIL TO REJECT H0 (p >= {ALPHA}) [NOT SIGNIFICANT]")
            print(f"  Interpretation: No significant change in SINR between campaign periods")

# %% Create Time Series Visualizations
print_heading("CREATING VISUALIZATIONS")

# Set style
sns.set_style("whitegrid")

# Prepare data for plotting - convert to pandas
daily_stats_pd = daily_stats.to_pandas()
daily_stats_pd['location_label'] = daily_stats_pd.apply(
    lambda row: f"Loc {row['location_num']}\n{row['location_type']}", axis=1
)

# Time Series - RSRP
fig1, ax1 = plt.subplots(figsize=(14, 7))

for loc in sorted(LOCATIONS.keys()):
    loc_data = daily_stats_pd[daily_stats_pd['location_num'] == loc].sort_values('date_parsed')
    label = f"Loc {loc} ({LOCATIONS[loc]})"

    ax1.plot(loc_data['date_parsed'], loc_data['mean_rsrp'], marker='o', linewidth=2,
             markersize=8, label=label)

    # Add confidence interval (mean ± std)
    ax1.fill_between(loc_data['date_parsed'],
                      loc_data['mean_rsrp'] - loc_data['std_rsrp'],
                      loc_data['mean_rsrp'] + loc_data['std_rsrp'],
                      alpha=0.2)

ax1.set_title(f'RSRP Evolution Over Time - Operator {selected_operator}\n(Higher is Better)',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Mean RSRP (dBm) → Higher is Better', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_file = OUTPUT_DIR / f'temporal_rsrp_timeseries_operator_{selected_operator}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

plt.show()

# Time Series - SINR
fig2, ax2 = plt.subplots(figsize=(14, 7))

for loc in sorted(LOCATIONS.keys()):
    loc_data = daily_stats_pd[daily_stats_pd['location_num'] == loc].sort_values('date_parsed')
    label = f"Loc {loc} ({LOCATIONS[loc]})"

    ax2.plot(loc_data['date_parsed'], loc_data['mean_sinr'], marker='o', linewidth=2,
             markersize=8, label=label)

    # Add confidence interval
    ax2.fill_between(loc_data['date_parsed'],
                      loc_data['mean_sinr'] - loc_data['std_sinr'],
                      loc_data['mean_sinr'] + loc_data['std_sinr'],
                      alpha=0.2)

ax2.set_title(f'SINR Evolution Over Time - Operator {selected_operator}\n(Higher is Better)',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Mean SINR (dB) → Higher is Better', fontsize=12)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_file = OUTPUT_DIR / f'temporal_sinr_timeseries_operator_{selected_operator}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

plt.show()

# Time Series - Coverage Quality
fig3, ax3 = plt.subplots(figsize=(14, 7))

for loc in sorted(LOCATIONS.keys()):
    loc_data = daily_stats_pd[daily_stats_pd['location_num'] == loc].sort_values('date_parsed')
    label = f"Loc {loc} ({LOCATIONS[loc]})"

    ax3.plot(loc_data['date_parsed'], loc_data['coverage_quality_pct'], marker='o',
             linewidth=2, markersize=8, label=label)

ax3.set_title(f'Coverage Quality Evolution Over Time - Operator {selected_operator}\n(% Measurements with RSRP > -110 dBm)',
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Coverage Quality (%) → Higher is Better', fontsize=12)
ax3.set_ylim([0, 105])
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_file = OUTPUT_DIR / f'temporal_coverage_quality_operator_{selected_operator}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

plt.show()

# %% Early vs Late Distribution Comparisons
print_subheading("Creating Early vs Late Comparison Plots")

# Prepare data
df_comparison_pd = df_temporal_with_period.select([
    'location_num', 'location_type', 'campaign_period', 'SSS_RSRP', 'SSS-SINR'
]).to_pandas()
df_comparison_pd['location_label'] = df_comparison_pd.apply(
    lambda row: f"Loc {row['location_num']}\n{row['location_type']}", axis=1
)

# Early vs Late - RSRP
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

for idx, loc in enumerate(sorted(LOCATIONS.keys())):
    loc_data = df_comparison_pd[df_comparison_pd['location_num'] == loc]

    sns.violinplot(data=loc_data, x='campaign_period', y='SSS_RSRP', ax=axes4[idx],
                   palette='Set2', order=['Early', 'Late'])

    axes4[idx].set_title(f"Loc {loc}\n{LOCATIONS[loc]}", fontsize=12, fontweight='bold')
    axes4[idx].set_xlabel('Campaign Period', fontsize=11)
    axes4[idx].set_ylabel('RSRP (dBm)' if idx == 0 else '', fontsize=11)

fig4.suptitle(f'Early vs Late Campaign RSRP Comparison - Operator {selected_operator}\n(Higher is Better)',
              fontsize=14, fontweight='bold')
plt.tight_layout()

output_file = OUTPUT_DIR / f'temporal_early_vs_late_rsrp_operator_{selected_operator}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

plt.show()

# Early vs Late - SINR
fig5, axes5 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

for idx, loc in enumerate(sorted(LOCATIONS.keys())):
    loc_data = df_comparison_pd[df_comparison_pd['location_num'] == loc]

    sns.violinplot(data=loc_data, x='campaign_period', y='SSS-SINR', ax=axes5[idx],
                   palette='Set2', order=['Early', 'Late'])

    axes5[idx].set_title(f"Loc {loc}\n{LOCATIONS[loc]}", fontsize=12, fontweight='bold')
    axes5[idx].set_xlabel('Campaign Period', fontsize=11)
    axes5[idx].set_ylabel('SINR (dB)' if idx == 0 else '', fontsize=11)

fig5.suptitle(f'Early vs Late Campaign SINR Comparison - Operator {selected_operator}\n(Higher is Better)',
              fontsize=14, fontweight='bold')
plt.tight_layout()

output_file = OUTPUT_DIR / f'temporal_early_vs_late_sinr_operator_{selected_operator}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

plt.show()

# %% Coverage Heatmap
print_subheading("Creating Coverage Quality Heatmap")

# Prepare heatmap data
heatmap_data = daily_stats.select([
    'location_num', 'date_parsed', 'coverage_quality_pct'
]).to_pandas()

# Pivot for heatmap
heatmap_pivot = heatmap_data.pivot(index='location_num', columns='date_parsed', values='coverage_quality_pct')
heatmap_pivot.index = [f"Loc {loc}\n{LOCATIONS[loc]}" for loc in heatmap_pivot.index]

fig6, ax6 = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_pivot, annot=True, fmt='.1f', cmap='RdYlGn', cbar_kws={'label': 'Coverage Quality (%)'},
            vmin=0, vmax=100, ax=ax6)

ax6.set_title(f'Coverage Quality Heatmap by Location and Date - Operator {selected_operator}\n(% Measurements with RSRP > -110 dBm)',
              fontsize=14, fontweight='bold')
ax6.set_xlabel('Campaign Date', fontsize=12)
ax6.set_ylabel('Location', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_file = OUTPUT_DIR / f'temporal_coverage_heatmap_operator_{selected_operator}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

plt.show()

# %% Interpretation and Conclusions
print_heading("TEMPORAL ANALYSIS INTERPRETATION")

print("\nSUMMARY OF FINDINGS:\n")

# Overall temporal span
all_dates = df_temporal['date_parsed'].unique().sort()
print(f"Analysis Period: {all_dates[0]} to {all_dates[-1]}")
print(f"Total Days Analyzed: {len(all_dates)}")

# Per-location summary
for loc in sorted(LOCATIONS.keys()):
    print(f"\n{'-'*80}")
    print(f"LOCATION {loc} ({LOCATIONS[loc]})")
    print(f"{'-'*80}")

    loc_daily = daily_stats.filter(pl.col('location_num') == loc).sort('date_parsed')

    # Date range
    dates = loc_daily['date_parsed'].to_list()
    print(f"Campaign Dates: {dates[0]} to {dates[-1]} ({len(dates)} campaigns)")

    # RSRP changes
    rsrp_first = loc_daily['mean_rsrp'][0]
    rsrp_last = loc_daily['mean_rsrp'][-1]
    rsrp_change = rsrp_last - rsrp_first
    print(f"\nRSRP: {rsrp_first:.2f} dBm (first) → {rsrp_last:.2f} dBm (last)")
    print(f"  Change: {rsrp_change:+.2f} dBm ({rsrp_change/abs(rsrp_first)*100:+.1f}%)")

    # SINR changes
    sinr_first = loc_daily['mean_sinr'][0]
    sinr_last = loc_daily['mean_sinr'][-1]
    sinr_change = sinr_last - sinr_first
    print(f"\nSINR: {sinr_first:.2f} dB (first) → {sinr_last:.2f} dB (last)")
    print(f"  Change: {sinr_change:+.2f} dB ({sinr_change/abs(sinr_first)*100:+.1f}%)")

    # Coverage quality changes
    cov_first = loc_daily['coverage_quality_pct'][0]
    cov_last = loc_daily['coverage_quality_pct'][-1]
    cov_change = cov_last - cov_first
    print(f"\nCoverage Quality: {cov_first:.1f}% (first) → {cov_last:.1f}% (last)")
    print(f"  Change: {cov_change:+.1f} percentage points")
