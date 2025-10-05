# %% Imports
"""
Urban Canyon Effect Analysis for 5G Performance
Compares RSRP and SINR distributions across urban (Location 4) and suburban (Locations 7, 9) areas in Rome.
Includes Kruskal-Wallis and Dunn's post-hoc tests for statistical significance.
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scikit_posthocs import posthoc_dunn
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

# %% Configuration
DATA_DIR = Path("5G_2023_passive")
OUTPUT_DIR = Path("Output")
LOCATIONS = {
    4: "Urban (Rome City Center)",
    7: "Suburban",
    9: "Suburban"
}
ALPHA = 0.05  # Significance level

# %% Load Data
print_heading("Loading Data")

all_data = []

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

print(f"Selected Operator: Op[{selected_operator}]")

# Filter data for selected operator
df_filtered = df.filter(pl.col('operator') == selected_operator)

print(f"Records after filtering: {len(df_filtered):,}")
print("Records per location:")
for loc in sorted(df_filtered['location_num'].unique()):
    count = len(df_filtered.filter(pl.col('location_num') == loc))
    loc_name = LOCATIONS[loc]
    print(f"  Location {loc} ({loc_name}): {count:,}")

# %% Calculate Metrics
print_heading("METRIC CALCULATION")

# Use SSS_RSRP and SSS-SINR as primary metrics
df_metrics = df_filtered.select(['location_num', 'location_type', 'SSS_RSRP', 'SSS-SINR'])

# Remove missing values
df_metrics = df_metrics.drop_nulls(subset=['SSS_RSRP', 'SSS-SINR'])

print(f"Records with valid RSRP/SINR: {len(df_metrics):,}")

# %% RSRP Statistics by Location
print_subheading("RSRP Statistics by Location (dBm)")
rsrp_stats = df_metrics.group_by('location_num').agg([
    pl.col('SSS_RSRP').count().alias('count'),
    pl.col('SSS_RSRP').mean().alias('mean'),
    pl.col('SSS_RSRP').std().alias('std'),
    pl.col('SSS_RSRP').min().alias('min'),
    pl.col('SSS_RSRP').quantile(0.25).alias('25%'),
    pl.col('SSS_RSRP').quantile(0.50).alias('50%'),
    pl.col('SSS_RSRP').quantile(0.75).alias('75%'),
    pl.col('SSS_RSRP').max().alias('max')
]).sort('location_num')
print(rsrp_stats)

# %% SINR Statistics by Location
print_subheading("SINR Statistics by Location (dB)")
sinr_stats = df_metrics.group_by('location_num').agg([
    pl.col('SSS-SINR').count().alias('count'),
    pl.col('SSS-SINR').mean().alias('mean'),
    pl.col('SSS-SINR').std().alias('std'),
    pl.col('SSS-SINR').min().alias('min'),
    pl.col('SSS-SINR').quantile(0.25).alias('25%'),
    pl.col('SSS-SINR').quantile(0.50).alias('50%'),
    pl.col('SSS-SINR').quantile(0.75).alias('75%'),
    pl.col('SSS-SINR').max().alias('max')
]).sort('location_num')
print(sinr_stats)

# %% Statistical Tests - Kruskal-Wallis
print_heading("STATISTICAL ANALYSIS")

# Prepare data groups by location
groups_rsrp = [df_metrics.filter(pl.col('location_num') == loc)['SSS_RSRP'].to_numpy()
               for loc in sorted(df_metrics['location_num'].unique())]
groups_sinr = [df_metrics.filter(pl.col('location_num') == loc)['SSS-SINR'].to_numpy()
               for loc in sorted(df_metrics['location_num'].unique())]

# Kruskal-Wallis test for RSRP
print("\n" + "-"*80)
print("KRUSKAL-WALLIS TEST - RSRP")
print("-"*80)
print("H0: The mean RSRP across all location groups is equal")
print("H1: At least one location group has different mean RSRP")

kw_stat_rsrp, kw_pval_rsrp = stats.kruskal(*groups_rsrp)
print(f"\nTest Statistic: {kw_stat_rsrp:.4f}")
print(f"P-value: {kw_pval_rsrp:.6f}")
print(f"Significance level (α): {ALPHA}")

if kw_pval_rsrp < ALPHA:
    print(f"Result: REJECT H0 (p < {ALPHA})")
    print("Conclusion: Statistically significant difference in RSRP across locations")
    rsrp_significant = True
else:
    print(f"Result: FAIL TO REJECT H0 (p >= {ALPHA})")
    print("Conclusion: No statistically significant difference in RSRP across locations")
    rsrp_significant = False

# Kruskal-Wallis test for SINR
print("\n" + "-"*80)
print("KRUSKAL-WALLIS TEST - SINR")
print("-"*80)
print("H0: The mean SINR across all location groups is equal")
print("H1: At least one location group has different mean SINR")

kw_stat_sinr, kw_pval_sinr = stats.kruskal(*groups_sinr)
print(f"\nTest Statistic: {kw_stat_sinr:.4f}")
print(f"P-value: {kw_pval_sinr:.6f}")
print(f"Significance level (α): {ALPHA}")

if kw_pval_sinr < ALPHA:
    print(f"Result: REJECT H0 (p < {ALPHA})")
    print("Conclusion: Statistically significant difference in SINR across locations")
    sinr_significant = True
else:
    print(f"Result: FAIL TO REJECT H0 (p >= {ALPHA})")
    print("Conclusion: No statistically significant difference in SINR across locations")
    sinr_significant = False

# %% Statistical Tests - Dunn's Post-hoc (RSRP)
if rsrp_significant:
    print_subheading("DUNN'S POST-HOC TEST - RSRP (Pairwise Comparisons)")
    # Convert to pandas for posthoc_dunn (doesn't support polars yet)
    df_metrics_pd = df_metrics.select(['location_num', 'SSS_RSRP', 'SSS-SINR']).to_pandas()
    dunn_rsrp = posthoc_dunn(df_metrics_pd, val_col='SSS_RSRP', group_col='location_num', p_adjust='bonferroni')
    print("\nP-values (Bonferroni corrected):")
    print(dunn_rsrp)

    # Interpret pairwise comparisons
    print("\nPairwise Significance (α = 0.05):")
    locations = sorted(df_metrics['location_num'].unique())
    for i, loc1 in enumerate(locations):
        for loc2 in locations[i+1:]:
            pval = dunn_rsrp.loc[loc1, loc2]
            sig = "SIGNIFICANT" if pval < ALPHA else "NOT SIGNIFICANT"
            loc1_name = LOCATIONS[loc1]
            loc2_name = LOCATIONS[loc2]
            print(f"  Location {loc1} vs {loc2}: p={pval:.6f} [{sig}]")

# %% Statistical Tests - Dunn's Post-hoc (SINR)
if sinr_significant:
    print_subheading("DUNN'S POST-HOC TEST - SINR (Pairwise Comparisons)")
    # Convert to pandas for posthoc_dunn (doesn't support polars yet)
    df_metrics_pd = df_metrics.select(['location_num', 'SSS_RSRP', 'SSS-SINR']).to_pandas()
    dunn_sinr = posthoc_dunn(df_metrics_pd, val_col='SSS-SINR', group_col='location_num', p_adjust='bonferroni')
    print("\nP-values (Bonferroni corrected):")
    print(dunn_sinr)

    # Interpret pairwise comparisons
    print("\nPairwise Significance (α = 0.05):")
    locations = sorted(df_metrics['location_num'].unique())
    for i, loc1 in enumerate(locations):
        for loc2 in locations[i+1:]:
            pval = dunn_sinr.loc[loc1, loc2]
            sig = "SIGNIFICANT" if pval < ALPHA else "NOT SIGNIFICANT"
            loc1_name = LOCATIONS[loc1]
            loc2_name = LOCATIONS[loc2]
            print(f"  Location {loc1} vs {loc2}: p={pval:.6f} [{sig}]")

# %% Create Visualizations
print_heading("CREATING VISUALIZATIONS")

# Set style
sns.set_style("whitegrid")

# Prepare location labels with type
df_plot = df_metrics.with_columns(
    (pl.lit("Loc ") + pl.col('location_num').cast(pl.Utf8) + pl.lit("\n") + pl.col('location_type')).alias('location_label')
).select(['location_label', 'SSS_RSRP', 'SSS-SINR'])

# Convert to pandas for seaborn plotting
df_metrics_pd = df_plot.to_pandas()

# RSRP Distribution
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df_metrics_pd, x='location_label', y='SSS_RSRP', ax=ax1, palette='Set2')
ax1.set_title(f'RSRP Distribution by Location - Operator {selected_operator}\n(Higher is Better)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Location', fontsize=12)
ax1.set_ylabel('RSRP (dBm) → Higher is Better', fontsize=12)

# Add Kruskal-Wallis result
sig_text = "***" if kw_pval_rsrp < 0.001 else "**" if kw_pval_rsrp < 0.01 else "*" if kw_pval_rsrp < 0.05 else "ns"
ax1.text(0.5, 0.95, f"Kruskal-Wallis: p={kw_pval_rsrp:.6f} [{sig_text}]",
         transform=ax1.transAxes, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save RSRP figure
output_file_rsrp = OUTPUT_DIR / f'urban_canyon_rsrp_operator_{selected_operator}.png'
plt.savefig(output_file_rsrp, dpi=300, bbox_inches='tight')

plt.show()

# SINR Distribution
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df_metrics_pd, x='location_label', y='SSS-SINR', ax=ax2, palette='Set2')
ax2.set_title(f'SINR Distribution by Location - Operator {selected_operator}\n(Higher is Better)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Location', fontsize=12)
ax2.set_ylabel('SINR (dB) → Higher is Better', fontsize=12)

# Add Kruskal-Wallis result
sig_text = "***" if kw_pval_sinr < 0.001 else "**" if kw_pval_sinr < 0.01 else "*" if kw_pval_sinr < 0.05 else "ns"
ax2.text(0.5, 0.95, f"Kruskal-Wallis: p={kw_pval_sinr:.6f} [{sig_text}]",
         transform=ax2.transAxes, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save SINR figure
output_file_sinr = OUTPUT_DIR / f'urban_canyon_sinr_operator_{selected_operator}.png'
plt.savefig(output_file_sinr, dpi=300, bbox_inches='tight')

plt.show()

# %% Interpretation and Conclusions
print_heading("URBAN CANYON EFFECT INTERPRETATION")

# Get mean values
means = df_metrics.group_by('location_num').agg([
    pl.col('SSS_RSRP').mean().alias('mean_RSRP'),
    pl.col('SSS-SINR').mean().alias('mean_SINR')
]).sort('location_num')

urban_rsrp = means.filter(pl.col('location_num') == 4)['mean_RSRP'][0]
urban_sinr = means.filter(pl.col('location_num') == 4)['mean_SINR'][0]

suburban_means = means.filter(pl.col('location_num').is_in([7, 9]))
suburban_rsrp = suburban_means['mean_RSRP'].mean()
suburban_sinr = suburban_means['mean_SINR'].mean()

print(f"\nMean RSRP - Urban (Loc 4): {urban_rsrp:.2f} dBm")
print(f"Mean RSRP - Suburban (Loc 7, 9): {suburban_rsrp:.2f} dBm")
print(f"Difference: {urban_rsrp - suburban_rsrp:.2f} dBm")

print(f"\nMean SINR - Urban (Loc 4): {urban_sinr:.2f} dB")
print(f"Mean SINR - Suburban (Loc 7, 9): {suburban_sinr:.2f} dB")
print(f"Difference: {urban_sinr - suburban_sinr:.2f} dB")
