import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
FIGURES_DIR = 'exploratory/figures'

df = pd.read_csv('data/main.csv', encoding='utf-8-sig')

# Extract all judge score columns
score_cols = [c for c in df.columns if 'judge' in c.lower() and 'score' in c.lower()]

# Melt scores into long format for analysis
def extract_scores(df):
    records = []
    for _, row in df.iterrows():
        for col in score_cols:
            val = row[col]
            if pd.notna(val) and val != 'N/A' and val != 0:
                try:
                    score = float(val)
                    if score > 0:
                        parts = col.split('_')
                        week = int(parts[0].replace('week', ''))
                        judge = int(parts[1].replace('judge', ''))
                        records.append({
                            'celebrity': row['celebrity_name'],
                            'season': row['season'],
                            'placement': row['placement'],
                            'industry': row['celebrity_industry'],
                            'age': row['celebrity_age_during_season'],
                            'week': week,
                            'judge': judge,
                            'score': score
                        })
                except:
                    pass
    return pd.DataFrame(records)

print("Extracting scores...")
scores_df = extract_scores(df)
print(f"Total valid scores: {len(scores_df)}")

# 1. Overall score distribution with fitted distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
scores = scores_df['score'].values
ax.hist(scores, bins=30, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(scores.min(), scores.max(), 100)
# Fit normal
mu, std = stats.norm.fit(scores)
ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal(μ={mu:.1f}, σ={std:.1f})')
# Fit skewnorm
a, loc, scale = stats.skewnorm.fit(scores)
ax.plot(x, stats.skewnorm.pdf(x, a, loc, scale), 'g--', lw=2, label=f'SkewNorm(a={a:.1f})')
ax.set_xlabel('Score')
ax.set_ylabel('Density')
ax.set_title('Overall Judge Score Distribution')
ax.legend()

# 2. Score distribution by week
ax = axes[0, 1]
week_means = scores_df.groupby('week')['score'].agg(['mean', 'std']).reset_index()
ax.errorbar(week_means['week'], week_means['mean'], yerr=week_means['std'],
            marker='o', capsize=3, capthick=1)
ax.set_xlabel('Week')
ax.set_ylabel('Score (mean ± std)')
ax.set_title('Score Distribution by Week')

# 3. Score distribution by judge
ax = axes[1, 0]
judge_scores = [scores_df[scores_df['judge']==j]['score'].values for j in range(1, 5)]
bp = ax.boxplot(judge_scores, labels=[f'Judge {i}' for i in range(1, 5)])
ax.set_ylabel('Score')
ax.set_title('Score Distribution by Judge')

# 4. Age distribution
ax = axes[1, 1]
ages = df['celebrity_age_during_season'].dropna().values
ax.hist(ages, bins=20, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(ages.min(), ages.max(), 100)
mu, std = stats.norm.fit(ages)
ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal(μ={mu:.1f}, σ={std:.1f})')
a, b, loc, scale = stats.beta.fit(ages)
ax.plot(x, stats.beta.pdf(x, a, b, loc, scale), 'b--', lw=2, label=f'Beta(a={a:.1f}, b={b:.1f})')
ax.set_xlabel('Age')
ax.set_ylabel('Density')
ax.set_title('Celebrity Age Distribution')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/distributions_overview.png', dpi=150)
plt.close()

# 5. Industry breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
industry_counts = df['celebrity_industry'].value_counts()
ax.barh(industry_counts.index, industry_counts.values)
ax.set_xlabel('Count')
ax.set_title('Celebrity Count by Industry')

ax = axes[1]
industry_placement = df.groupby('celebrity_industry')['placement'].mean().sort_values()
ax.barh(industry_placement.index, industry_placement.values)
ax.set_xlabel('Average Placement (lower is better)')
ax.set_title('Average Placement by Industry')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/industry_analysis.png', dpi=150)
plt.close()

# 6. Placement analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
placements = df['placement'].dropna().values
ax.hist(placements, bins=range(1, int(placements.max())+2), density=True,
        alpha=0.7, edgecolor='black', align='left')
# Fit geometric distribution
p_geom = 1 / placements.mean()
x_geom = np.arange(1, int(placements.max())+1)
ax.plot(x_geom, stats.geom.pmf(x_geom, p_geom), 'ro-', label=f'Geometric(p={p_geom:.3f})')
ax.set_xlabel('Placement')
ax.set_ylabel('Density')
ax.set_title('Placement Distribution')
ax.legend()

# 7. Score vs Placement
ax = axes[0, 1]
avg_scores = scores_df.groupby('celebrity')['score'].mean().reset_index()
avg_scores.columns = ['celebrity', 'avg_score']
celeb_placement = df[['celebrity_name', 'placement']].drop_duplicates()
merged = avg_scores.merge(celeb_placement, left_on='celebrity', right_on='celebrity_name')
ax.scatter(merged['avg_score'], merged['placement'], alpha=0.5)
slope, intercept, r, p, se = stats.linregress(merged['avg_score'], merged['placement'])
x_line = np.linspace(merged['avg_score'].min(), merged['avg_score'].max(), 100)
ax.plot(x_line, slope*x_line + intercept, 'r-', label=f'r={r:.2f}, p={p:.2e}')
ax.set_xlabel('Average Score')
ax.set_ylabel('Placement')
ax.set_title('Average Score vs Placement')
ax.legend()

# 8. Season effects
ax = axes[1, 0]
season_scores = scores_df.groupby('season')['score'].agg(['mean', 'std']).reset_index()
ax.errorbar(season_scores['season'], season_scores['mean'], yerr=season_scores['std'],
            marker='o', capsize=3)
ax.set_xlabel('Season')
ax.set_ylabel('Score (mean ± std)')
ax.set_title('Score Distribution by Season')

# 9. Score progression for top vs bottom performers
ax = axes[1, 1]
top_celebs = df[df['placement'] <= 3]['celebrity_name'].values
bottom_celebs = df[df['placement'] >= df['placement'].quantile(0.75)]['celebrity_name'].values
top_prog = scores_df[scores_df['celebrity'].isin(top_celebs)].groupby('week')['score'].mean()
bottom_prog = scores_df[scores_df['celebrity'].isin(bottom_celebs)].groupby('week')['score'].mean()
ax.plot(top_prog.index, top_prog.values, 'go-', label='Top 3')
ax.plot(bottom_prog.index, bottom_prog.values, 'rs-', label='Bottom 25%')
ax.set_xlabel('Week')
ax.set_ylabel('Average Score')
ax.set_title('Score Progression: Top vs Bottom Performers')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/placement_analysis.png', dpi=150)
plt.close()

# 10. Judge agreement analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
judge_pairs = scores_df.pivot_table(index=['celebrity', 'week', 'season'],
                                     columns='judge', values='score').dropna()
if len(judge_pairs.columns) >= 2:
    corr_matrix = judge_pairs.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax,
                xticklabels=[f'J{i}' for i in corr_matrix.columns],
                yticklabels=[f'J{i}' for i in corr_matrix.index])
    ax.set_title('Judge Score Correlation')

# Score variance within same performance
ax = axes[1]
score_std = scores_df.groupby(['celebrity', 'week', 'season'])['score'].std().dropna()
ax.hist(score_std.values, bins=30, density=True, alpha=0.7, edgecolor='black')
mu, std = stats.norm.fit(score_std.values)
x = np.linspace(0, score_std.max(), 100)
ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
shape, loc, scale = stats.gamma.fit(score_std.values)
ax.plot(x, stats.gamma.pdf(x, shape, loc, scale), 'g--', lw=2,
        label=f'Gamma(k={shape:.2f})')
ax.set_xlabel('Standard Deviation of Judge Scores (per performance)')
ax.set_ylabel('Density')
ax.set_title('Judge Disagreement Distribution')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/judge_analysis.png', dpi=150)
plt.close()

# 11. Week-to-week score change distribution
fig, ax = plt.subplots(figsize=(10, 6))
celeb_week_scores = scores_df.groupby(['celebrity', 'week'])['score'].mean().reset_index()
celeb_week_scores = celeb_week_scores.sort_values(['celebrity', 'week'])
celeb_week_scores['score_change'] = celeb_week_scores.groupby('celebrity')['score'].diff()
changes = celeb_week_scores['score_change'].dropna().values

ax.hist(changes, bins=40, density=True, alpha=0.7, edgecolor='black')
mu, std = stats.norm.fit(changes)
x = np.linspace(changes.min(), changes.max(), 100)
ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
df_t, loc_t, scale_t = stats.t.fit(changes)
ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), 'g--', lw=2,
        label=f't-dist(df={df_t:.1f})')
ax.set_xlabel('Week-to-Week Score Change')
ax.set_ylabel('Density')
ax.set_title('Score Change Distribution (consecutive weeks)')
ax.legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/score_changes.png', dpi=150)
plt.close()

# 12. Country/region breakdown
fig, ax = plt.subplots(figsize=(10, 6))
country_counts = df['celebrity_homecountry/region'].value_counts().head(15)
ax.barh(country_counts.index, country_counts.values)
ax.set_xlabel('Count')
ax.set_title('Celebrity Home Country/Region')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/country_distribution.png', dpi=150)
plt.close()

# 13. Standalone judge score distribution with large buckets
fig, ax = plt.subplots(figsize=(12, 7))
scores = scores_df['score'].values
bins = np.arange(0, 11, 1)
ax.hist(scores, bins=bins, density=True, alpha=0.7, edgecolor='black', linewidth=1.2)
x = np.linspace(0, 10, 200)
mu, std = stats.norm.fit(scores)
ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2.5, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
a, loc, scale = stats.skewnorm.fit(scores)
ax.plot(x, stats.skewnorm.pdf(x, a, loc, scale), 'g--', lw=2.5, label=f'SkewNorm(a={a:.2f})')
ax.set_xlabel('Judge Score', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Overall Judge Score Distribution', fontsize=16)
ax.set_xticks(range(0, 11))
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/judge_score_distribution.png', dpi=150)
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nDataset: {len(df)} contestants across {df['season'].nunique()} seasons")
print(f"Score range: {scores_df['score'].min():.1f} - {scores_df['score'].max():.1f}")
print(f"Mean score: {scores_df['score'].mean():.2f} (std: {scores_df['score'].std():.2f})")
print(f"Age range: {ages.min():.0f} - {ages.max():.0f} (mean: {ages.mean():.1f})")

print("\nScore distribution fit tests (Kolmogorov-Smirnov):")
ks_norm = stats.kstest(scores, 'norm', args=(mu, std))
print(f"  Normal: KS={ks_norm.statistic:.3f}, p={ks_norm.pvalue:.3e}")

print("\nTop industries by count:")
print(industry_counts.head(5).to_string())

print("\nPlacement correlation with avg score: r={:.3f}".format(r))

print("\nFigures saved to exploratory/figures/")
