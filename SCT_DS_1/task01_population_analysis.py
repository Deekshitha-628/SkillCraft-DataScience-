# SkillCraft Technology - Task 01
# Population Distribution Analysis using World Bank Data
# Dataset: https://data.worldbank.org/indicator/SP.POP.TOTL

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Style Setup ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.labelcolor':  'white',
    'xtick.color':      'white',
    'ytick.color':      'white',
    'text.color':       'white',
    'grid.color':       '#2a2a4a',
    'axes.titlesize':   14,
    'axes.titleweight': 'bold',
})

# ── Load Data ─────────────────────────────────────────────────────────────────
# Download CSV from: https://data.worldbank.org/indicator/SP.POP.TOTL
# Place the downloaded CSV in the same folder as this script
# The file is usually named: API_SP.POP.TOTL_DS2_en_csv_v2_XXXXXX.csv

try:
    # Try loading the downloaded World Bank CSV
    df = pd.read_csv('API_SP.POP.TOTL_DS2_en_csv_v2.csv', skiprows=4)
    df2022 = df[['Country Name', '2022']].dropna()
    df2022.columns = ['Country', 'Population']
    df2022['Population'] = df2022['Population'].astype(float)
    # Remove aggregates (World, regions, income groups)
    exclude = ['World', 'income', 'OECD', 'Euro', 'Arab', 'Africa',
               'Asia', 'Europe', 'America', 'Middle', 'South', 'North',
               'Pacific', 'Caribbean', 'Sahara', 'IDA', 'IBRD', 'Low',
               'Upper', 'High', 'fragile', 'small', 'Heavily', 'Post',
               'Pre', 'Least', 'blend', 'Late', 'Early', 'dividend']
    mask = ~df2022['Country'].str.contains('|'.join(exclude), case=False, na=False)
    df2022 = df2022[mask].reset_index(drop=True)
    print(f"✅ Loaded {len(df2022)} countries from CSV")

except FileNotFoundError:
    # Fallback: use built-in sample data (World Bank 2022 values)
    print("⚠️  CSV not found — using built-in World Bank 2022 data")
    data = {
        'Country': [
            'India','China','United States','Indonesia','Pakistan',
            'Brazil','Nigeria','Bangladesh','Russia','Ethiopia',
            'Mexico','Japan','Philippines','Egypt','DR Congo',
            'Vietnam','Iran','Turkey','Germany','Thailand',
            'United Kingdom','France','Tanzania','South Africa','Kenya',
            'Myanmar','Colombia','Spain','Uganda','Argentina',
            'Algeria','Sudan','Iraq','Ukraine','Canada',
            'Morocco','Saudi Arabia','Uzbekistan','Peru','Afghanistan',
            'Venezuela','Malaysia','Ghana','Mozambique','Angola',
            'Yemen','Nepal','Cameroon','Madagascar','Australia',
            'Côte d\'Ivoire','Niger','Sri Lanka','Burkina Faso','Mali',
            'Romania','Chile','Kazakhstan','Netherlands','Ecuador',
            'Guatemala','Cambodia','Zimbabwe','Senegal','Rwanda',
            'Chad','Bolivia','Zambia','Somalia','Portugal',
            'Belgium','Sweden','Jordan','Honduras','Papua New Guinea',
            'UAE','Tajikistan','Hungary','Serbia','Laos',
            'New Zealand','Singapore','Denmark','Finland','Norway',
            'Ireland','Kuwait','Panama','Croatia','Costa Rica',
            'New Zealand','Bahrain','Uruguay','Jamaica','Qatar',
        ],
        'Population': [
            1428628000,1425671000,338289000,275501000,235825000,
            215313000,218541000,169356000,144236000,123380000,
            127504000,125124000,115560000,104258000,99010000,
            97340000,86871000,84680000,83200000,71697000,
            67749000,65658000,63588000,59893000,54027000,
            53798000,51874000,47486000,47958000,45510000,
            44903000,43992000,41179000,43531000,38245000,
            37458000,36408000,35300000,32972000,40099000,
            28302000,33574000,32395000,32790000,34503000,
            33696000,29609000,27915000,28916000,25978000,
            27053000,25252000,22156000,22673000,22395000,
            19038000,19603000,19000000,17618000,17797000,
            16858000,16589000,15092000,17316000,13600000,
            17413000,11832000,19473000,17065000,10247000,
            11590000,10549000,10200000,9972000,10329000,
            9890000,9750000,9710000,6764000,9593000,
            9441000,5637000,5882000,5540000,5434000,
            5056000,4310000,4351000,4008000,5153000,
            5099000,1463000,4351000,2827000,2695000,
        ]
    }
    df2022 = pd.DataFrame(data)
    print(f"✅ Using built-in data: {len(df2022)} countries")

df2022 = df2022.sort_values('Population', ascending=False).reset_index(drop=True)
df2022['Pop_Millions'] = df2022['Population'] / 1e6

# ── CHART 1: Bar Chart — Top 15 Countries ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#1a1a2e')

top15 = df2022.head(15)
palette = sns.color_palette("Blues_r", n_colors=15)

bars = sns.barplot(
    data=top15,
    x='Country',
    y='Pop_Millions',
    palette=palette,
    ax=ax,
    edgecolor='white',
    linewidth=0.5
)

# Add value labels on top of each bar
for bar, val in zip(bars.patches, top15['Pop_Millions']):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        f'{val:.0f}M',
        ha='center', va='bottom',
        fontsize=9, color='white', fontweight='bold'
    )

ax.set_title('Top 15 Most Populous Countries (2022)', pad=20, color='white')
ax.set_xlabel('Country', labelpad=10)
ax.set_ylabel('Population (Millions)', labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=10)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}M'))

plt.tight_layout()
plt.savefig('bar_chart_top15.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
print("✅ Saved: bar_chart_top15.png")
plt.show()

# ── CHART 2: Histogram — Population Distribution (Log Scale) ─────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#1a1a2e')

log_pops = np.log10(df2022['Population'].clip(lower=1))
n, bins, patches = ax.hist(log_pops, bins=25, edgecolor='white',
                            linewidth=0.6, color='#5DCAA5')

# Color bars by population range
colors_hist = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(patches)))
for patch, color in zip(patches, colors_hist):
    patch.set_facecolor(color)

# Custom x-axis labels (log scale → readable values)
tick_vals  = [4, 5, 6, 7, 8, 9]
tick_labels = ['10K', '100K', '1M', '10M', '100M', '1B']
ax.set_xticks(tick_vals)
ax.set_xticklabels(tick_labels, fontsize=11)

ax.set_title('Distribution of Country Populations — Log Scale (2022)',
             pad=20, color='white')
ax.set_xlabel('Population (log scale)', labelpad=10)
ax.set_ylabel('Number of Countries', labelpad=10)

# Annotation
ax.annotate('Most countries have\n< 50M people',
            xy=(7.5, n.max() * 0.7),
            fontsize=10, color='#FAC775',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a4a',
                      edgecolor='#FAC775', linewidth=0.8))

plt.tight_layout()
plt.savefig('histogram_population.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
print("✅ Saved: histogram_population.png")
plt.show()

# ── CHART 3: Horizontal Bar — Top 10 with % share ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#1a1a2e')

top10 = df2022.head(10).copy()
total_world = 8000000000
top10['Share'] = (top10['Population'] / total_world * 100).round(2)
top10_sorted = top10.sort_values('Population')

colors_h = sns.color_palette("mako", n_colors=10)
bars_h = ax.barh(top10_sorted['Country'], top10_sorted['Pop_Millions'],
                 color=colors_h, edgecolor='white', linewidth=0.5)

for bar, share in zip(bars_h, top10_sorted['Share']):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f'{share:.1f}% of world',
            va='center', fontsize=9, color='#FAC775')

ax.set_title('Top 10 Countries by Population — Share of World (2022)',
             pad=20, color='white')
ax.set_xlabel('Population (Millions)', labelpad=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}M'))

plt.tight_layout()
plt.savefig('horizontal_bar_share.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
print("✅ Saved: horizontal_bar_share.png")
plt.show()

# ── Summary Stats ─────────────────────────────────────────────────────────────
print("\n── Summary Statistics ───────────────────────")
print(f"Total countries analysed : {len(df2022)}")
print(f"Most populous            : {df2022.iloc[0]['Country']} "
      f"({df2022.iloc[0]['Pop_Millions']:.0f}M)")
print(f"Least populous           : {df2022.iloc[-1]['Country']} "
      f"({df2022.iloc[-1]['Pop_Millions']:.2f}M)")
print(f"Median population        : {df2022['Pop_Millions'].median():.2f}M")
print(f"Mean population          : {df2022['Pop_Millions'].mean():.2f}M")
print("─────────────────────────────────────────────")
print("All 3 charts saved! Upload to GitHub & LinkedIn.")
