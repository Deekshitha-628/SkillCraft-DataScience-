# SkillCraft Technology - Task 04
# Traffic Accident Analysis - US Accidents Dataset
# Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor':'#1a1a2e','axes.facecolor':'#16213e',
    'axes.labelcolor':'white','xtick.color':'white',
    'ytick.color':'white','text.color':'white',
    'grid.color':'#2a2a4a','axes.titlesize':13,
    'axes.titleweight':'bold','axes.titlecolor':'white'
})
COLORS = ['#5DCAA5','#EF9F27','#378ADD','#D4537E','#7F77DD',
          '#F4D03F','#A569BD','#45B39D','#EC7063','#5DADE2']

# ── Load Data (1M rows sample) ────────────────────────────────────────────────
print("Loading US Accidents dataset (1M rows)...")
df = pd.read_csv('US_Accidents_March23.csv', nrows=1000000)
df.columns = df.columns.str.strip()
print(f"✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ── Clean & Parse ─────────────────────────────────────────────────────────────
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hour']       = df['Start_Time'].dt.hour
df['Month']      = df['Start_Time'].dt.month
df['DayOfWeek']  = df['Start_Time'].dt.day_name()
df['Year']       = df['Start_Time'].dt.year

df['Temperature(F)']  = pd.to_numeric(df['Temperature(F)'],  errors='coerce')
df['Visibility(mi)']  = pd.to_numeric(df['Visibility(mi)'],  errors='coerce')
df['Wind_Speed(mph)'] = pd.to_numeric(df['Wind_Speed(mph)'], errors='coerce')
df['Humidity(%)']     = pd.to_numeric(df['Humidity(%)'],     errors='coerce')

print(f"   Severity distribution:\n{df['Severity'].value_counts().to_string()}")
print(f"   Date range: {df['Start_Time'].min()} → {df['Start_Time'].max()}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Overview Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('US Traffic Accident Analysis — Overview Dashboard',
             fontsize=18, fontweight='bold', color='white', y=0.98)
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# Plot 1: Accidents by Severity
ax1 = fig.add_subplot(gs[0, 0])
sev = df['Severity'].value_counts().sort_index()
bars = ax1.bar(sev.index.astype(str), sev.values,
               color=COLORS[:4], edgecolor='white', linewidth=0.5)
for b, v in zip(bars, sev.values):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+500,
             f'{v:,}', ha='center', fontsize=9,
             fontweight='bold', color='white')
ax1.set_title('Accidents by Severity (1-4)')
ax1.set_xlabel('Severity Level')
ax1.set_ylabel('Number of Accidents')

# Plot 2: Accidents by Hour
ax2 = fig.add_subplot(gs[0, 1])
hourly = df['Hour'].value_counts().sort_index()
ax2.plot(hourly.index, hourly.values, color='#5DCAA5',
         linewidth=2, marker='o', markersize=4)
ax2.fill_between(hourly.index, hourly.values,
                 alpha=0.3, color='#5DCAA5')
ax2.set_title('Accidents by Hour of Day')
ax2.set_xlabel('Hour (0-23)')
ax2.set_ylabel('Number of Accidents')
ax2.set_xticks(range(0, 24, 2))

# Plot 3: Top 10 States
ax3 = fig.add_subplot(gs[0, 2])
top_states = df['State'].value_counts().head(10)
bars3 = ax3.barh(top_states.index[::-1], top_states.values[::-1],
                 color=COLORS, edgecolor='white', linewidth=0.5)
for b, v in zip(bars3, top_states.values[::-1]):
    ax3.text(b.get_width()+200, b.get_y()+b.get_height()/2,
             f'{v:,}', va='center', fontsize=8,
             fontweight='bold', color='white')
ax3.set_title('Top 10 States by Accidents')
ax3.set_xlabel('Number of Accidents')

# Plot 4: Accidents by Day of Week
ax4 = fig.add_subplot(gs[1, 0])
day_order = ['Monday','Tuesday','Wednesday','Thursday',
             'Friday','Saturday','Sunday']
dow = df['DayOfWeek'].value_counts().reindex(day_order)
bars4 = ax4.bar(range(7), dow.values,
                color=COLORS[:7], edgecolor='white', linewidth=0.5)
ax4.set_xticks(range(7))
ax4.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
for b, v in zip(bars4, dow.values):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+200,
             f'{v:,}', ha='center', fontsize=8,
             fontweight='bold', color='white')
ax4.set_title('Accidents by Day of Week')
ax4.set_ylabel('Number of Accidents')

# Plot 5: Accidents by Month
ax5 = fig.add_subplot(gs[1, 1])
monthly = df['Month'].value_counts().sort_index()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
bars5 = ax5.bar(range(1, len(monthly)+1), monthly.values,
                color=COLORS[:len(monthly)], edgecolor='white', linewidth=0.5)
ax5.set_xticks(range(1, len(monthly)+1))
ax5.set_xticklabels(month_names[:len(monthly)], rotation=45)
ax5.set_title('Accidents by Month')
ax5.set_ylabel('Number of Accidents')

# Plot 6: Weather Condition Top 10
ax6 = fig.add_subplot(gs[1, 2])
weather = df['Weather_Condition'].value_counts().head(8)
bars6 = ax6.barh(weather.index[::-1], weather.values[::-1],
                 color=COLORS, edgecolor='white', linewidth=0.5)
for b, v in zip(bars6, weather.values[::-1]):
    ax6.text(b.get_width()+200, b.get_y()+b.get_height()/2,
             f'{v:,}', va='center', fontsize=8,
             fontweight='bold', color='white')
ax6.set_title('Top Weather Conditions')
ax6.set_xlabel('Number of Accidents')

plt.savefig('chart1_overview_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("\n✅ Saved: chart1_overview_dashboard.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Severity vs Conditions
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Severity vs Road & Weather Conditions',
             fontsize=16, fontweight='bold', color='white')

# Severity vs Sunrise/Sunset
ax = axes[0]
ss = df.groupby('Sunrise_Sunset')['Severity'].mean()
bars = ax.bar(ss.index, ss.values,
              color=['#EF9F27','#378ADD'], edgecolor='white', width=0.4)
for b, v in zip(bars, ss.values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f'{v:.2f}', ha='center', fontsize=12,
            fontweight='bold', color='white')
ax.set_title('Avg Severity: Day vs Night')
ax.set_ylabel('Average Severity')
ax.set_ylim(0, 3.5)

# Severity vs Temperature bins
ax2 = axes[1]
df['TempBin'] = pd.cut(df['Temperature(F)'],
                        bins=[-50, 0, 32, 60, 80, 100, 150],
                        labels=['<0°F','0-32°F','32-60°F',
                                '60-80°F','80-100°F','>100°F'])
tb = df.groupby('TempBin', observed=True)['Severity'].mean()
bars2 = ax2.bar(tb.index, tb.values,
                color=COLORS[:len(tb)], edgecolor='white')
for b, v in zip(bars2, tb.values):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f'{v:.2f}', ha='center', fontsize=10,
             fontweight='bold', color='white')
ax2.set_title('Avg Severity by Temperature')
ax2.set_ylabel('Average Severity')
ax2.set_xticklabels(tb.index, rotation=30)
ax2.set_ylim(0, 3.5)

# Severity vs Visibility bins
ax3 = axes[2]
df['VisBin'] = pd.cut(df['Visibility(mi)'],
                       bins=[0, 1, 3, 5, 10, 100],
                       labels=['<1mi','1-3mi','3-5mi','5-10mi','>10mi'])
vb = df.groupby('VisBin', observed=True)['Severity'].mean()
bars3 = ax3.bar(vb.index, vb.values,
                color=COLORS[:len(vb)], edgecolor='white')
for b, v in zip(bars3, vb.values):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f'{v:.2f}', ha='center', fontsize=10,
             fontweight='bold', color='white')
ax3.set_title('Avg Severity by Visibility')
ax3.set_ylabel('Average Severity')
ax3.set_ylim(0, 3.5)

plt.tight_layout()
plt.savefig('chart2_severity_conditions.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Saved: chart2_severity_conditions.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Heatmap: Hour vs Day of Week
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')

df['DayNum'] = df['Start_Time'].dt.dayofweek
pivot = df.groupby(['DayNum','Hour']).size().unstack(fill_value=0)
pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.1,
            cbar_kws={'shrink':0.8, 'label':'Accident Count'})
ax.set_title('Accident Heatmap — Hour vs Day of Week',
             pad=15, fontsize=15)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Day of Week')
plt.tight_layout()
plt.savefig('chart3_heatmap_hour_day.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Saved: chart3_heatmap_hour_day.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Road Feature Impact
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#1a1a2e')

road_features = ['Amenity','Bump','Crossing','Give_Way','Junction',
                 'No_Exit','Railway','Station','Stop','Traffic_Signal']
feature_counts = []
for f in road_features:
    if f in df.columns:
        count = df[df[f]==True].shape[0]
        feature_counts.append((f, count))

feat_df = pd.DataFrame(feature_counts, columns=['Feature','Count'])
feat_df = feat_df.sort_values('Count', ascending=True)
bars = ax.barh(feat_df['Feature'], feat_df['Count'],
               color=COLORS[:len(feat_df)], edgecolor='white', linewidth=0.5)
for b, v in zip(bars, feat_df['Count']):
    ax.text(b.get_width()+100, b.get_y()+b.get_height()/2,
            f'{v:,}', va='center', fontsize=9,
            fontweight='bold', color='white')
ax.set_title('Accidents Near Road Features')
ax.set_xlabel('Number of Accidents')
plt.tight_layout()
plt.savefig('chart4_road_features.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Saved: chart4_road_features.png")
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Key Insights ─────────────────────────────")
print(f"Total accidents analysed : {len(df):,}")
print(f"Most dangerous state     : {df['State'].value_counts().idxmax()}")
print(f"Peak accident hour       : {df['Hour'].value_counts().idxmax()}:00")
print(f"Most common severity     : {df['Severity'].value_counts().idxmax()}")
print(f"Most common weather      : {df['Weather_Condition'].value_counts().idxmax()}")
night = df[df['Sunrise_Sunset']=='Night']['Severity'].mean()
day   = df[df['Sunrise_Sunset']=='Day']['Severity'].mean()
print(f"Avg severity (Day)       : {day:.2f}")
print(f"Avg severity (Night)     : {night:.2f}")
print("─────────────────────────────────────────────")
print("All 4 charts saved! Upload to GitHub & LinkedIn.")