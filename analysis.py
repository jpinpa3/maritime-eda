import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. LOAD & CLEAN ───────────────────────────────────────────────────────────
print("Loading AIS data...")
df = pd.read_csv('ais_data.csv')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = df.dropna(subset=['mmsi'])
df['sog'] = pd.to_numeric(df['sog'], errors='coerce')
df['shiptype'] = df['shiptype'].fillna('Unknown')
df['navigationalstatus'] = df['navigationalstatus'].fillna('Unknown')
print(f"Records: {len(df):,} | Unique vessels: {df['mmsi'].nunique():,}")

# ── 2. PLOTS ──────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 11), facecolor='#0a0a0f')
fig.suptitle('Maritime Traffic EDA — AIS Dataset\nJavier Piñeiro | Data Science Portfolio',
             fontsize=15, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

CYAN   = '#00e5ff'
PURPLE = '#7b61ff'
GREEN  = '#00ff9d'

# Plot 1: Ship type distribution
ax1 = fig.add_subplot(gs[0, 0])
top_types = df['shiptype'].value_counts().head(10)
bars = ax1.barh(top_types.index[::-1], top_types.values[::-1], color=CYAN, edgecolor='none')
ax1.set_title('Top 10 Vessel Types', color='white', fontsize=11, pad=10)
ax1.set_xlabel('Number of Records', color='#aaaaaa', fontsize=9)
ax1.tick_params(colors='#aaaaaa', labelsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#333344')
ax1.spines['bottom'].set_color('#333344')
for bar in bars:
    bar.set_alpha(0.85)

# Plot 2: Speed Over Ground distribution
ax2 = fig.add_subplot(gs[0, 1])
speeds = df['sog'].dropna()
speeds = speeds[(speeds > 0) & (speeds < 35)]
ax2.hist(speeds, bins=40, color=PURPLE, edgecolor='none', alpha=0.85)
ax2.axvline(speeds.mean(), color=CYAN, linestyle='--', linewidth=1.5, label=f'Mean: {speeds.mean():.1f} kn')
ax2.axvline(speeds.median(), color=GREEN, linestyle='--', linewidth=1.5, label=f'Median: {speeds.median():.1f} kn')
ax2.set_title('Speed Over Ground Distribution', color='white', fontsize=11, pad=10)
ax2.set_xlabel('Speed (knots)', color='#aaaaaa', fontsize=9)
ax2.set_ylabel('Frequency', color='#aaaaaa', fontsize=9)
ax2.tick_params(colors='#aaaaaa', labelsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#333344')
ax2.spines['bottom'].set_color('#333344')
ax2.legend(fontsize=8, facecolor='#1a1a26', edgecolor='#333344', labelcolor='white')

# Plot 3: Navigational status
ax3 = fig.add_subplot(gs[1, 0])
status = df['navigationalstatus'].value_counts().head(8)
colors = [CYAN, PURPLE, GREEN, '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#ff922b']
wedges, texts, autotexts = ax3.pie(
    status.values, labels=None, autopct='%1.1f%%',
    colors=colors[:len(status)], startangle=140,
    pctdistance=0.75, textprops={'color': 'white', 'fontsize': 7}
)
ax3.set_title('Navigational Status', color='white', fontsize=11, pad=10)
ax3.legend(status.index, loc='lower left', fontsize=7,
           facecolor='#1a1a26', edgecolor='#333344', labelcolor='white')

# Plot 4: Vessel size (length distribution)
ax4 = fig.add_subplot(gs[1, 1])
lengths = df['length'].dropna()
lengths = lengths[(lengths > 5) & (lengths < 400)]
ax4.hist(lengths, bins=40, color=GREEN, edgecolor='none', alpha=0.85)
ax4.axvline(lengths.mean(), color=CYAN, linestyle='--', linewidth=1.5, label=f'Mean: {lengths.mean():.0f} m')
ax4.set_title('Vessel Length Distribution', color='white', fontsize=11, pad=10)
ax4.set_xlabel('Length (metres)', color='#aaaaaa', fontsize=9)
ax4.set_ylabel('Frequency', color='#aaaaaa', fontsize=9)
ax4.tick_params(colors='#aaaaaa', labelsize=8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color('#333344')
ax4.spines['bottom'].set_color('#333344')
ax4.legend(fontsize=8, facecolor='#1a1a26', edgecolor='#333344', labelcolor='white')

plt.savefig('maritime_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
print("\n✅ Chart saved: maritime_analysis.png")
plt.show()

# ── 3. SUMMARY ────────────────────────────────────────────────────────────────
print("\n── SUMMARY STATISTICS ───────────────────────────────────────")
print(f"Total records      : {len(df):,}")
print(f"Unique vessels     : {df['mmsi'].nunique():,}")
print(f"Avg speed (knots)  : {df['sog'].mean():.2f}")
print(f"Top vessel type    : {df['shiptype'].value_counts().index[0]}")
print(f"Avg vessel length  : {df['length'].mean():.1f} m")