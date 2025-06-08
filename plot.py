import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Nuovi dati come stringhe (',' decimale, '-' o 'X' = fallimento) ---
sx_times_str = [
    '64,247', '-', '51,176', '25,648', '128,437', '-', '53,169', '146,366', '36,258',
    '23,604', '25,964', '13,647', '81,153', '12,498', '68,647', '20,082', '-',
    '135,951', '25,145', '15,386', '185,505', '64,96', '67,832', '-', '-', '33,569',
    '124,486', '134,58', '65,839', '84,792', '-', '37,226', '-', '64,595', '22,889',
    '-', '100,065', '-', '55,841', '75,193', '98,688', '196,802', '23,727', '65,348',
    '-', '-', '27,39', '163,195', '-', '31,04'
]
dx_times_str = [
    '75,594', '-', '61,914', '22,949', '124,643', '-', '84,239', '-', '37,105',
    '-', '114,545', '21,834', '92,929', '13,267', '95,105', '12,499', '-', '74,945',
    '23,445', '15,391', '81,137', '67,803', '60,693', '-', '94,44', '33,73',
    '-', '32,683', '54,558', '-', '59,676', '34,109', '-', '114,167', '26,942',
    '-', '134,103', '-', '-', '127,292', '43,403', '86,945', '15,272', '145,774',
    '-', '-', '23,581', '124,446', '-', '32,916'
]

# parsing: float o NaN
def parse_times(lst):
    out = []
    for v in lst:
        if v in ['-', 'X']:
            out.append(np.nan)
        else:
            out.append(float(v.replace(',', '.')))
    return np.array(out)

sx_times = parse_times(sx_times_str)
dx_times = parse_times(dx_times_str)

# --- coordinate e distanze tra punti successivi ---
coords = np.array([
    [-3, 7], [6, 1], [-6, -7], [1, -4], [-4, 8], [7, 2], [0, -3], [8,  3],
    [-7, 4], [-2, 3], [4, -1], [3, -6], [-6, 1], [-8, 3], [-1, -2], [2, -2],
    [6, 8], [-8, -8], [-2, -4], [2, -5], [3,  7], [0,  2], [-4, -6], [8,  6],
    [-3, -1], [7, -8], [1,  4], [-6, -1], [6, -3], [-1,  6], [-5, -8], [8, -2],
    [-8, 5], [5, -5], [-2, -8], [-7, -3], [2,  8], [-4, 5], [6, -6], [-8, -1],
    [1, -8], [3, 7], [-3, 8], [4, -5], [-6, 7], [7, -4], [0, -8], [-1, 7],
    [5, 3], [-7, 3]
])
dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)

# --- velocità: includi solo segmenti validi in ENTRAMBI ---
valid_seg = (~np.isnan(sx_times[:-1])) & (~np.isnan(dx_times[:-1]))
vel_sx = dists[valid_seg] / sx_times[:-1][valid_seg]
vel_dx = dists[valid_seg] / dx_times[:-1][valid_seg]

# --- fallimenti totali ---
fail_sx = np.isnan(sx_times).sum()
fail_dx = np.isnan(dx_times).sum()

# --- Plot settings ---
sns.set_style('darkgrid')

# 1) Tempi: violin + strip
df_times = pd.DataFrame({
    'Tempo (s)': np.concatenate([sx_times[~np.isnan(sx_times)],
                                 dx_times[~np.isnan(dx_times)]]),
    'Config': ['SX']*np.count_nonzero(~np.isnan(sx_times)) +
              ['DX']*np.count_nonzero(~np.isnan(dx_times))
})
plt.figure(figsize=(8,6))
sns.violinplot(x='Config', y='Tempo (s)', data=df_times,
               inner=None, palette=['#1f77b4','#ff7f0e'])
sns.stripplot(x='Config', y='Tempo (s)', data=df_times,
              color='k', size=4, jitter=True, alpha=0.7)
plt.title('Execution time distribution')
plt.savefig('violinplot_tempi.eps', format='eps')
plt.show()

# 2) Fallimenti: bar plot
plt.figure(figsize=(6,5))
bars = plt.bar(['SX','DX'], [fail_sx, fail_dx],
               color=['#1f77b4','#ff7f0e'], edgecolor='k')
plt.title('Failures number')
plt.ylim(0, max(fail_sx,fail_dx)+3)
for bar, f in zip(bars, [fail_sx, fail_dx]):
    plt.text(bar.get_x()+bar.get_width()/2, f+0.2, str(f),
             ha='center', va='bottom', fontweight='bold')
plt.savefig('barplot_fallimenti.eps', format='eps')
plt.show()

# 3) Velocità: violin + strip
df_vel = pd.DataFrame({
    'Velocità (unit/s)': np.concatenate([vel_sx, vel_dx]),
    'Config': ['SX']*len(vel_sx) + ['DX']*len(vel_dx)
})
plt.figure(figsize=(8,6))
sns.violinplot(x='Config', y='Velocità (unit/s)', data=df_vel,
               inner=None, palette=['#1f77b4','#ff7f0e'])
sns.stripplot(x='Config', y='Velocità (unit/s)', data=df_vel,
              color='k', size=4, jitter=True, alpha=0.7)
plt.title('Average speed distribution')
plt.savefig('violinplot_velocita.eps', format='eps')
plt.show()
