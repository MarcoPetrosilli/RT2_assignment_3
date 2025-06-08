import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

# Coordinate dei punti
coords = np.array([
    [-3, 7], [6, 1], [-6, -7], [1, -4], [-4, 8], [7, 2], [0, -3], [8, 3],
    [-7, 4], [-2, 3], [4, -1], [3, -6], [-6, 1], [-8, 3], [-1, -2], [2, -2],
    [6, 8], [-8, -8], [-2, -4], [2, -5], [3, 7], [0, 2], [-4, -6], [8, 6],
    [-3, -1], [7, -8], [1, 4], [-6, -1], [6, -3], [-1, 6], [-5, -8], [8, -2],
    [-8, 5], [5, -5], [-2, -8], [-7, -3], [2, 8], [-4, 5], [6, -6], [-8, -1],
    [1, -8], [3, 7], [-3, 8], [4, -5], [-6, 7], [7, -4], [0, -8], [-1, 7],
    [5, 3], [-7, 3]
])

# Tempi con fallimenti
sx_times = [
    '64,247', '-', '51,176', '25,648', '128,437', '-', '53,169', '146,366', '36,258',
    '23,604', '25,964', '13,647', '81,153', '12,498', '68,647', '20,082', '-',
    '135,951', '25,145', '15,386', '185,505', '64,96', '67,832', '-', '-', '33,569',
    '124,486', '134,58', '65,839', '84,792', '-', '37,226', '-', '64,595', '22,889',
    '-', '100,065', '-', '55,841', '75,193', '98,688', '196,802', '23,727', '65,348',
    '-', '-', '27,39', '163,195', '-', '31,04'
]
dx_times = [
    '75,594', '-', '61,914', '22,949', '124,643', '-', '84,239', '-', '37,105',
    '-', '114,545', '21,834', '92,929', '13,267', '95,105', '12,499', '-', '74,945',
    '23,445', '15,391', '81,137', '67,803', '60,693', '-', '94,44', '33,73',
    '-', '32,683', '54,558', '-', '59,676', '34,109', '-', '114,167', '26,942',
    '-', '134,103', '-', '-', '127,292', '43,403', '86,945', '15,272', '145,774',
    '-', '-', '23,581', '124,446', '-', '32,916'
]

# Parsing helper
def parse_list(lst):
    return np.array([float(v.replace(',', '.')) if v not in ['-', 'X'] else np.nan for v in lst])

data_sx = parse_list(sx_times)
data_dx = parse_list(dx_times)

# Filtri validi
valid_sx = ~np.isnan(data_sx)
valid_dx = ~np.isnan(data_dx)
both_valid = valid_sx & valid_dx

# 1. Statistiche descrittive tempi
mean_sx, std_sx = np.nanmean(data_sx), np.nanstd(data_sx, ddof=1)
mean_dx, std_dx = np.nanmean(data_dx), np.nanstd(data_dx, ddof=1)
print(f"Media SX: {mean_sx:.2f}, Dev std SX: {std_sx:.2f}")
print(f"Media DX: {mean_dx:.2f}, Dev std DX: {std_dx:.2f}\n")

# 2. Welch's t-test sui tempi
#t_stat, p_two = stats.ttest_ind(data_sx[both_valid], data_dx[both_valid], equal_var=False)
#print(f"t-statistic: {t_stat:.3f}")
#print(f"p-value two-sided (H0: μ_SX = μ_DX): {p_two:.3f}")
#print(f"p-value one-sided (H0: μ_SX >= μ_DX): {p_two/2 if t_stat < 0 else 1 - p_two/2:.3f}")
#print(f"p-value one-sided (H0: μ_SX <= μ_DX): {p_two/2 if t_stat > 0 else 1 - p_two/2:.3f}\n")

# 2. Paired t-test sui tempi
paired_data_sx = data_sx[both_valid]
paired_data_dx = data_dx[both_valid]

t_stat, p_two = stats.ttest_rel(paired_data_sx, paired_data_dx)
print(f"[Paired t-test]")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value two-sided (H0: μ_SX = μ_DX): {p_two:.3f}")
print(f"p-value one-sided (H0: μ_SX >= μ_DX): {p_two/2 if t_stat < 0 else 1 - p_two/2:.3f}")
print(f"p-value one-sided (H0: μ_SX <= μ_DX): {p_two/2 if t_stat > 0 else 1 - p_two/2:.3f}\n")


# 3. Proporzioni di fallimento
fail_sx = np.sum(np.isnan(data_sx))
fail_dx = np.sum(np.isnan(data_dx))
n = np.array([len(data_sx), len(data_dx)])
k = np.array([fail_sx, fail_dx])

z_stat, p_two_prop = proportions_ztest(k, n)
print(f"z-statistic prop: {z_stat:.3f}")
print(f"p-value two-sided (H0: fail_rate_SX = fail_rate_DX): {p_two_prop:.3f}")
print(f"p-value one-sided (H0: fail_rate_SX >= fail_rate_DX): {proportions_ztest(k, n, alternative='smaller')[1]:.3f}")
print(f"p-value one-sided (H0: fail_rate_SX <= fail_rate_DX): {proportions_ztest(k, n, alternative='larger')[1]:.3f}\n")

# 4. Chi-square test
table = np.array([[fail_sx, n[0] - fail_sx], [fail_dx, n[1] - fail_dx]])
chi2, p_chi, _, _ = chi2_contingency(table)
print(f"Chi2: {chi2:.3f}, p-value (H0: independence between config and failure): {p_chi:.3f}\n")

# 5. Velocità media
dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
tsx = data_sx[:-1]
tdx = data_dx[:-1]
valid_vel = ~np.isnan(tsx) & ~np.isnan(tdx)

vel_sx = dists[valid_vel] / tsx[valid_vel]
vel_dx = dists[valid_vel] / tdx[valid_vel]

mean_v_sx, std_v_sx = np.mean(vel_sx), np.std(vel_sx, ddof=1)
mean_v_dx, std_v_dx = np.mean(vel_dx), np.std(vel_dx, ddof=1)
print(f"Mean vel SX: {mean_v_sx:.4f}, SD: {std_v_sx:.4f}")
print(f"Mean vel DX: {mean_v_dx:.4f}, SD: {std_v_dx:.4f}")

# t-test velocità
#t_v, p_two_v = stats.ttest_ind(vel_sx, vel_dx, equal_var=False)
#print(f"t-vel: {t_v:.3f}")
#print(f"p-value two-sided (H0: vel_SX = vel_DX): {p_two_v:.3f}")
#print(f"p-value one-sided (H0: vel_SX >= vel_DX): {p_two_v/2 if t_v < 0 else 1 - p_two_v/2:.3f}")
#print(f"p-value one-sided (H0: vel_SX <= vel_DX): {p_two_v/2 if t_v > 0 else 1 - p_two_v/2:.3f}")

# Paired t-test velocità
t_v, p_two_v = stats.ttest_rel(vel_sx, vel_dx)
print(f"[Paired t-test sulle velocità]")
print(f"t-statistic: {t_v:.3f}")
print(f"p-value two-sided (H0: vel_SX = vel_DX): {p_two_v:.3f}")
print(f"p-value one-sided (H0: vel_SX >= vel_DX): {p_two_v/2 if t_v < 0 else 1 - p_two_v/2:.3f}")
print(f"p-value one-sided (H0: vel_SX <= vel_DX): {p_two_v/2 if t_v > 0 else 1 - p_two_v/2:.3f}")