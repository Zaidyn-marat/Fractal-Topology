import numpy as np
import matplotlib.pyplot as plt

# Эксперименттік деректер (实验数据)
x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Qalındyǵy (cm)
pb_lnz = np.array([0, -0.0018, -0.15, -0.74, -0.78, -0.81, -0.85])  # Qorǵan – Pb
cu_lnz = np.array([0, -0.11, -0.71, -0.75, -0.82, -0.85, -0.90])    # Qorǵan – Cu

# Sýretti sızbaǵa túsiý funktsııasy
def fit_and_plot(x, y, label, color):
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    mu = abs(slope)
    fit_line = np.polyval(coeffs, x)

    plt.scatter(x, y, color=color, label=f'{label} nuktelerı')
    plt.plot(x, fit_line, '--', color=color, label=f'{label} sızyǵy: μ = {mu:.2f} cm⁻¹')
    return mu

# Pb - Qorǵanynyń sızbasy
plt.figure(figsize=(8, 5))
plt.title("Gamma-sáýlegenıń sińiriliýi – Qorǵan: Qurǵashyn", fontsize=14)
plt.xlabel("Qalynǵdyǵy (cm)", fontsize=12)
plt.ylabel("ln(Z)", fontsize=12)
plt.grid(True, alpha=0.3)
mu_pb = fit_and_plot(x, pb_lnz, 'Qurǵashyn (Pb)', 'blue')
plt.legend()
plt.tight_layout()
plt.show()

# Cu - Qorǵanynyń sızbasy
plt.figure(figsize=(8, 5))
plt.title("Gamma-sáýlegenıń sińiriliýi – Qorǵan: Mis", fontsize=14)
plt.xlabel("Qalynǵdyǵy (cm)", fontsize=12)
plt.ylabel("ln(Z)", fontsize=12)
plt.grid(True, alpha=0.3)
mu_cu = fit_and_plot(x, cu_lnz, 'Mis (Cu)', 'red')
plt.legend()
plt.tight_layout()
plt.show()

# Nátıjeler
print(f"[Qurǵashyn (Pb)] Sińiriliý koefficienti μ = {mu_pb:.2f} cm⁻¹")
print(f"[Mis (Cu)] Sińiriliý koefficienti μ = {mu_cu:.2f} cm⁻¹")
