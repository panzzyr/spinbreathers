# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

# ==============================================================================
# --- КОНФИГУРАЦИЯ ---
# ==============================================================================

N_SITES = 101
ATOMIC_OMEGA = 2.1
ATOMIC_PERIODS = 1

J_EXCHANGE = 1.0
D_DMI = 0.01
A_ANISOTROPY = 0.41
H_EFF = 0.01
B_ME_COUPLING = 5.0 

T_MAGNETIC_END = 50.0
TIME_SCALING_FACTOR = 1.0
FRAME_COUNT = 50
SOLVER_METHOD = 'DOP853'
MAX_STEP = 0.1
RESULTS_FOLDER = f"breather_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ==============================================================================

def fpu_equations(t, Y, N):
    q = Y[:N]; p = Y[N:]
    dqdt = p
    dpdt = np.zeros_like(p)
    for i in range(N):
        ip = (i + 1) % N; im = (i - 1) % N
        lin = q[ip] + q[im] - 2 * q[i]
        nonlin = (q[ip] - q[i])**3 - (q[i] - q[im])**3
        dpdt[i] = lin + nonlin
    return np.concatenate([dqdt, dpdt])

def get_atomic_pumping_function(N, omega):
    print("-> Шаг 1: Моделирование атомного бризера для получения функции накачки...")
    diff_sq = omega**2 - 4.0
    if diff_sq <= 0: raise ValueError("Omega^2 должен быть > 4.")
    A_ilm = np.sqrt(diff_sq / 6.0)
    q0 = np.zeros(N)
    n0 = N // 2
    for n in range(N):
        factor = (-1)**n
        arg = np.sqrt(6) * A_ilm * (n - n0)
        q0[n] = factor * A_ilm * (2.0 / (np.exp(arg) + np.exp(-arg)))
    y0_atomic = np.concatenate([q0, np.zeros(N)])
    t_end_atomic = (2 * np.pi / omega) * ATOMIC_PERIODS
    print("   ...Запускаю симуляцию атомной решетки (может занять время)...")
    sol = solve_ivp(
        lambda t, y: fpu_equations(t, y, N),
        (0, t_end_atomic), y0_atomic, method='RK45', dense_output=True
    )
    t_eval = np.linspace(0, t_end_atomic, 100)
    q_vals = sol.sol(t_eval)[:N]
    amplitudes = np.max(np.abs(q_vals), axis=1)
    print(f"   ...Амплитуды колебаний атомов найдены (макс. в центре: {amplitudes[N//2]:.4f}).")

    def pumping_function(p, t_atomic):
        idx = p % N
        staggered_factor = (-1)**idx
        return staggered_factor * amplitudes[idx] * np.cos(omega * t_atomic)
        
    return pumping_function

def calculate_conical_phase(N, J, D, A, H_eff):
    print("-> Шаг 2: Расчет начального состояния (конусной фазы)...")
    denom = J * (2 * (np.sqrt(1 + (D / (2 * J))**2) - 1) + (A / J))
    if np.abs(denom) < 1e-9:
        cos_theta = 1.0 if H_eff > 0 else -1.0
    else:
        cos_theta = H_eff / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    q_spiral = np.arctan(-D / (2 * J))
    print(f"   ...Параметры конусной фазы: q={q_spiral:.3f}, theta={np.degrees(theta):.2f}°")
    Y0 = np.zeros(2 * N)
    for p in range(N):
        phi_p = q_spiral * p
        Y0[2 * p] = np.sin(theta) * np.cos(phi_p)
        Y0[2 * p + 1] = np.sin(theta) * np.sin(phi_p)
    noise_level = 1e-5
    Y0 += noise_level * np.random.randn(2 * N)
    print(f"   ...Добавлено начальное возмущение с уровнем {noise_level}.")
    return Y0

def magnetic_equations(t_mag, Y, N, J, D, A, H_eff, B_me, q_pump_func, K):
    Sx = Y[0::2]; Sy = Y[1::2]
    sz_sq = 1 - Sx**2 - Sy**2
    Sz = np.sqrt(np.maximum(0, sz_sq))
    dSx_dt = np.zeros(N); dSy_dt = np.zeros(N)
    t_atomic = K * t_mag
    for p in range(N):
        pp1 = (p + 1) % N; pm1 = (p - 1) % N
        q_pp1 = q_pump_func(pp1, t_atomic); q_pm1 = q_pump_func(pm1, t_atomic)
        Sp_p = Sx[p] + 1j * Sy[p]; Sp_pp1 = Sx[pp1] + 1j * Sy[pp1]; Sp_pm1 = Sx[pm1] + 1j * Sy[pm1]
        RHS = -J * Sz[p] * (Sp_pp1 + Sp_pm1) + J * Sp_p * (Sz[pp1] + Sz[pm1])
        RHS += 1j * D * (Sz[p] * Sp_pp1 - Sz[pp1] * Sp_p) - 1j * D * (Sz[pm1] * Sp_p - Sz[p] * Sp_pm1)
        RHS += 2 * A * Sz[p] * Sp_p + H_eff * Sp_p + B_me * (q_pp1 - q_pm1) * Sp_p
        dS_plus_dt = -1j * RHS
        dSx_dt[p] = np.real(dS_plus_dt); dSy_dt[p] = np.imag(dS_plus_dt)
    return np.column_stack((dSx_dt, dSy_dt)).ravel()

def calculate_magnetic_energy(Y_sol, t_sol, N, J, D, A, H_eff, B_me, q_pump_func, K):
    energies = []
    Sx = Y_sol[0::2, :]; Sy = Y_sol[1::2, :]
    Sz = np.sqrt(np.maximum(0, 1 - Sx**2 - Sy**2))
    for i in tqdm(range(len(t_sol)), desc="   ...Расчет энергии"):
        t = t_sol[i]; t_atomic = K * t
        total_energy = 0
        for p in range(N):
            pp1 = (p + 1) % N
            total_energy -= J * (Sx[p, i]*Sx[pp1, i] + Sy[p, i]*Sy[pp1, i] + Sz[p, i]*Sz[pp1, i])
            total_energy -= D * (Sx[p, i]*Sy[pp1, i] - Sy[p, i]*Sx[pp1, i])
            total_energy -= A * Sz[p, i]**2 - H_eff * Sz[p, i]
            q_pp1 = q_pump_func(pp1, t_atomic); q_pm1 = q_pump_func((p - 1) % N, t_atomic)
            total_energy -= B_me * (q_pp1 - q_pm1) * Sz[p, i]
        energies.append(total_energy)
    return np.array(energies)






def create_sz_evolution_gif(Sz_data, t_vals, filename, param_text):
    print("-> Шаг 4.1: Создание GIF анимации...")
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    filenames = []
    for i in tqdm(range(Sz_data.shape[1]), desc="   ...Генерация кадров"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(Sz_data[:, i], 'o-', color='royalblue')
        ax.set_title(f'Эволюция S_z компоненты | Время = {t_vals[i]:.2f}')
        ax.set_xlabel('Узел решетки (p)'); ax.set_ylabel('S_z')
        ax.set_ylim(-1.05, 1.05); ax.grid(True, linestyle='--', alpha=0.6)
        
        # Добавляем легенду с параметрами
        plt.figtext(0.02, 0.02, param_text, ha="left", va="bottom", fontsize=8,
                    bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        filepath = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(filepath)
        plt.close(fig)
        filenames.append(filepath)
    with imageio.get_writer(filename, mode='I', duration=0.05) as writer:
        for f in tqdm(filenames, desc="   ...Сборка GIF"):
            image = imageio.v2.imread(f)
            writer.append_data(image)
    for f in filenames: os.remove(f)
    os.rmdir(temp_dir)
    print(f"   ...GIF файл сохранен: {filename}")

def create_energy_plot(energies, t_vals, filename, param_text):
    print("-> Шаг 4.2: Создание графика энергии...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_vals, energies, color='crimson')
    ax.set_title('Эволюция полной энергии магнитной системы')
    ax.set_xlabel('Время (t)'); ax.set_ylabel('Энергия (E)')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Добавляем легенду с параметрами
    plt.figtext(0.02, 0.02, param_text, ha="left", va="bottom", fontsize=8,
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.savefig(filename)
    plt.close(fig)
    print(f"   ...График энергии сохранен: {filename}")

if __name__ == "__main__":
    print("="*50); print("--- ЗАПУСК МОДЕЛИРОВАНИЯ ---"); print("="*50)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # --- Формируем строку с параметрами для легенды --- 
    param_string = (
        f"Параметры симуляции:\n"
        f"N_SITES = {N_SITES}, T_END = {T_MAGNETIC_END}\n"
        f"--- Атомная подсистема ---\n"
        f"Ω = {ATOMIC_OMEGA}\n"
        f"--- Магнитная подсистема (норм. на J) ---\n"
        f"D/J = {D_DMI:.2f}, A/J = {A_ANISOTROPY:.2f}, H_eff/J = {H_EFF:.2f}\n"
        f"B_me = {B_ME_COUPLING:.2f}, K (time_scale) = {TIME_SCALING_FACTOR:.2f}"
    )
    print("--- Стартовая конфигурация ---")
    print(param_string)
    print("----------------------------")

    atomic_pump = get_atomic_pumping_function(N_SITES, ATOMIC_OMEGA)
    Y0_magnetic = calculate_conical_phase(N_SITES, J_EXCHANGE, D_DMI, A_ANISOTROPY, H_EFF)
    print("-> Шаг 3: Запуск динамического моделирования магнитной системы...")
    sol_magnetic = solve_ivp(
        magnetic_equations, (0, T_MAGNETIC_END), Y0_magnetic, method=SOLVER_METHOD,
        dense_output=True, max_step=MAX_STEP,
        args=(N_SITES, J_EXCHANGE, D_DMI, A_ANISOTROPY, H_EFF, B_ME_COUPLING, atomic_pump, TIME_SCALING_FACTOR),
    )
    print("   ...Динамическое моделирование завершено.")
    print("-> Шаг 4: Постобработка и визуализация результатов...")
    t_eval = np.linspace(0, T_MAGNETIC_END, FRAME_COUNT)
    Y_eval = sol_magnetic.sol(t_eval)
    Sx_eval = Y_eval[0::2, :]; Sy_eval = Y_eval[1::2, :]
    
    # --- ВОТ ИСПРАВЛЕННАЯ СТРОКА ---
    Sz_eval = np.sqrt(np.maximum(0, 1 - Sx_eval**2 - Sy_eval**2))
    
    # --- Передаем строку с параметрами в функции визуализации ---
    gif_path = os.path.join(RESULTS_FOLDER, "sz_evolution.gif")
    create_sz_evolution_gif(Sz_eval, t_eval, gif_path, param_string)
    
    energies = calculate_magnetic_energy(
        Y_eval, t_eval, N_SITES, J_EXCHANGE, D_DMI, A_ANISOTROPY, 
        H_EFF, B_ME_COUPLING, atomic_pump, TIME_SCALING_FACTOR
    )
    energy_plot_path = os.path.join(RESULTS_FOLDER, "energy_evolution.png")
    create_energy_plot(energies, t_eval, energy_plot_path, param_string)
    
    print("\n" + "="*50); print("--- МОДЕЛИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО ---"); print(f"Результаты сохранены в папке: {RESULTS_FOLDER}"); print("="*50)
