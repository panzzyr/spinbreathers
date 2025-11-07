# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import logging
import multiprocessing
from functools import partial
from pathlib import Path

from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from tqdm import tqdm

# ==============================================================================
# --- ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ МОДЕЛИ ---
# ==============================================================================

# --- Параметры решетки и симуляции ---
N_SITES = 101
T_MAGNETIC_END = 50.0
FRAME_COUNT = 100
RESULTS_FOLDER_PREFIX = "breather_run"

# --- Физические параметры: Атомная подсистема (модель ФПУ) ---
ATOMIC_OMEGA = 2.1
ATOMIC_PERIODS = 5

# --- Физические параметры: Магнитная подсистема (безразмерные) ---
# Все параметры нормированы на 2J в соответствии с научной работой
J_EXCHANGE = 1.0                 # Используется как масштабный множитель
D_DMI = 0.16                     # Параметр D/2J
A_ANISOTROPY = 0.15              # Параметр A/2J (анизотропия "легкая плоскость")
H_EFF = 0.10                     # Параметр H/2J (поле чуть выше критического)

# --- Параметры связи и масштабирования ---
B_ME_COUPLING = 0.5              # Константа магнитоэлектрической связи
ENABLE_ME_COUPLING = False       # Отключено для проверки базовой модели
TIME_SCALING_FACTOR = 1.0        # t_atomic = K * t_magnetic

# --- Параметры численного решателя ---
SOLVER_METHOD = 'Radau'
SOLVER_RTOL = 1e-6
SOLVER_ATOL = 1e-6

# --- Технические константы ---
EPSILON = 1e-12                  # Малая константа для избежания деления на ноль

# ==============================================================================
# --- ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ И ФУНКЦИИ ---
# ==============================================================================

class AtomicPumpingFunction:
    """
    Класс-обертка для векторизованной функции атомной накачки.
    Решает проблему с передачей локальных функций в другие процессы
    при параллельных вычислениях (multiprocessing).
    """
    def __init__(self, profile_amplitudes, staggered_factors, omega):
        self.profile_amplitudes = profile_amplitudes
        self.staggered_factors = staggered_factors
        self.omega = omega

    def __call__(self, p_indices, t_atomic):
        """Делает экземпляр класса вызываемым, как обычную функцию."""
        return self.staggered_factors * self.profile_amplitudes * np.cos(self.omega * t_atomic)

def setup_logging(log_path):
    """Настраивает систему логирования для вывода в консоль и файл."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

# ==============================================================================
# --- АТОМНАЯ ПОДСИСТЕМА (БРИЗЕР) ---
# ==============================================================================

def fpu_equations(t, Y, N):
    """Система уравнений для модели Ферми-Паста-Улама."""
    q = Y[:N]; p = Y[N:]
    dqdt = p
    q_plus_1 = np.roll(q, -1); q_minus_1 = np.roll(q, 1)
    linear_force = q_plus_1 + q_minus_1 - 2 * q
    nonlinear_force = (q_plus_1 - q)**3 - (q - q_minus_1)**3
    dpdt = linear_force + nonlinear_force
    return np.concatenate([dqdt, dpdt])

def get_atomic_pumping_function(N, omega):
    """
    Моделирует атомный бризер и возвращает вызываемый объект
    для расчета смещений атомов в любой момент времени.
    """
    logging.info("-> Шаг 1: Моделирование атомного бризера для получения функции накачки...")
    diff_sq = omega**2 - 4.0
    if diff_sq <= 0: raise ValueError("Omega^2 должен быть > 4.0 для существования бризера.")

    A_ilm = np.sqrt(diff_sq / 6.0)
    n0 = N // 2
    x_data = np.arange(N)
    q0 = (-1)**x_data * A_ilm * (2.0 / np.cosh(np.sqrt(6) * A_ilm * (x_data - n0)))

    y0_atomic = np.concatenate([q0, np.zeros(N)])
    t_end_atomic = (2 * np.pi / omega) * ATOMIC_PERIODS

    logging.info("   ...Запускаю симуляцию атомной решетки для уточнения профиля...")
    sol = solve_ivp(fun=lambda t, y: fpu_equations(t, y, N), t_span=(0, t_end_atomic),
                    y0=y0_atomic, method='RK45', dense_output=True)

    q_vals = sol.sol(np.linspace(0, t_end_atomic, 200))[:N]
    amplitudes = np.max(np.abs(q_vals), axis=1)
    logging.info(f"   ...Амплитуды колебаний атомов найдены (макс. в центре: {amplitudes[n0]:.4f}).")

    def gaussian_profile(x, amp, center, sigma): return amp * np.exp(-((x - center)**2) / (2 * sigma**2))

    try:
        params, _ = curve_fit(gaussian_profile, x_data, amplitudes, p0=[np.max(amplitudes), n0, 5.0])
        profile_amplitudes = gaussian_profile(x_data, *params)
        logging.info("   ...Профиль успешно аппроксимирован Гауссианой.")
    except RuntimeError:
        logging.warning("   ...Аппроксимация не удалась, используется исходный численный профиль.")
        profile_amplitudes = amplitudes

    staggered_factors = (-1)**x_data
    return AtomicPumpingFunction(profile_amplitudes, staggered_factors, omega)

# ==============================================================================
# --- МАГНИТНАЯ ПОДСИСТЕМА (УРАВНЕНИЯ ДВИЖЕНИЯ) ---
# ==============================================================================

def calculate_conical_phase(N, d, b, h):
    """
    Расчет начального состояния (конической спирали) на основе
    безразмерных параметров: d=D/2J, b=A/2J, h=H/2J.
    """
    logging.info("-> Шаг 2: Расчет начального состояния (конусной фазы)...")
    denominator = 2 * (np.sqrt(1 + d**2) - 1) - 2 * b
    
    # Проверяем, не является ли поле сверхкритическим
    if h > abs(denominator):
        logging.info("   ...Обнаружено сверхкритическое поле. Устанавливается состояние полной поляризации.")
        theta0 = 0.0 # Все спины выровнены по полю H (вдоль +Z)
        q_spiral = 0.0 # Спираль отсутствует
    else:
        cos_theta_val = h / denominator if np.abs(denominator) > EPSILON else 1.0
        cos_theta_val = np.clip(cos_theta_val, -1.0, 1.0)
        theta0 = np.arccos(cos_theta_val)
        q_spiral = np.arctan(d)

    logging.info(f"   ...Параметры конусной фазы: q={q_spiral:.3f}, theta={np.degrees(theta0):.2f}°")

    Y0 = np.zeros(2 * N)
    p_indices = np.arange(N)
    phi0 = q_spiral * p_indices
    Y0[0::2] = theta0
    Y0[1::2] = phi0
    Y0 += 1e-5 * np.random.randn(2 * N)
    return Y0

pbar = None
def magnetic_equations_with_progress(t_mag, Y, *args):
    """Обертка для уравнений, обновляющая индикатор прогресса."""
    global pbar
    if pbar is not None and pbar.n < t_mag:
        pbar.update(t_mag - pbar.n)
    return magnetic_equations_spherical(t_mag, Y, *args)

def magnetic_equations_spherical(t_mag, Y, N, D_norm, A_norm, H_norm, B_me_norm, q_pump_func, K):
    """
    Система уравнений Ландау-Лифшица в сферических координатах.
    Все параметры (D, A, H, B_me) нормированы на 2J.
    """
    theta, phi = Y[0::2], Y[1::2]

    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    Sx, Sy, Sz = sin_theta * cos_phi, sin_theta * sin_phi, cos_theta

    Sx_p1, Sx_m1 = np.roll(Sx, -1), np.roll(Sx, 1)
    Sy_p1, Sy_m1 = np.roll(Sy, -1), np.roll(Sy, 1)
    Sz_p1, Sz_m1 = np.roll(Sz, -1), np.roll(Sz, 1)

    # --- Расчет компонент эффективного поля H_eff = -d(H/2J)/dS ---
    Hx = (Sx_p1 + Sx_m1) - D_norm * (Sy_p1 - Sy_m1)
    Hy = (Sy_p1 + Sy_m1) + D_norm * (Sx_p1 - Sx_m1)
    Hz = (Sz_p1 + Sz_m1) - 2 * A_norm * Sz + H_norm * np.ones(N)

    if B_me_norm != 0.0:
        p_indices = np.arange(N)
        q_p = q_pump_func(p_indices, K * t_mag)
        q_p1 = np.roll(q_p, -1); q_m1 = np.roll(q_p, 1)
        q_p1[-1], q_m1[0] = 0, 0
        Hz += B_me_norm * (q_p1 - q_m1)

    # --- Граничные условия для открытой цепочки ---
    Hx[0]  = Sx_p1[0] - D_norm * Sy_p1[0]
    Hy[0]  = Sy_p1[0] + D_norm * Sx_p1[0]
    Hz[0]  = Sz_p1[0] - 2 * A_norm * Sz[0] + H_norm
    if B_me_norm != 0.0: Hz[0] += B_me_norm * q_p1[0]

    Hx[-1] = Sx_m1[-1] + D_norm * Sy_m1[-1]
    Hy[-1] = Sy_m1[-1] - D_norm * Sx_m1[-1]
    Hz[-1] = Sz_m1[-1] - 2 * A_norm * Sz[-1] + H_norm
    if B_me_norm != 0.0: Hz[-1] += B_me_norm * (-q_m1[-1])

    # --- Преобразование декартова поля H_eff в сферические компоненты ---
    H_theta = Hx * cos_theta * cos_phi + Hy * cos_theta * sin_phi - Hz * sin_theta
    H_phi   = -Hx * sin_phi + Hy * cos_phi

    # --- Уравнения Ландау-Лифшица для углов ---
    d_theta_dt = H_phi
    d_phi_dt = -H_theta / (sin_theta + EPSILON)

    return np.column_stack((d_theta_dt, d_phi_dt)).ravel()

def convert_angles_to_spins(Y_sol):
    """Преобразует решение в углах (theta, phi) в декартовы спины."""
    theta, phi = Y_sol[0::2, :], Y_sol[1::2, :]
    Sx = np.sin(theta) * np.cos(phi)
    Sy = np.sin(theta) * np.sin(phi)
    Sz = np.cos(theta)
    return Sx, Sy, Sz

def calculate_magnetic_energy(Sx, Sy, Sz, t_sol, N, D_norm, A_norm, H_norm, B_me_norm, q_pump_func, K):
    """Вычисляет полную нормированную энергию магнитной системы E/2J."""
    logging.info("   ...Расчет эволюции энергии...")
    energies = []
    p_indices = np.arange(N)
    for i in tqdm(range(len(t_sol)), desc="      ...Энергия"):
        E_anisotropy = A_norm * np.sum(Sz[:, i]**2)
        E_zeeman = -H_norm * np.sum(Sz[:, i])

        E_me_coupling = 0.0
        # Считаем этот вклад, только если он не равен нулю (т.е. связь включена)
        if B_me_norm != 0.0:
            t_atomic = K * t_sol[i]
            q_p = q_pump_func(p_indices, t_atomic)
            q_p1 = np.roll(q_p, -1); q_m1 = np.roll(q_p, 1)
            q_p1[-1], q_m1[0] = 0, 0
            E_me_coupling = -B_me_norm * np.sum((q_p1 - q_m1) * Sz[:, i])

        Sx_p, Sx_p1 = Sx[:-1, i], Sx[1:, i]
        Sy_p, Sy_p1 = Sy[:-1, i], Sy[1:, i]
        Sz_p, Sz_p1 = Sz[:-1, i], Sz[1:, i]

        E_exchange = -np.sum(Sx_p*Sx_p1 + Sy_p*Sy_p1 + Sz_p*Sz_p1)
        E_dmi = D_norm * np.sum(Sx_p*Sy_p1 - Sy_p*Sx_p1)

        energies.append(E_anisotropy + E_zeeman + E_me_coupling + E_exchange + E_dmi)
    return np.array(energies)

# ==============================================================================
# --- ВИЗУАЛИЗАЦИЯ ---
# ==============================================================================

def render_frame(frame_idx, t_vals, Sz_data, Sx_data, Sy_data, energies, pump_func, N, q_ylim, energy_ylim, param_text, temp_dir):
    """Отрисовывает один кадр для последующей сборки в GIF."""
    t_current = t_vals[frame_idx]
    lattice_points = np.arange(N)
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Полная динамика системы | Время = {t_current:.2f}', fontsize=16)

    ax1 = axs[0, 0]
    q_values = pump_func(lattice_points, TIME_SCALING_FACTOR * t_current)
    ax1.plot(lattice_points, q_values, 'o-', color='green'); ax1.set_title('1. Колебания атомной решетки (q_p)'); ax1.set_xlabel('Узел (p)'); ax1.set_ylabel('Смещение (q)'); ax1.set_ylim(-q_ylim, q_ylim); ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = axs[0, 1]
    ax2.plot(lattice_points, Sz_data[:, frame_idx], 'o-', color='royalblue'); ax2.set_title('2. Эволюция S_z компоненты'); ax2.set_xlabel('Узел (p)'); ax2.set_ylabel('S_z'); ax2.set_ylim(-1.05, 1.05); ax2.grid(True, linestyle='--', alpha=0.6)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d'); ax3.set_title('3. 3D представление спинов')
    ax3.quiver(lattice_points, 0, 0, Sx_data[:, frame_idx], Sy_data[:, frame_idx], Sz_data[:, frame_idx], length=0.8, normalize=True)
    ax3.set_xlim(0, N); ax3.set_ylim(-1, 1); ax3.set_zlim(-1, 1); ax3.set_xlabel('Узел (p)'); ax3.set_ylabel('S_y'); ax3.set_zlabel('S_z'); ax3.view_init(elev=20, azim=-75)

    ax4 = axs[1, 1]
    ax4.plot(t_vals[:frame_idx+1], energies[:frame_idx+1], color='crimson'); ax4.set_title('4. Эволюция полной энергии'); ax4.set_xlabel('Время (t)'); ax4.set_ylabel('Энергия (E/2J)'); ax4.set_xlim(0, t_vals[-1]); ax4.set_ylim(energy_ylim); ax4.grid(True, linestyle='--', alpha=0.6)

    plt.figtext(0.02, 0.02, param_text, ha="left", va="bottom", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    filepath = temp_dir / f"frame_{frame_idx:04d}.png"
    plt.savefig(filepath)
    plt.close(fig)
    return str(filepath)

def create_combined_visualization_parallel(Sz_data, Sx_data, Sy_data, energies, t_vals, pump_func, N, filename, param_text):
    """Создает GIF-анимацию, распараллеливая генерацию кадров."""
    logging.info("-> Шаг 4.1: Создание комбинированной GIF анимации (параллельно)...")
    temp_dir = Path("temp_frames"); temp_dir.mkdir(exist_ok=True)
    
    min_e, max_e = np.min(energies), np.max(energies)
    e_range = max_e - min_e if max_e > min_e else 1.0
    energy_ylim = (min_e - e_range * 0.1, max_e + e_range * 0.1)

    q_at_zero = np.abs(pump_func(np.arange(N), 0))
    max_abs_q = np.max(q_at_zero)
    q_ylim = max_abs_q * 1.15 if max_abs_q > 0 else 1.0

    task_func = partial(render_frame, t_vals=t_vals, Sz_data=Sz_data, Sx_data=Sx_data,
                        Sy_data=Sy_data, energies=energies, pump_func=pump_func,
                        N=N, q_ylim=q_ylim, energy_ylim=energy_ylim,
                        param_text=param_text, temp_dir=temp_dir)

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"   ...Используется {num_workers} ядер для генерации {FRAME_COUNT} кадров.")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        filenames = list(tqdm(pool.imap(task_func, range(FRAME_COUNT)), total=FRAME_COUNT, desc="   ...Генерация кадров"))

    filenames.sort()
    with imageio.get_writer(filename, mode='I', duration=1000/20, loop=0) as writer:
        for f in tqdm(filenames, desc="   ...Сборка GIF"):
            writer.append_data(imageio.v2.imread(f))
    for f in filenames: os.remove(f)
    os.rmdir(temp_dir)
    logging.info(f"   ...Комбинированный GIF файл сохранен: {filename}")

# ==============================================================================
# --- ОСНОВНОЙ ИСПОЛНЯЕМЫЙ БЛОК ---
# ==============================================================================

def main():
    """Главная функция, запускающая весь процесс моделирования."""
    results_dir = Path(f"{RESULTS_FOLDER_PREFIX}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results_dir.mkdir(exist_ok=True)
    setup_logging(results_dir / "simulation.log")

    logging.info("="*60 + "\n--- ЗАПУСК МОДЕЛИРОВАНИЯ (сферические координаты) ---\n" + "="*60)
    param_string = (f"ПАРАМЕТРЫ:\n" f"N={N_SITES}, T_END={T_MAGNETIC_END}, Ω={ATOMIC_OMEGA}, B_me={B_ME_COUPLING:.2f}\n"
                    f"D/2J={D_DMI:.2f}, A/2J={A_ANISOTROPY:.2f}, H/2J={H_EFF:.2f}\n"
                    f"РЕШАТЕЛЬ: {SOLVER_METHOD}, Rtol={SOLVER_RTOL}, Atol={SOLVER_ATOL}")
    logging.info("\n--- Стартовая конфигурация ---\n" + param_string + "\n" + "-"*28 + "\n")

    J_NORM = 2.0 * J_EXCHANGE # Величина, на которую нормируется Гамильтониан
    b_me_effective = B_ME_COUPLING if ENABLE_ME_COUPLING else 0.0
    logging.info(f"Магнитоупругое взаимодействие: {'ВКЛЮЧЕНО' if ENABLE_ME_COUPLING else 'ВЫКЛЮЧЕНО'}")

    atomic_pump = get_atomic_pumping_function(N_SITES, ATOMIC_OMEGA)
    Y0_magnetic = calculate_conical_phase(N_SITES, D_DMI, A_ANISOTROPY, H_EFF)
    
    logging.info("\n-> Шаг 3: Запуск динамического моделирования магнитной системы...")
    global pbar
    with tqdm(total=T_MAGNETIC_END, desc="   ...Интегрирование") as pbar_instance:
        pbar = pbar_instance
        sol_magnetic = solve_ivp(
            fun=magnetic_equations_with_progress, t_span=(0, T_MAGNETIC_END), y0=Y0_magnetic,
            method=SOLVER_METHOD, dense_output=True, rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
            args=(N_SITES, D_DMI, A_ANISOTROPY, H_EFF, b_me_effective / J_NORM, atomic_pump, TIME_SCALING_FACTOR),
        )
    pbar = None
    
    logging.info(f"   ...Динамическое моделирование завершено. Статус: {sol_magnetic.message}")
    if not sol_magnetic.success:
        logging.error("!!! Решатель не справился с задачей. Моделирование прервано. !!!")
        return

    logging.info("\n-> Шаг 4: Постобработка и визуализация результатов...")
    t_eval = np.linspace(0, T_MAGNETIC_END, FRAME_COUNT)
    Y_eval = sol_magnetic.sol(t_eval)
    
    Sx_eval, Sy_eval, Sz_eval = convert_angles_to_spins(Y_eval)

    energies = calculate_magnetic_energy(Sx_eval, Sy_eval, Sz_eval, t_eval, N_SITES, D_DMI, A_ANISOTROPY, H_EFF, b_me_effective / J_NORM, atomic_pump, TIME_SCALING_FACTOR)
    gif_path = results_dir / "combined_dynamics.gif"
    create_combined_visualization_parallel(Sz_eval, Sx_eval, Sy_eval, energies, t_eval, atomic_pump, N_SITES, gif_path, param_string)

    logging.info("\n" + "="*60 + "\n--- МОДЕЛИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО ---\n" + f"Результаты сохранены в папке: {results_dir}\n" + "="*60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()