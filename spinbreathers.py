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
T_MAGNETIC_END = 300.0
FRAME_COUNT = 1000
RESULTS_FOLDER_PREFIX = "breather_run"

# --- Физические параметры: Атомная подсистема (модель ФПУ) ---
ATOMIC_OMEGA = 2.1
ATOMIC_PERIODS = 5

# --- Физические параметры: Магнитная подсистема (безразмерные) ---
# Все параметры нормированы на 2J в соответствии с научной работой
J_EXCHANGE = 1.0                 # Используется как масштабный множитель
D_DMI = 0.16                     # Параметр D/2J
A_ANISOTROPY = 0.15              # Параметр A/2J (анизотропия "легкая плоскость")
H_EFF = 0.40                      # Параметр H/2J (поле)

# --- Параметры связи и масштабирования ---
B_ME_COUPLING = 0.15              # Константа магнитоэлектрической связи
ENABLE_ME_COUPLING = True        # Включено для проверки модели со связью
TIME_SCALING_FACTOR = -15.0        # t_atomic = K * t_magnetic

# --- Параметры численного решателя ---
SOLVER_METHOD = 'DOP853'
SOLVER_RTOL = 1e-6
SOLVER_ATOL = 0

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

def calculate_initial_state_cartesian(N, d, a, h, b_me, pump_func):
    """
    Расчет гибридного начального состояния:
    - Центральные 5 узлов: конусная фаза, рассчитанная из ЛОКАЛЬНОЙ деформации.
    - Остальные узлы: принудительное ферромагнитное состояние.
    """
    logging.info("-> Шаг 2: Расчет гибридного начального состояния (локализованный бризер)...")

    # --- Шаг 1: Инициализируем всю цепочку в ФМ состоянии (theta=0) ---
    Sx0 = np.zeros(N)
    Sy0 = np.zeros(N)
    logging.info("   ...Вся цепочка инициализирована в ФМ состоянии (Sx=0, Sy=0).")

    # --- Шаг 2: Определяем центральные 5 узлов для модификации ---
    center_idx = N // 2
    # Создаем срез для 5 центральных элементов
    central_indices = np.arange(center_idx - 2, center_idx + 3)
    logging.info(f"   ...Будут модифицированы центральные узлы с индексами: {central_indices}.")

    # --- Шаг 3: Рассчитываем профиль максимальных амплитуд для вычисления деформации ---
    p_indices_all = np.arange(N)
    q_p_max_amplitudes = pump_func.staggered_factors * pump_func.profile_amplitudes
    q_p1_max = np.roll(q_p_max_amplitudes, -1)
    q_m1_max = np.roll(q_p_max_amplitudes, 1)
    q_p1_max[-1], q_m1_max[0] = 0, 0 # Граничные условия

    # --- Шаг 4: Цикл только по центральным узлам для расчета их локального угла theta_p ---
    q_spiral = np.arctan(-d) # Спиральный вектор q постоянен для всей системы

    for p in central_indices:
        # 4.1. Вычисляем локальную максимальную деформацию для узла p
        local_max_strain = q_p1_max[p] - q_m1_max[p]

        # 4.2. Вычисляем локальную эффективную анизотропию (взял модуль деформации)
        a_eff_local = a + b_me * np.abs(local_max_strain)

        # 4.3. Расчет локального критического поля и угла theta_p
        denominator = 2 * (np.sqrt(1 + d**2) - 1) + 2 * a_eff_local

        if h > abs(denominator):
            # Даже в центре поле может оказаться сверхкритическим, если деформация мала
            theta_p = 0.0
        else:
            cos_theta_val = np.clip(h / denominator, -1.0, 1.0)
            theta_p = np.arccos(cos_theta_val)

        # 4.4. Рассчитываем и записываем компоненты Sx, Sy для данного узла p
        phi_p = q_spiral * p
        Sx0[p] = np.sin(theta_p) * np.cos(phi_p)
        Sy0[p] = np.sin(theta_p) * np.sin(phi_p)
        logging.info(f"      - Узел p={p}: strain={local_max_strain:.3f}, A_eff={a_eff_local:.3f}, theta={np.degrees(theta_p):.2f}°")


    logging.info(f"   ...Параметры конусной фазы (для центра): q={q_spiral:.3f}")

    # --- Шаг 5: Собираем финальный вектор начальных условий ---
    Y0 = np.column_stack((Sx0, Sy0)).ravel()
    Y0 += 1e-5 * np.random.randn(2 * N) # Добавляем малый шум для численной устойчивости
    return Y0

pbar = None
def magnetic_equations_with_progress(t_mag, Y, *args):
    """Обертка для уравнений, обновляющая индикатор прогресса."""
    global pbar
    if pbar is not None and pbar.n < t_mag:
        pbar.update(t_mag - pbar.n)
    return magnetic_equations_cartesian(t_mag, Y, *args)

def magnetic_equations_cartesian(t_mag, Y, N, D_norm, A_norm, H_norm, B_me_norm, q_pump_func, K):
    """
    Система уравнений, где МУ-связь корректно модулирует анизотропию.
    """
    Sx, Sy = Y[0::2], Y[1::2]
    Sz = np.sqrt(np.maximum(0, 1 - Sx**2 - Sy**2))
    S_plus = Sx + 1j * Sy

    S_plus_p1 = np.roll(S_plus, -1); S_plus_m1 = np.roll(S_plus, 1)
    Sz_p1 = np.roll(Sz, -1); Sz_m1 = np.roll(Sz, 1)
    
    # --- КОРРЕКТНАЯ РЕАЛИЗАЦИЯ ВСЕХ ЧЛЕНОВ ---
    RHS_ex = -(Sz * (S_plus_p1 + S_plus_m1) - S_plus * (Sz_p1 + Sz_m1))
    RHS_dmi = -1j * D_norm * Sz * (S_plus_p1 - S_plus_m1)
    RHS_zeeman = H_norm * S_plus

    # МУ-связь как модуляция анизотропии
    p_indices = np.arange(N)
    q_p = q_pump_func(p_indices, K * t_mag)
    q_p1 = np.roll(q_p, -1); q_m1 = np.roll(q_p, 1)
    q_p1[-1], q_m1[0] = 0, 0
    strain = q_p1 - q_m1
    effective_anisotropy = A_norm + B_me_norm * strain
    RHS_aniso = effective_anisotropy * Sz * S_plus
    
    # --- Граничные условия для открытой цепочки ---
    # Для p=0 (нет левого соседа)
    RHS_ex[0] = -(Sz[0] * S_plus_p1[0] - S_plus[0] * Sz_p1[0])
    RHS_dmi[0] = -1j * D_norm * Sz[0] * S_plus_p1[0]
    
    # Для p=N-1 (нет правого соседа)
    RHS_ex[-1] = -(Sz[-1] * S_plus_m1[-1] - S_plus[-1] * Sz_m1[-1])
    RHS_dmi[-1] = -1j * D_norm * Sz[-1] * (-S_plus_m1[-1])

    RHS = RHS_ex + RHS_dmi - RHS_aniso + RHS_zeeman
    
    dSx_dt = np.imag(RHS)
    dSy_dt = -np.real(RHS)

    return np.column_stack((dSx_dt, dSy_dt)).ravel()

def calculate_magnetic_energy(Sx, Sy, Sz, t_sol, N, D_norm, A_norm, H_norm, B_me_norm, q_pump_func, K):
    """Вычисляет полную нормированную энергию E/2J, СОГЛАСОВАННУЮ с уравнениями."""
    logging.info("   ...Расчет эволюции энергии...")
    energies = []
    p_indices = np.arange(N)
    for i in tqdm(range(len(t_sol)), desc="      ...Энергия"):
        E_zeeman = -H_norm * np.sum(Sz[:, i])

        # Энергия анизотропии и МУ-связи объединены
        t_atomic = K * t_sol[i]
        q_p = q_pump_func(p_indices, t_atomic)
        q_p1 = np.roll(q_p, -1); q_m1 = np.roll(q_p, 1)
        q_p1[-1], q_m1[0] = 0, 0
        strain = q_p1 - q_m1
        effective_anisotropy = A_norm + B_me_norm * strain
        E_anisotropy_total = np.sum(effective_anisotropy * (Sz[:, i]**2))

        Sx_p, Sx_p1 = Sx[:-1, i], Sx[1:, i]
        Sy_p, Sy_p1 = Sy[:-1, i], Sy[1:, i]
        Sz_p, Sz_p1 = Sz[:-1, i], Sz[1:, i]

        E_exchange = -np.sum(Sx_p*Sx_p1 + Sy_p*Sy_p1 + Sz_p*Sz_p1)
        E_dmi = D_norm * np.sum(Sx_p*Sy_p1 - Sy_p*Sx_p1)

        energies.append(E_anisotropy_total + E_zeeman + E_exchange + E_dmi)
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
    ax1.plot(lattice_points, Sx_data[:, frame_idx], 'o-', color='royalblue'); ax1.set_title('2. Эволюция S_x компоненты'); ax1.set_xlabel('Узел (p)'); ax1.set_ylabel('S_x'); ax1.set_ylim(-1.05, 1.05); ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = axs[0, 1]
    ax2.plot(lattice_points, Sz_data[:, frame_idx], 'o-', color='royalblue'); ax2.set_title('2. Эволюция S_z компоненты'); ax2.set_xlabel('Узел (p)'); ax2.set_ylabel('S_z'); ax2.set_ylim(-1.05, 1.05); ax2.grid(True, linestyle='--', alpha=0.6)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d'); ax3.set_title('3. 3D представление спинов')
    ax3.quiver(lattice_points, 0, 0, Sz_data[:, frame_idx], Sx_data[:, frame_idx], Sy_data[:, frame_idx], length=0.8, normalize=True)
    ax3.set_xlim(0, N); ax3.set_ylim(-1, 1); ax3.set_zlim(-1, 1); ax3.set_xlabel('Узел (p)'); ax3.set_ylabel('S_x'); ax3.set_zlabel('S_y'); ax3.view_init(elev=30, azim=-120)

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
    logging.info(f"   ...Комбинированный GIF файл сохранен: {filename}")

# ==============================================================================
# --- ОСНОВНОЙ ИСПОЛНЯЕМЫЙ БЛОК ---
# ==============================================================================

def main():
    """Главная функция, запускающая весь процесс моделирования."""
    results_dir = Path(f"{RESULTS_FOLDER_PREFIX}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results_dir.mkdir(exist_ok=True)
    setup_logging(results_dir / "simulation.log")

    logging.info("="*60 + "\n--- ЗАПУСК МОДЕЛИРОВАНИЯ (декартовы координаты) ---\n" + "="*60)
    param_string = (f"ПАРАМЕТРЫ:\n" f"N={N_SITES}, T_END={T_MAGNETIC_END}, Ω={ATOMIC_OMEGA}, B_me={B_ME_COUPLING:.2f}\n"
                    f"D/2J={D_DMI:.2f}, A/2J={A_ANISOTROPY:.2f}, H/2J={H_EFF:.2f}\n"
                    f"РЕШАТЕЛЬ: {SOLVER_METHOD}, Rtol={SOLVER_RTOL}, Atol={SOLVER_ATOL}")
    logging.info("\n--- Стартовая конфигурация ---\n" + param_string + "\n" + "-"*28 + "\n")

    J_NORM = 2.0 * J_EXCHANGE 
    b_me_effective = B_ME_COUPLING if ENABLE_ME_COUPLING else 0.0
    logging.info(f"Магнитоупругое взаимодействие: {'ВКЛЮЧЕНО' if ENABLE_ME_COUPLING else 'ВЫКЛЮЧЕНО'}")

    atomic_pump = get_atomic_pumping_function(N_SITES, ATOMIC_OMEGA)
    Y0_magnetic = calculate_initial_state_cartesian(N_SITES, D_DMI, A_ANISOTROPY, H_EFF, b_me_effective, atomic_pump)
    
    logging.info("\n-> Шаг 3: Запуск динамического моделирования магнитной системы...")
    global pbar
    with tqdm(total=T_MAGNETIC_END, desc="   ...Интегрирование") as pbar_instance:
        pbar = pbar_instance
        sol_magnetic = solve_ivp(
            fun=magnetic_equations_with_progress, t_span=(0, T_MAGNETIC_END), y0=Y0_magnetic,
            method=SOLVER_METHOD, dense_output=True, rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
            args=(N_SITES, D_DMI, A_ANISOTROPY, H_EFF, b_me_effective, atomic_pump, TIME_SCALING_FACTOR),
        )
    pbar = None
    
    logging.info(f"   ...Динамическое моделирование завершено. Статус: {sol_magnetic.message}")
    if not sol_magnetic.success:
        logging.error("!!! Решатель не справился с задачей. Моделирование прервано. !!!")
        return

    logging.info("\n-> Шаг 4: Постобработка и визуализация результатов...")
    t_eval = np.linspace(0, T_MAGNETIC_END, FRAME_COUNT)
    Y_eval = sol_magnetic.sol(t_eval)
    
    Sx_eval = Y_eval[0::2, :]
    Sy_eval = Y_eval[1::2, :]

    # --- Принудительная перенормировка для численной стабильности ---
    logging.info("   ...Выполнение принудительной перенормировки векторов спинов...")
    norm_sq = Sx_eval**2 + Sy_eval**2
    mask = norm_sq > 1.0
    # Добавляем EPSILON чтобы избежать деления на ноль, если norm_sq[mask] будет содержать нули (маловероятно)
    norm = np.sqrt(norm_sq[mask] + EPSILON) 
    Sx_eval[mask] /= norm
    Sy_eval[mask] /= norm
    logging.info(f"      ...Перенормировано {np.sum(mask)} точек, где S_x^2 + S_y^2 > 1.")
    # --- Конец блока перенормировки ---

    Sz_eval = np.sqrt(np.maximum(0, 1 - Sx_eval**2 - Sy_eval**2))

    energies = calculate_magnetic_energy(Sx_eval, Sy_eval, Sz_eval, t_eval, N_SITES, D_DMI, A_ANISOTROPY, H_EFF, b_me_effective, atomic_pump, TIME_SCALING_FACTOR)
    gif_path = results_dir / "combined_dynamics.gif"
    create_combined_visualization_parallel(Sz_eval, Sx_eval, Sy_eval, energies, t_eval, atomic_pump, N_SITES, gif_path, param_string)

    logging.info("\n" + "="*60 + "\n--- МОДЕЛИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО ---\n" + f"Результаты сохранены в папке: {results_dir}\n" + "="*60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()