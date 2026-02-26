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
T_MAGNETIC_END = 500.0
FRAME_COUNT = 5000
RESULTS_FOLDER_PREFIX = "new fft, 0.15 kick, 0.15 B_me, 0.02 K, D=0.16, A=0.15, H=0.40, "

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
TIME_SCALING_FACTOR = 0.02        # t_atomic = K * t_magnetic

# --- Параметры численного решателя ---
SOLVER_METHOD = 'DOP853'
SOLVER_RTOL = 1e-6
SOLVER_ATOL = 0

# --- Технические константы ---
EPSILON = 1e-12                  # Малая константа для избежания деления на ноль

# ==============================================================================
# --- ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ И ФУНКЦИИ ---
# ==============================================================================

def perform_fft_analysis_stages(t_dense, Sx_dense, N, save_path, pump_freq, num_stages=5):
    """
    Разбивает историю эволюции на num_stages этапов.
    Строит 1D-спектр центрального узла и 2D-спектр всей цепочки.
    """
    import logging
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    logging.info(f"-> Анализ Фурье: генерация {num_stages} комбинированных слепков...")
    
    chunk_size = len(t_dense) // num_stages
    center_idx = N // 2
    
    base_dir = os.path.dirname(save_path)
    base_name = os.path.basename(save_path).replace('.png', '')

    for stage in range(num_stages):
        start_idx = stage * chunk_size
        end_idx = (stage + 1) * chunk_size if stage < num_stages - 1 else len(t_dense)
        
        t_chunk = t_dense[start_idx:end_idx]
        Sx_chunk = Sx_dense[:, start_idx:end_idx]
        dt = t_chunk[1] - t_chunk[0]
        
        # --- Чистое Фурье (без окон и вычитания среднего) ---
        fft_vals = np.abs(np.fft.rfft(Sx_chunk, axis=1))
        freqs = np.fft.rfftfreq(Sx_chunk.shape[1], dt)
        
        # Данные для центрального узла
        fft_center = fft_vals[center_idx, :]
        
        # --- ВИЗУАЛИЗАЦИЯ ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        t_start_val, t_end_val = t_chunk[0], t_chunk[-1]
        fig.suptitle(f'Этап {stage + 1}/{num_stages} | t $\in$ [{t_start_val:.1f}, {t_end_val:.1f}]', fontsize=14)

        # 1. График центрального элемента (1D)
        ax1.plot(freqs, fft_center, color='forestgreen', lw=1.5)
        ax1.axvline(x=pump_freq, color='r', linestyle='--', label=f'Накачка (Ω={pump_freq:.2f})')
        ax1.set_title(f'Спектр центрального узла (N={center_idx})')
        ax1.set_ylabel('Амплитуда')
        ax1.set_xlim(0, 5.0)
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # 2. Тепловая карта всех элементов (2D)
        freq_mask = freqs <= 5.0
        im = ax2.imshow(fft_vals[:, freq_mask], aspect='auto', origin='lower', cmap='inferno',
                        extent=[freqs[freq_mask][0], freqs[freq_mask][-1], 0, N])
        ax2.set_title('Пространственный спектр всей цепочки')
        ax2.set_ylabel('Узел (N)')
        ax2.set_xlabel('Частота (ω)')
        ax2.axvline(x=pump_freq, color='cyan', linestyle='--', linewidth=1.5)
        plt.colorbar(im, ax=ax2, label='Амплитуда')

        plt.tight_layout()
        
        stage_save_path = os.path.join(base_dir, f"{base_name}_stage_{stage+1}.png")
        plt.savefig(stage_save_path)
        plt.close()
        
    logging.info(f"   ...Комбинированные спектры сохранены в {base_dir}")
    
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

def calculate_initial_state_cartesian_decoupled(N, d, a, h, initial_strain_kick=0.15):
    """
    Генерация гибридного начального состояния с жестко заданной 
    амплитудой начального возмущения (initial_strain_kick), 
    которая не зависит от глобального B_ME_COUPLING.
    """
    import logging
    logging.info(f"-> Инициализация с фиксированным возмущением анизотропии: {initial_strain_kick}")

    Sx0 = np.zeros(N)
    Sy0 = np.zeros(N)

    center_idx = N // 2
    central_indices = np.arange(center_idx - 2, center_idx + 3)
    
    q_spiral = np.arctan(-d) 

    for p in central_indices:
        # Эффективная анизотропия задается ручным "пинком", а не через B_me * strain
        a_eff_local = a + initial_strain_kick
        
        denominator = 2 * (np.sqrt(1 + d**2) - 1) + 2 * a_eff_local

        if h > abs(denominator):
            theta_p = 0.0
        else:
            cos_theta_val = np.clip(h / denominator, -1.0, 1.0)
            theta_p = np.arccos(cos_theta_val)

        phi_p = q_spiral * p
        Sx0[p] = np.sin(theta_p) * np.cos(phi_p)
        Sy0[p] = np.sin(theta_p) * np.sin(phi_p)

    Y0 = np.column_stack((Sx0, Sy0)).ravel()
    Y0 += 1e-5 * np.random.randn(2 * N)
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
    """Отрисовывает кадр с расширенной сеткой 2x3 (добавлен Sy)."""
    t_current = t_vals[frame_idx]
    lattice_points = np.arange(N)
    
    # Делаем фигуру шире (18x10), сетка 2 строки х 3 колонки
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Dynamics | t = {t_current:.2f}', fontsize=16)

    # --- ВЕРХНИЙ РЯД: Компоненты спина ---
    
    # 1. Sx
    ax1 = axs[0, 0]
    ax1.plot(lattice_points, Sx_data[:, frame_idx], 'o-', color='royalblue', markersize=4)
    ax1.set_title(r'$S_x$ component')
    ax1.set_ylim(-1.05, 1.05); ax1.grid(True, alpha=0.4)

    # 2. Sy (НОВОЕ)
    ax2 = axs[0, 1]
    ax2.plot(lattice_points, Sy_data[:, frame_idx], 'o-', color='darkorange', markersize=4)
    ax2.set_title(r'$S_y$ component')
    ax2.set_ylim(-1.05, 1.05); ax2.grid(True, alpha=0.4)
    # Убираем метки Y, чтобы не загромождать (общая шкала)
    ax2.set_yticklabels([])

    # 3. Sz
    ax3 = axs[0, 2]
    ax3.plot(lattice_points, Sz_data[:, frame_idx], 'o-', color='forestgreen', markersize=4)
    ax3.set_title(r'$S_z$ component')
    ax3.set_ylim(-1.05, 1.05); ax3.grid(True, alpha=0.4)
    ax3.set_yticklabels([])

    # --- НИЖНИЙ РЯД: Физика и 3D ---

    # 4. Атомная накачка (q) - вернул, чтобы видеть причину движения
    ax4 = axs[1, 0]
    q_values = pump_func(lattice_points, t_current) 
    ax4.plot(lattice_points, q_values, '.-', color='gray', alpha=0.7)
    ax4.set_title(r'Atomic lattice ($q_n$)')
    ax4.set_ylim(-q_ylim, q_ylim); ax4.grid(True, alpha=0.4)
    ax4.set_xlabel('Site (n)')

    # 5. Энергия
    ax5 = axs[1, 1]
    ax5.plot(t_vals[:frame_idx+1], energies[:frame_idx+1], color='crimson', lw=1.5)
    ax5.set_title('Total Energy')
    ax5.set_xlabel('Time'); ax5.set_xlim(0, t_vals[-1])
    ax5.set_ylim(energy_ylim); ax5.grid(True, alpha=0.4)

    # 6. 3D Вид
    # Удаляем обычный subplot и добавляем 3d projection на его место
    fig.delaxes(axs[1, 2])
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.quiver(lattice_points, 0, 0, 
               Sz_data[:, frame_idx], Sx_data[:, frame_idx], Sy_data[:, frame_idx], 
               length=0.8, normalize=True, color='black', alpha=0.8)
    ax6.set_title('3D Spin View')
    ax6.set_ylim(-1, 1); ax6.set_zlim(-1, 1)
    ax6.set_xlabel('n'); ax6.set_ylabel('Sx'); ax6.set_zlabel('Sy')
    ax6.view_init(elev=20, azim=-60)

    # Текст с параметрами
    plt.figtext(0.01, 0.01, param_text, ha="left", va="bottom", fontsize=9, 
                bbox={"facecolor":"white", "alpha":0.9, "pad":3})
    
    plt.tight_layout()
    
    filepath = temp_dir / f"frame_{frame_idx:04d}.png"
    plt.savefig(filepath)
    plt.close(fig)
    return str(filepath)

def create_combined_visualization_parallel(Sz_data, Sx_data, Sy_data, energies, t_vals, pump_func, N, filename, param_text, temp_dir_keep_step=10):
    """Создает GIF и чистит кадры, оставляя каждый N-й."""
    logging.info("-> Шаг 4.1: Создание комбинированной GIF анимации...")
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
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        filenames = list(tqdm(pool.imap(task_func, range(len(t_vals))), total=len(t_vals), desc="   ...Генерация кадров"))

    filenames.sort()
    
    # Сборка GIF
    with imageio.get_writer(filename, mode='I', duration=1000/20, loop=0) as writer:
        for f in tqdm(filenames, desc="   ...Сборка GIF"):
            writer.append_data(imageio.v2.imread(f))
            
    # Умная очистка
    logging.info(f"   ...Очистка временных файлов (сохранение каждого {temp_dir_keep_step}-го)...")
    for i, f in enumerate(filenames):
        if i % temp_dir_keep_step != 0:
            try:
                os.remove(f)
            except OSError: pass
    
    logging.info(f"   ...GIF сохранен: {filename}")
# ==============================================================================
# --- ОСНОВНОЙ ИСПОЛНЯЕМЫЙ БЛОК ---
# ==============================================================================

def main():
    """Главная функция, запускающая весь процесс моделирования."""
    results_dir = Path(f"{RESULTS_FOLDER_PREFIX}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    results_dir.mkdir(exist_ok=True)
    setup_logging(results_dir / "simulation.log")

    logging.info("="*60 + "\n--- ЗАПУСК МОДЕЛИРОВАНИЯ ---\n" + "="*60)
    # ИСПРАВЛЕНИЕ 1: Заменили спецсимвол Ω на Omega во избежание ошибки кодировки Windows
    param_string = (f"ПАРАМЕТРЫ:\n" f"N={N_SITES}, T_END={T_MAGNETIC_END}, Omega={ATOMIC_OMEGA}, K={TIME_SCALING_FACTOR}\n"
                    f"D/2J={D_DMI:.2f}, A/2J={A_ANISOTROPY:.2f}, H/2J={H_EFF:.2f}, B_me={B_ME_COUPLING:.2f}\n"
                    f"РЕШАТЕЛЬ: {SOLVER_METHOD}, Rtol={SOLVER_RTOL}")
    logging.info("\n--- Стартовая конфигурация ---\n" + param_string + "\n" + "-"*28 + "\n")

    J_NORM = 2.0 * J_EXCHANGE 
    b_me_effective = B_ME_COUPLING if ENABLE_ME_COUPLING else 0.0

    atomic_pump = get_atomic_pumping_function(N_SITES, ATOMIC_OMEGA)
    Y0_magnetic = calculate_initial_state_cartesian_decoupled(N_SITES, D_DMI, A_ANISOTROPY, H_EFF, initial_strain_kick=0.0)
    
    logging.info("\n-> Шаг 3: Запуск динамического моделирования...")
    global pbar
    with tqdm(total=T_MAGNETIC_END, desc="   ...Интегрирование") as pbar_instance:
        pbar = pbar_instance
        sol_magnetic = solve_ivp(
            fun=magnetic_equations_with_progress, t_span=(0, T_MAGNETIC_END), y0=Y0_magnetic,
            method=SOLVER_METHOD, dense_output=True, rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
            args=(N_SITES, D_DMI, A_ANISOTROPY, H_EFF, b_me_effective, atomic_pump, TIME_SCALING_FACTOR),
        )
    pbar = None
    
    if not sol_magnetic.success:
        logging.error("!!! Решатель упал !!!")
        return

    logging.info("\n-> Шаг 4: Постобработка и визуализация...")
    
    # --- ЭТАП А: Данные для Фурье (Плотная сетка) ---
    N_FFT = 2000 # Высокое разрешение для спектра
    t_dense = np.linspace(0, T_MAGNETIC_END, N_FFT)
    Y_dense = sol_magnetic.sol(t_dense)
    Sx_dense = Y_dense[0::2, :]
    
    # Расчет эффективной частоты накачки (с учетом Time Scaling)
    # Если t_atomic = K * t_mag, то частота в магнитной системе = Omega * |K|
    effective_pump_freq = ATOMIC_OMEGA * abs(TIME_SCALING_FACTOR)
    
    perform_fft_analysis_stages(t_dense, Sx_dense, N_SITES, results_dir / "spectrum_heatmap.png", effective_pump_freq)

    # --- ЭТАП Б: Данные для GIF (Редкая сетка) ---
    t_gif = np.linspace(0, T_MAGNETIC_END, FRAME_COUNT)
    Y_gif = sol_magnetic.sol(t_gif)
    Sx_gif, Sy_gif = Y_gif[0::2, :], Y_gif[1::2, :]
    
    # Перенормировка для GIF
    norm = np.sqrt(np.maximum(Sx_gif**2 + Sy_gif**2, EPSILON))
    Sx_gif[norm > 1.0] /= norm[norm > 1.0]
    Sy_gif[norm > 1.0] /= norm[norm > 1.0]
    Sz_gif = np.sqrt(np.maximum(0, 1 - Sx_gif**2 - Sy_gif**2))

    energies = calculate_magnetic_energy(Sx_gif, Sy_gif, Sz_gif, t_gif, N_SITES, D_DMI, A_ANISOTROPY, H_EFF, b_me_effective, atomic_pump, TIME_SCALING_FACTOR)
    
    create_combined_visualization_parallel(Sz_gif, Sx_gif, Sy_gif, energies, t_gif, atomic_pump, N_SITES, results_dir / "dynamics.gif", param_string)

    logging.info("\n" + "="*60 + f"\nГОТОВО! Результаты в: {results_dir}\n" + "="*60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()