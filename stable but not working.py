# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
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
N_SITES = 101                   # Количество узлов в цепочке (нечетное для симметрии)
T_MAGNETIC_END = 50.0            # Полное время симуляции для магнитной системы
FRAME_COUNT = 100                # Количество кадров для итоговой анимации и графиков
RESULTS_FOLDER_PREFIX = "breather_run" # Префикс для папки с результатами

# --- Физические параметры: Атомная подсистема (модель ФПУ) ---
ATOMIC_OMEGA = 2.1               # Частота бризера. (Ω > 2.0)
ATOMIC_PERIODS = 5               # Сколько периодов бризера моделировать для получения профиля

# --- Физические параметры: Магнитная подсистема (все нормировано на J) ---
J_EXCHANGE = 1.0                 # Константа обменного взаимодействия, >0 для антиферромагнетика
D_DMI = 0.17                     # Константа взаимодействия Дзялошинского-Мория (D/J)
A_ANISOTROPY = 0.41              # Константа одноионной анизотропии "легкая ось" (A/J)
H_EFF = 0.01                     # Внешнее магнитное поле вдоль оси z (H/J)

# --- Параметры связи и масштабирования ---
B_ME_COUPLING = 5.0              # Константа магнитоэлектрической связи
ENABLE_ME_COUPLING = False        # Включить магнитоэлектрическую связь
TIME_SCALING_FACTOR = 1.0        # t_atomic = K * t_magnetic

# --- Параметры численного решателя ---
SOLVER_METHOD = 'DOP853'         # Рекомендуется для высокой точности
MAX_STEP = 0.1                   # Максимальный шаг для решателя


def fpu_equations(t, Y, N):
    """
    Система уравнений для модели Ферми-Паста-Улама (ФПУ) с кубической
    нелинейностью. Описывает динамику атомных смещений.

    Args:
        t (float): Текущее время (не используется в этой модели, но требуется решателем).
        Y (np.ndarray): Вектор состояния [q_1, ..., q_N, p_1, ..., p_N],
                        где q - смещения, p - импульсы.
        N (int): Число узлов в решетке.

    Returns:
        np.ndarray: Производная вектора состояния dY/dt.
    """
    q = Y[:N]
    p = Y[N:]
    
    dqdt = p
    
    # Используем np.roll для получения соседей для всей решетки сразу
    q_plus_1 = np.roll(q, -1)  # q[i+1]
    q_minus_1 = np.roll(q, 1)   # q[i-1]
    
    # Все вычисления выполняются над массивами, без циклов
    linear_force = q_plus_1 + q_minus_1 - 2 * q
    nonlinear_force = (q_plus_1 - q)**3 - (q - q_minus_1)**3
    
    dpdt = linear_force + nonlinear_force
    
    return np.concatenate([dqdt, dpdt])


def get_atomic_pumping_function(N, omega):
    """
    Моделирует атомный бризер для получения функции накачки F(p, t).
    Бризер — это внутренне локализованное нелинейное колебание в решетке.
    Здесь используется аналитическое решение для бризера Сиверса-Такено
    в приближении вращающейся волны (RWA) как начальное условие.

    Args:
        N (int): Число узлов.
        omega (float): Частота бризера.

    Returns:
        callable: Функция pumping_function(p, t_atomic), возвращающая
                  амплитуду смещения на узле p в момент времени t_atomic.
    """
    print("-> Шаг 1: Моделирование атомного бризера для получения функции накачки...")
    
    diff_sq = omega**2 - 4.0
    if diff_sq <= 0:
        raise ValueError("Omega^2 должен быть > 4.0 для существования бризера.")
    
    # Амплитуда бризера в центре из аналитического решения в RWA
    A_ilm = np.sqrt(diff_sq / 6.0)
    
    # Создание начального профиля бризера
    q0 = np.zeros(N)
    n0 = N // 2  # Центр решетки
    for n in range(N):
        factor = (-1)**n # "Staggered" или шахматный профиль, характерный для бризера
        arg = np.sqrt(6) * A_ilm * (n - n0)
        # Гиперболический секанс sech(x) = 2 / (e^x + e^-x)
        q0[n] = factor * A_ilm * (2.0 / (np.exp(arg) + np.exp(-arg)))
    
    # Мы могли бы просто использовать аналитическое решение, но запуск короткой симуляции уточняет профиль и делает его более стабильным.
    y0_atomic = np.concatenate([q0, np.zeros(N)]) # Начальные импульсы равны нулю
    t_end_atomic = (2 * np.pi / omega) * ATOMIC_PERIODS
    
    print("   ...Запускаю симуляцию атомной решетки для уточнения профиля...")
    sol = solve_ivp(
        fun=lambda t, y: fpu_equations(t, y, N),
        t_span=(0, t_end_atomic),
        y0=y0_atomic,
        method='RK45',
        dense_output=True
    )
    
    # Извлекаем максимальные амплитуды колебаний для каждого атома
    t_eval = np.linspace(0, t_end_atomic, 200) # Более плотная сетка для поиска максимума
    q_vals = sol.sol(t_eval)[:N]
    amplitudes = np.max(np.abs(q_vals), axis=1)
    
    print(f"   ...Амплитуды колебаний атомов найдены (макс. в центре: {amplitudes[n0]:.4f}).")

    # 1. Определяем функцию для аппроксимации (Гауссиана)
    def gaussian_profile(x, amplitude, center, sigma):
        return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))
    
    # 2. Аппроксимируем полученные амплитуды Гауссианой
    x_data = np.arange(N)
    initial_guess = [np.max(amplitudes), n0, 5.0]
    use_fit = False
    try:
        params, _ = curve_fit(gaussian_profile, x_data, amplitudes, p0=initial_guess)
        A_fit, center_fit, sigma_fit = params
        print("\n   --- Экстраполяция функции атомного бризера ---")
        print(f"   Успешная аппроксимация профиля Гауссианой:")
        print(f"   Amplitude(p) = {A_fit:.3f} * exp(-(p - {center_fit:.2f})^2 / (2 * {sigma_fit:.2f}^2))")
        print("   --------------------------------------------\n")
        use_fit = True
    except RuntimeError:
        print("   ...Аппроксимация не удалась, используется исходный численный профиль.")

    # 3. Создаем новую функцию накачки на основе аппроксимации
    def pumping_function(p, t_atomic):
        """
        Возвращает смещение атома 'p' в момент 't_atomic'.
        Использует либо аналитическую аппроксимацию, либо численные данные.
        """
        idx = p % N
        staggered_factor = (-1)**idx
    
        if use_fit:
            # Используем гладкую функцию
            amplitude_p = gaussian_profile(p, A_fit, center_fit, sigma_fit)
        else:
            # Возвращаемся к старому методу, если аппроксимация не удалась
            amplitude_p = amplitudes[idx]
        
        return staggered_factor * amplitude_p * np.cos(omega * t_atomic)
    
    return pumping_function



def calculate_conical_phase(N, J, D, A, H_eff):
    """
    Рассчитывает начальное состояние спинов - конусную (геликоидальную) фазу.
    Это основное состояние возникает из-за конкуренции между обменным 
    взаимодействием (J), Дзялошинского-Мория (D) и внешним полем (H_eff).

    Args:
        N (int): Число узлов.
        J, D, A, H_eff (float): Параметры магнитной системы.

    Returns:
        np.ndarray: Начальный вектор состояния для магнитной системы [Sx1, Sy1, Sx2, Sy2, ...].
    """
    print("-> Шаг 2: Расчет начального состояния (конусной фазы)...")

    # Эти формулы получаются минимизацией классической энергии системы
    denominator = J * (2 * (np.sqrt(1 + (D / (2 * J))**2) - 1) + (A / J))
    
    if np.abs(denominator) < 1e-9:
        cos_theta = 1.0 if H_eff > 0 else -1.0 # Полная поляризация, если знаменатель ноль
    else:
        cos_theta = H_eff / denominator
    
    # Ограничиваем значение, чтобы избежать ошибок в arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Волновой вектор спирали
    q_spiral = np.arctan(-D / (2 * J))
    
    print(f"   ...Параметры конусной фазы: q={q_spiral:.3f}, theta={np.degrees(theta):.2f}°")
    
    # Создаем начальный вектор спинов
    Y0 = np.zeros(2 * N)
    for p in range(N):
        phi_p = q_spiral * p
        Y0[2 * p]     = np.sin(theta) * np.cos(phi_p) # S_x
        Y0[2 * p + 1] = np.sin(theta) * np.sin(phi_p) # S_y
        # S_z = cos(theta) для всех спинов
        
    # Добавляем небольшое случайное возмущение, чтобы помочь системе
    # выйти из метастабильных состояний.
    noise_level = 1e-5
    Y0 += noise_level * np.random.randn(2 * N)
    print(f"   ...Добавлено начальное возмущение с уровнем {noise_level}.")
    
    return Y0


def magnetic_equations(t_mag, Y, N, J, D, A, H_eff, B_me, q_pump_func, K):
    """
    Система уравнений Ландау-Лифшица для динамики спинов.
    dS/dt = - [S x H_eff]
    где H_eff - эффективное поле, включающее все взаимодействия.

    Args:
        t_mag (float): Текущее время в магнитной подсистеме.
        Y (np.ndarray): Вектор состояния [Sx1, Sy1, Sx2, Sy2, ...].
        N (int): Число узлов.
        J, D, A, H_eff, B_me (float): Физические параметры.
        q_pump_func (callable): Функция атомной накачки.
        K (float): Масштаб времени.

    Returns:
        np.ndarray: Производная вектора состояния dY/dt.
    """
    Sx = Y[0::2]; Sy = Y[1::2]
    Sz = np.sqrt(np.maximum(0, 1 - Sx**2 - Sy**2))
    
    # Комплексное представление спина для всей решетки
    S_plus = Sx + 1j * Sy
    
    # --- Получаем соседей с помощью сдвига ---
    # Правые соседи
    Sz_p1 = np.roll(Sz, -1); S_plus_p1 = np.roll(S_plus, -1)
    # Левые соседи
    Sz_m1 = np.roll(Sz, 1); S_plus_m1 = np.roll(S_plus, 1)

    # --- Обменное взаимодействие ---
    term_exchange = -J * Sz * (S_plus_p1 + S_plus_m1) + J * S_plus * (Sz_p1 + Sz_m1)
    # "Обнуляем" взаимодействие на границах
    term_exchange[0] = -J * Sz[0] * S_plus_p1[0] + J * S_plus[0] * Sz_p1[0] # Нет левого соседа
    term_exchange[-1] = -J * Sz[-1] * S_plus_m1[-1] + J * S_plus[-1] * Sz_m1[-1] # Нет правого соседа

    # --- Взаимодействие Дзялошинского-Мория ---
    term_dmi = 1j * D * (Sz * S_plus_p1 - Sz_p1 * S_plus) - 1j * D * (S_plus_m1 * S_plus - Sz * S_plus_m1)
    # Обнуляем на границах
    term_dmi[0] = 1j * D * (Sz[0] * S_plus_p1[0] - Sz_p1[0] * S_plus[0])
    term_dmi[-1] = -1j * D * (S_plus_m1[-1] * S_plus[-1] - Sz[-1] * S_plus_m1[-1])

    # --- Одноузельные вклады ---
    term_anisotropy = 2 * A * Sz * S_plus
    term_zeeman = H_eff * S_plus
    
    # --- Магнитоэлектрическая связь ---
    t_atomic = K * t_mag
    p_indices = np.arange(N)
    q_p1 = np.array([q_pump_func(p + 1, t_atomic) for p in p_indices])
    q_m1 = np.array([q_pump_func(p - 1, t_atomic) for p in p_indices])
    q_p1[-1] = 0 # Нет правого соседа у последнего
    q_m1[0] = 0   # Нет левого соседа у первого
    term_me_coupling = B_me * (q_p1 - q_m1) * S_plus

    # --- Суммируем все ---
    RHS = term_exchange + term_dmi + term_anisotropy + term_zeeman + term_me_coupling
    dS_plus_dt = -1j * RHS
    
    dSx_dt = np.real(dS_plus_dt)
    dSy_dt = np.imag(dS_plus_dt)
    
    return np.column_stack((dSx_dt, dSy_dt)).ravel()

def calculate_magnetic_energy(Y_sol, t_sol, N, J, D, A, H_eff, B_me, q_pump_func, K):
    """
    Вычисляет полную энергию магнитной подсистемы в каждый момент времени.
    
    Args:
        Y_sol (np.ndarray): Решение системы уравнений (массив спинов).
        t_sol (np.ndarray): Временные точки.
        (остальные параметры как в magnetic_equations)

    Returns:
        np.ndarray: Массив значений энергии.
    """
    energies = []
    Sx = Y_sol[0::2, :]; Sy = Y_sol[1::2, :]
    Sz = np.sqrt(np.maximum(0, 1 - Sx**2 - Sy**2))

    print("   ...Расчет эволюции энергии...")
    p_indices = np.arange(N)

    for i in tqdm(range(len(t_sol)), desc="      ...Прогресс"):
        t_atomic = K * t_sol[i]
        
        # --- Одноузельные вклады ---
        E_anisotropy = -A * np.sum(Sz[:, i]**2)
        E_zeeman = -H_eff * np.sum(Sz[:, i])
        
        # --- Магнитоэлектрический вклад ---
        q_p1 = np.array([q_pump_func(p + 1, t_atomic) for p in p_indices])
        q_m1 = np.array([q_pump_func(p - 1, t_atomic) for p in p_indices])
        q_p1[-1] = 0; q_m1[0] = 0
        E_me_coupling = -B_me * np.sum((q_p1 - q_m1) * Sz[:, i])

        # --- Двухузельные вклады (срезами для открытых границ) ---
        # Взаимодействие между узлами p и p+1, считаем для p от 0 до N-2
        Sx_p = Sx[:-1, i]; Sx_p1 = Sx[1:, i]
        Sy_p = Sy[:-1, i]; Sy_p1 = Sy[1:, i]
        Sz_p = Sz[:-1, i]; Sz_p1 = Sz[1:, i]

        E_exchange = -J * np.sum(Sx_p*Sx_p1 + Sy_p*Sy_p1 + Sz_p*Sz_p1)
        E_dmi = -D * np.sum(Sx_p*Sy_p1 - Sy_p*Sx_p1)
        
        total_energy = E_anisotropy + E_zeeman + E_me_coupling + E_exchange + E_dmi
        energies.append(total_energy)
        
    return np.array(energies)


# ==============================================================================
# --- ФУНКЦИИ ВИЗУАЛИЗАЦИИ ---
# ==============================================================================

def create_combined_visualization(Sz_data, Sx_data, Sy_data, energies, t_vals, pump_func, N, omega, filename, param_text):
    """
    Создает комбинированную GIF анимацию, включающую 4 графика.
    """
    print("-> Шаг 4.1: Создание комбинированной GIF анимации (4 в 1)...")
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    filenames = []
    lattice_points = np.arange(N)
    min_energy, max_energy = np.min(energies), np.max(energies)

    # Находим абсолютную максимальную амплитуду для фиксации оси Y
    max_abs_q = 0
    for p in lattice_points:
        # Оцениваем по амплитуде из функции Гаусса или численной
        amp_p = np.abs(pump_func(p, 0) / np.cos(0)) # Делим на cos(0) чтобы получить амплитуду
        if amp_p > max_abs_q:
            max_abs_q = amp_p
    q_ylim = max_abs_q * 1.15 # Добавим небольшой запас

    for i in tqdm(range(len(t_vals)), desc="   ...Генерация кадров"):
        t_current = t_vals[i]
        
        # Создаем фигуру с сеткой 2x2
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Полная динамика системы | Время = {t_current:.2f}', fontsize=16)

        # --- 1. Атомный бризер ---
        ax1 = axs[0, 0]
        q_values = [pump_func(p, t_current) for p in lattice_points]
        ax1.plot(lattice_points, q_values, 'o-', color='green')
        ax1.set_title('1. Колебания атомной решетки (q_p)')
        ax1.set_xlabel('Узел решетки (p)')
        ax1.set_ylabel('Смещение (q)')
        ax1.set_ylim(-q_ylim, q_ylim)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- 2. Эволюция S_z ---
        ax2 = axs[0, 1]
        ax2.plot(lattice_points, Sz_data[:, i], 'o-', color='royalblue')
        ax2.set_title('2. Эволюция S_z компоненты')
        ax2.set_xlabel('Узел решетки (p)')
        ax2.set_ylabel('S_z')
        ax2.set_ylim(-1.05, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.6)

        # --- 3. 3D Эволюция спинов ---
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.set_title('3. 3D представление спинов')
        # Рисуем "стрелки" (векторы) для каждого спина
        ax3.quiver(lattice_points, 0, 0, Sx_data[:, i], Sy_data[:, i], Sz_data[:, i], length=0.8, normalize=True)
        ax3.set_xlim(0, N)
        ax3.set_ylim(-1, 1)
        ax3.set_zlim(-1, 1)
        ax3.set_xlabel('Узел (p)')
        ax3.set_ylabel('S_y')
        ax3.set_zlabel('S_z')
        ax3.view_init(elev=20, azim=-75) # Подбираем хороший ракурс

        # --- 4. Эволюция энергии ---
        ax4 = axs[1, 1]
        ax4.plot(t_vals[:i+1], energies[:i+1], color='crimson')
        ax4.set_title('4. Эволюция полной энергии')
        ax4.set_xlabel('Время (t)')
        ax4.set_ylabel('Энергия (E)')
        ax4.set_xlim(0, t_vals[-1])
        ax4.set_ylim(min_energy - abs(min_energy)*0.05, max_energy + abs(max_energy)*0.05)
        ax4.grid(True, linestyle='--', alpha=0.6)

        # Добавляем общую информацию о параметрах
        plt.figtext(0.02, 0.02, param_text, ha="left", va="bottom", fontsize=10,
                    bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Оставляем место для suptitle и figtext
        
        filepath = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(filepath)
        plt.close(fig)
        filenames.append(filepath)
        
    # Сборка кадров в GIF
    with imageio.get_writer(filename, mode='I', duration=1000/20) as writer: # 20 fps
        for f in tqdm(filenames, desc="   ...Сборка GIF"):
            image = imageio.v2.imread(f)
            writer.append_data(image)
            
    # Очистка
    for f in filenames: os.remove(f)
    os.rmdir(temp_dir)
    print(f"   ...Комбинированный GIF файл сохранен: {filename}")

# ==============================================================================
# --- ОСНОВНОЙ ИСПОЛНЯЕМЫЙ БЛОК ---
# ==============================================================================

def main():
    """
    Главная функция, запускающая весь процесс моделирования.
    """
    print("="*60)
    print("--- ЗАПУСК МОДЕЛИРОВАНИЯ СПИНОВОЙ ДИНАМИКИ ПОД ДЕЙСТВИЕМ БРИЗЕРА ---")
    print("="*60)

    # Создаем папку для результатов с уникальным именем (отметка времени)
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"{RESULTS_FOLDER_PREFIX}_{run_timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # --- Формируем строку с параметрами для легенды на графиках ---
    param_string = (
        f"Параметры симуляции:\n"
        f"N_SITES = {N_SITES}, T_END = {T_MAGNETIC_END}\n"
        f"--- Атомная подсистема ---\n"
        f"Ω = {ATOMIC_OMEGA}\n"
        f"--- Магнитная подсистема (норм. на J) ---\n"
        f"D/J = {D_DMI:.2f}, A/J = {A_ANISOTROPY:.2f}, H_eff/J = {H_EFF:.2f}\n"
        f"B_me = {B_ME_COUPLING:.2f}, K (time_scale) = {TIME_SCALING_FACTOR:.2f}"
    )
    print("\n--- Стартовая конфигурация ---")
    print(param_string)
    print("----------------------------\n")

    J_NORM = 2.0 * J_EXCHANGE
    print(f"Используется нормировка на 2J = {J_NORM}\n")

    b_me_effective = B_ME_COUPLING if ENABLE_ME_COUPLING else 0.0
    print(f"Магнитоупругое взаимодействие: {'ВКЛЮЧЕНО' if ENABLE_ME_COUPLING else 'ВЫКЛЮЧЕНО'}")

    # --- Шаг 1: Получение функции накачки от атомного бризера ---
    atomic_pump = get_atomic_pumping_function(N_SITES, ATOMIC_OMEGA)
    
    # --- Шаг 2: Расчет начального состояния магнитной системы ---
    Y0_magnetic = calculate_conical_phase(N_SITES, J_EXCHANGE / J_NORM, D_DMI / J_NORM, A_ANISOTROPY / J_NORM, H_EFF / J_NORM)
    
    # --- Шаг 3: Основная симуляция динамики ---
    print("\n-> Шаг 3: Запуск динамического моделирования магнитной системы...")
    sol_magnetic = solve_ivp(
        fun=magnetic_equations,
        t_span=(0, T_MAGNETIC_END),
        y0=Y0_magnetic,
        method=SOLVER_METHOD,
        dense_output=True,
        max_step=MAX_STEP,
        args=(N_SITES, J_EXCHANGE / J_NORM, D_DMI / J_NORM, A_ANISOTROPY / J_NORM, H_EFF / J_NORM, b_me_effective / J_NORM, atomic_pump, TIME_SCALING_FACTOR),
    )
    print("   ...Динамическое моделирование завершено.")

    # --- Шаг 4: Постобработка и визуализация ---
    print("\n-> Шаг 4: Постобработка и визуализация результатов...")
    
    # Получаем решения в заданных временных точках
    t_eval = np.linspace(0, T_MAGNETIC_END, FRAME_COUNT)
    Y_eval = sol_magnetic.sol(t_eval)
    
    Sx_eval = Y_eval[0::2, :]
    Sy_eval = Y_eval[1::2, :]
    Sz_eval = np.sqrt(np.maximum(0, 1 - Sx_eval**2 - Sy_eval**2))

    energies = calculate_magnetic_energy(
    Y_eval, t_eval, N_SITES, J_EXCHANGE / J_NORM, D_DMI / J_NORM, A_ANISOTROPY / J_NORM, 
    H_EFF / J_NORM, b_me_effective / J_NORM, atomic_pump, TIME_SCALING_FACTOR
)

    gif_path = os.path.join(results_dir, "combined_dynamics.gif")
    create_combined_visualization(
    Sz_eval, Sx_eval, Sy_eval, energies, t_eval, 
    atomic_pump, N_SITES, ATOMIC_OMEGA, gif_path, param_string
)

    print("\n" + "="*60)
    print("--- МОДЕЛИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО ---")
    print(f"Результаты сохранены в папке: {results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()