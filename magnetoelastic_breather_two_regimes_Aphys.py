# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Исправленная версия расчета магнитоупругого возбуждения магнитных бризеров.

Что именно здесь исправлено относительно предыдущего файла.
----------------------------------------------------------
1. Ось цепочки фиксирована как ось Z магнитной модели.
   То есть Sx, Sy -- поперечные компоненты, а Sz -- продольная компонента
   вдоль цепочки и внешнего поля.

2. Режимы dark / bright диагностируются не по отдельным Sx и Sy, а по
   инвариантным величинам:
       S_perp = sqrt(Sx^2 + Sy^2),
       Sz.
   Именно эти величины отвечают на физический вопрос, лежат ли спины
   поперек оси цепочки или вдоль нее.

3. Для автономного магнитного профиля используется не только snapshot в lab frame,
   но и точная вращающаяся фаза статьи:
       s_n^+(tau) = s_n * exp(i k0 n - i omega_lab tau),
       omega_lab = Omega + beta.
   Поэтому теперь можно корректно смотреть и lab-frame картину, и профиль
   в rotating frame.

4. Все графики, по которым нужно судить о соответствии статье, переведены на
   язык S_perp, Sz и rotating-frame профиля. Сырые карты Sx/Sy оставлены
   только как дополнительная информация.

5. Исправлена опечатка в имени корневой папки результатов.

Скрипт рассчитан на запуск из IDE без аргументов.
Все комментарии написаны по-русски.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# =============================================================================
# --- ОБЩИЕ НАСТРОЙКИ ---
# =============================================================================

RUN_MODES = (
    "bottom_dark",
    "bottom_bright",
)

RESULTS_ROOT = Path("magnetoelastic_two_regimes_results_fixed")
PLOT_DPI = 170
SAVE_NUMPY_ARCHIVE = True

# Возвращаем две gif-анимации из исходной версии:
# 13_profiles_stacked_components.gif
# 14_spin_3d_orientation.gif
SAVE_GIFS = True
ANIMATION_MAX_FRAMES = 220*4
ANIMATION_FPS = 15
ANIMATION_TITLE_FONTSIZE = 18
ANIMATION_LABEL_FONTSIZE = 15
ANIMATION_TICK_FONTSIZE = 13

SPIN_3D_FIGSIZE = (18.0, 13.0)
SPIN_3D_SLOWDOWN_FACTOR = 12
SPIN_3D_TITLE_FONTSIZE = 20
SPIN_3D_LABEL_FONTSIZE = 17
SPIN_3D_TICK_FONTSIZE = 13
SPIN_3D_SUPTITLE_FONTSIZE = 16


# =============================================================================
# --- АТОМНАЯ ПОДСИСТЕМА ---
# =============================================================================

N_SITES = 101
ATOMIC_OMEGA = 2.1
ATOMIC_RELAX_PERIODS = 8
ATOMIC_SAMPLES_PER_PERIOD = 1200


# =============================================================================
# --- МАГНИТНАЯ ДИНАМИКА И МАГНИТОУПРУГАЯ СВЯЗЬ ---
# =============================================================================

RHO_TIME_SCALING = 0.02

# Производные J и D по деформации связи.
# Именно они соответствуют разложению J(ell) и D(ell) около равновесной длины.
KAPPA_J = 0.04
KAPPA_D = 0.02

# Константа магнитоупругой связи для анизотропии.
# Здесь A не раскладывается по деформации как самостоятельная функция.
# Временная зависимость идет только через локальную координатную/деформационную
# переменную bar_eta_n(t), а сама константа B_A от времени не зависит:
#     A_n(t) = A_0 + B_A * bar_eta_n(t).
B_A = 0.15

ENABLE_J_DRIVE = True
ENABLE_D_DRIVE = True
ENABLE_A_DRIVE = True

# Если True, в драйве используются нормированные профили, как в старой версии.
# Это удобно для сохранения прежнего масштаба численных экспериментов.
# Если False, используются ненормированные центрированные величины bond_eta/site_eta,
# что ближе к буквальной записи формул из заметки.
USE_NORMALIZED_DRIVE = True

ALPHA_BULK = 3.0e-5
ALPHA_EDGE_MAX = 8.0e-4
EDGE_DAMPING_WIDTH = 14

TAU_RAMP = 60.0
TAU_END = 540.0*4
MAGNETIC_TIME_SAMPLES = 2200*2

# Для верификации статьи шум по умолчанию выключен.
ADD_TINY_NOISE = False
NOISE_LEVEL = 1e-6


# =============================================================================
# --- ДАТАКЛАССЫ ---
# =============================================================================

@dataclass
class AtomicBreatherData:
    q_period: np.ndarray
    p_period: np.ndarray
    bond_eta: np.ndarray
    site_eta: np.ndarray
    bond_eta_norm: np.ndarray
    site_eta_norm: np.ndarray
    amplitude_profile: np.ndarray
    period: float


@dataclass
class RegimeParameters:
    name: str
    description: str
    d_dmi: float          # D / (2J)
    b_anisotropy: float   # B = A / (2J)
    beta_field: float     # beta = H / (2JS)
    omega_eff: float      # Omega = (hbar*omega - H) / (2JS)
    seed_amplitude: float
    seed_width: float
    seed_kind: str        # "bright" или "dark"
    article_hint: str


@dataclass
class MagneticSolution:
    tau: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sz: np.ndarray
    s_perp: np.ndarray
    s_rot_real: np.ndarray
    s_rot_imag: np.ndarray
    s_rot_abs: np.ndarray
    energy: np.ndarray
    ipr: np.ndarray
    center_signal: np.ndarray
    local_b_center: np.ndarray
    local_j_center: np.ndarray
    local_d_center: np.ndarray


# =============================================================================
# --- ПРЕСЕТЫ РЕЖИМОВ ---
# =============================================================================

REGIMES: dict[str, RegimeParameters] = {
    "bottom_dark": RegimeParameters(
        name="bottom_dark",
        description="Нижний темный бризер: easy-plane режим, релевантный статье для CrNb3S6.",
        d_dmi=0.16,
        b_anisotropy=0.15,
        beta_field=1.50,
        omega_eff=-0.28,
        seed_amplitude=0.50,
        seed_width=10.0,
        seed_kind="dark",
        article_hint="D/2J=0.16, B=0.15, Omega=-0.28, beta=1.5.",
    ),
    "bottom_bright": RegimeParameters(
        name="bottom_bright",
        description="Нижний светлый бризер: easy-axis режим статьи при B=-1.0 и Omega=1.95.",
        d_dmi=0.16,
        b_anisotropy=-1.0,
        beta_field=0.40,
        omega_eff=1.95,
        seed_amplitude=0.25,
        seed_width=6.0,
        seed_kind="bright",
        article_hint="D/2J=0.16, B=-1.0, Omega=1.95.",
    ),
}


# =============================================================================
# --- АТОМНАЯ ЗАДАЧА FPU-BETA ---
# =============================================================================


def fpu_equations(_t: float, y: np.ndarray, n_sites: int) -> np.ndarray:
    q = y[:n_sites]
    p = y[n_sites:]

    q_plus = np.roll(q, -1)
    q_minus = np.roll(q, 1)

    dqdt = p
    linear_force = q_plus + q_minus - 2.0 * q
    nonlinear_force = (q_plus - q) ** 3 - (q - q_minus) ** 3
    dpdt = linear_force + nonlinear_force

    return np.concatenate([dqdt, dpdt])



def build_atomic_initial_guess(n_sites: int, omega: float) -> np.ndarray:
    diff_sq = omega ** 2 - 4.0
    if diff_sq <= 0.0:
        raise ValueError("Для hard atomic breather требуется omega^2 > 4.")

    a_ilm = math.sqrt(diff_sq / 6.0)
    center = n_sites // 2
    x = np.arange(n_sites, dtype=float)

    q0 = ((-1.0) ** x) * a_ilm * (2.0 / np.cosh(math.sqrt(6.0) * a_ilm * (x - center)))
    p0 = np.zeros(n_sites, dtype=float)

    return np.concatenate([q0, p0])



def simulate_atomic_breather() -> AtomicBreatherData:
    y0 = build_atomic_initial_guess(N_SITES, ATOMIC_OMEGA)
    atomic_period = 2.0 * math.pi / ATOMIC_OMEGA
    t_end = ATOMIC_RELAX_PERIODS * atomic_period

    sol = solve_ivp(
        fun=lambda t, y: fpu_equations(t, y, N_SITES),
        t_span=(0.0, t_end),
        y0=y0,
        method="RK45",
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Ошибка атомного решателя: {sol.message}")

    t0 = t_end - atomic_period
    t_period = np.linspace(t0, t_end, ATOMIC_SAMPLES_PER_PERIOD, endpoint=False)
    y_period = sol.sol(t_period)

    q_period = y_period[:N_SITES]
    p_period = y_period[N_SITES:]

    bond_raw = q_period[1:, :] - q_period[:-1, :]

    bond_full = np.zeros_like(q_period)
    bond_full[:-1, :] = bond_raw
    bond_full[-1, :] = bond_raw[-1, :]

    site_raw = np.zeros_like(q_period)
    site_raw[0, :] = 0.5 * bond_raw[0, :]
    site_raw[-1, :] = 0.5 * bond_raw[-1, :]
    site_raw[1:-1, :] = 0.5 * (bond_raw[:-1, :] + bond_raw[1:, :])

    bond_centered = bond_full - np.mean(bond_full, axis=1, keepdims=True)
    site_centered = site_raw - np.mean(site_raw, axis=1, keepdims=True)

    bond_scale = max(float(np.max(np.abs(bond_centered))), 1e-12)
    site_scale = max(float(np.max(np.abs(site_centered))), 1e-12)

    bond_norm = bond_centered / bond_scale
    site_norm = site_centered / site_scale
    amplitude_profile = np.max(np.abs(q_period), axis=1)

    # В размерностях этой FPU-модели решетка уже безразмерна, поэтому удобно
    # трактовать bond_centered и site_centered как безразмерные деформационные поля.
    return AtomicBreatherData(
        q_period=q_period,
        p_period=p_period,
        bond_eta=bond_centered,
        site_eta=site_centered,
        bond_eta_norm=bond_norm,
        site_eta_norm=site_norm,
        amplitude_profile=amplitude_profile,
        period=atomic_period,
    )


# =============================================================================
# --- СЛУЖЕБНЫЕ ФУНКЦИИ ДРАЙВА ---
# =============================================================================


def periodic_sample(table: np.ndarray, period: float, time_value: float) -> np.ndarray:
    n_time = table.shape[1]
    reduced = time_value % period
    idx_float = (reduced / period) * n_time

    i0 = int(np.floor(idx_float)) % n_time
    i1 = (i0 + 1) % n_time
    frac = idx_float - np.floor(idx_float)

    return (1.0 - frac) * table[:, i0] + frac * table[:, i1]



def smooth_ramp(time_value: float, ramp_end: float) -> float:
    if ramp_end <= 0.0:
        return 1.0
    if time_value <= 0.0:
        return 0.0
    if time_value >= ramp_end:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * time_value / ramp_end))




def get_magnetoelastic_fields(
    tau_magnetic: float,
    atomic: AtomicBreatherData,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает поля деформации, которые управляют магнитными коэффициентами.

    bond_eta соответствует деформации связи eta_n(t), влияющей на J_n и D_n.
    site_eta соответствует локальной site-переменной bar_eta_n(t), через которую
    модулируется анизотропия A_n.
    """
    tau_atomic = RHO_TIME_SCALING * tau_magnetic

    if USE_NORMALIZED_DRIVE:
        bond_eta = periodic_sample(atomic.bond_eta_norm, atomic.period, tau_atomic)
        site_eta = periodic_sample(atomic.site_eta_norm, atomic.period, tau_atomic)
    else:
        bond_eta = periodic_sample(atomic.bond_eta, atomic.period, tau_atomic)
        site_eta = periodic_sample(atomic.site_eta, atomic.period, tau_atomic)

    return bond_eta, site_eta


# =============================================================================
# --- АВТОНОМНЫЙ МАГНИТНЫЙ ПРОФИЛЬ В ROTATING FRAME ---
# =============================================================================


def stationary_profile_equations_open(s: np.ndarray, params: RegimeParameters) -> np.ndarray:
    """
    Eq. (3) статьи при ds_n/dtau = 0 и k0 a = -atan(D/2J), так что cos(k0 a + delta)=1.
    """
    q0 = params.d_dmi
    omega = params.omega_eff
    b = params.b_anisotropy
    coupling = math.sqrt(1.0 + q0 * q0)

    s_clip = np.clip(s, -0.999999, 0.999999)
    z = np.sqrt(np.clip(1.0 - s_clip ** 2, 0.0, 1.0))
    residual = np.zeros_like(s_clip)

    for n in range(len(s_clip)):
        s_left = s_clip[n - 1] if n > 0 else 0.0
        s_right = s_clip[n + 1] if n < len(s_clip) - 1 else 0.0
        z_left = z[n - 1] if n > 0 else 0.0
        z_right = z[n + 1] if n < len(s_clip) - 1 else 0.0

        residual[n] = (
            -omega * s_clip[n]
            + s_clip[n] * (z_left + z_right)
            - (s_left + s_right) * coupling * z[n]
            - 2.0 * b * s_clip[n] * z[n]
        )

    return residual



def build_stationary_seed(params: RegimeParameters) -> np.ndarray:
    n = np.arange(N_SITES, dtype=float)
    center = (N_SITES - 1) / 2.0
    x = n - center

    if params.seed_kind == "bright":
        seed = params.seed_amplitude / np.cosh(x / params.seed_width)
    elif params.seed_kind == "dark":
        seed = params.seed_amplitude * np.tanh(x / params.seed_width)
    else:
        raise ValueError(f"Неизвестный тип seed: {params.seed_kind}")

    return np.clip(seed, -0.95, 0.95)



def refine_stationary_profile(params: RegimeParameters) -> np.ndarray:
    base_seed = build_stationary_seed(params)

    candidates: list[np.ndarray] = [
        base_seed,
        0.9 * base_seed,
        1.1 * base_seed,
    ]

    if params.seed_kind == "bright":
        n = np.arange(N_SITES, dtype=float)
        center = (N_SITES - 1) / 2.0
        x = n - center
        candidates.append(params.seed_amplitude / np.cosh(x / (params.seed_width * 1.2)))

    best_x = None
    best_cost = np.inf
    lower = -0.999 * np.ones(N_SITES)
    upper = 0.999 * np.ones(N_SITES)

    for x0 in candidates:
        result = least_squares(
            fun=lambda s: stationary_profile_equations_open(s, params),
            x0=np.clip(x0, -0.95, 0.95),
            bounds=(lower, upper),
            method="trf",
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            max_nfev=80,
            verbose=0,
        )

        cost = float(np.linalg.norm(result.fun))
        if cost < best_cost:
            best_cost = cost
            best_x = result.x.copy()

        if cost < 1e-8:
            break

    if best_x is None:
        raise RuntimeError("Не удалось построить автономный магнитный профиль.")

    if params.seed_kind == "bright":
        best_x = 0.5 * (best_x + best_x[::-1])

    return np.clip(best_x, -0.999, 0.999)


# =============================================================================
# --- ПЕРЕХОД МЕЖДУ ROTATING FRAME И LAB FRAME ---
# =============================================================================


def carrier_wave_number(params: RegimeParameters) -> float:
    return -math.atan(params.d_dmi)



def omega_lab(params: RegimeParameters) -> float:
    """
    В lab frame фаза вращается с частотой omega_lab = Omega + beta.
    Это следует из определения эффективной частоты статьи:
        Omega = (hbar * omega - H) / (2JS).
    """
    return params.omega_eff + params.beta_field



def autonomous_lab_spins(profile: np.ndarray, tau_value: float, params: RegimeParameters) -> np.ndarray:
    """
    Точная автономная орбита, соответствующая ansatz статьи:
        s_n^+(tau) = s_n * exp(i k0 n - i omega_lab tau).
    """
    n = np.arange(N_SITES, dtype=float)
    phase = carrier_wave_number(params) * n - omega_lab(params) * tau_value
    s_plus = profile * np.exp(1j * phase)

    sx = np.real(s_plus)
    sy = np.imag(s_plus)
    sz = np.sqrt(np.clip(1.0 - np.abs(s_plus) ** 2, 0.0, 1.0))

    if ADD_TINY_NOISE:
        rng = np.random.default_rng(12345)
        sx = sx + NOISE_LEVEL * rng.standard_normal(N_SITES)
        sy = sy + NOISE_LEVEL * rng.standard_normal(N_SITES)
        sz = np.sqrt(np.clip(1.0 - sx ** 2 - sy ** 2, 0.0, 1.0))

    return np.column_stack((sx, sy, sz))



def rotating_frame_fields(sx: np.ndarray, sy: np.ndarray, tau: np.ndarray, params: RegimeParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Переводит решение из lab frame в rotating frame статьи.

    Важно: именно здесь становится видно, что s_n в темном режиме велико по модулю
    на краях и мало в центре, а в светлом наоборот. По отдельным Sx/Sy это судить нельзя.
    """
    n = np.arange(N_SITES, dtype=float)[:, None]
    phase = carrier_wave_number(params) * n - omega_lab(params) * tau[None, :]
    s_plus = sx + 1j * sy
    s_rot = s_plus * np.exp(-1j * phase)
    return np.real(s_rot), np.imag(s_rot), np.abs(s_rot)


# =============================================================================
# --- ЛОКАЛЬНЫЕ КОЭФФИЦИЕНТЫ ПОД ДРАЙВОМ ОТ АТОМНОГО БРИЗЕРА ---
# =============================================================================


def get_local_coefficients(
    tau_magnetic: float,
    atomic: AtomicBreatherData,
    params: RegimeParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ramp = smooth_ramp(tau_magnetic, TAU_RAMP)
    bond_eta, site_eta = get_magnetoelastic_fields(tau_magnetic, atomic)

    j_local = np.ones(N_SITES, dtype=float)
    d_local = np.full(N_SITES, params.d_dmi, dtype=float)
    b_local = np.full(N_SITES, params.b_anisotropy, dtype=float)

    # J_n(t) = J_0 * [1 + kappa_J * eta_n(t)]
    if ENABLE_J_DRIVE:
        j_local = j_local + ramp * KAPPA_J * bond_eta

    # D_n(t) = D_0 + kappa_D * eta_n(t)
    if ENABLE_D_DRIVE:
        d_local = d_local + ramp * KAPPA_D * bond_eta

    # Для анизотропии НЕ используется разложение A(eta).
    # Здесь вводится прямой магнитоупругий вклад с постоянной константой связи:
    #     A_n(t) = A_0 + B_A * bar_eta_n(t).
    # После безразмеривания это дает локальный коэффициент b_n(t).
    if ENABLE_A_DRIVE:
        b_local = b_local + ramp * B_A * site_eta

    return j_local, d_local, b_local


# =============================================================================
# --- ПРОФИЛЬ КРАЕВОГО ЗАТУХАНИЯ ---
# =============================================================================


def build_alpha_profile() -> np.ndarray:
    alpha = np.full(N_SITES, ALPHA_BULK, dtype=float)
    if EDGE_DAMPING_WIDTH <= 0:
        return alpha

    for i in range(EDGE_DAMPING_WIDTH):
        frac = (i + 1) / EDGE_DAMPING_WIDTH
        add = (frac ** 2) * (ALPHA_EDGE_MAX - ALPHA_BULK)
        alpha[i] = max(alpha[i], ALPHA_BULK + add)
        alpha[-1 - i] = max(alpha[-1 - i], ALPHA_BULK + add)

    return alpha


ALPHA_PROFILE = build_alpha_profile()


# =============================================================================
# --- ПОЛНАЯ СПИНОВАЯ ДИНАМИКА В ФОРМЕ LL ---
# =============================================================================


def magnetic_rhs_ll(
    tau_magnetic: float,
    y: np.ndarray,
    atomic: AtomicBreatherData,
    params: RegimeParameters,
) -> np.ndarray:
    spins = y.reshape(N_SITES, 3)

    norms = np.linalg.norm(spins, axis=1, keepdims=True)
    norms = np.where(norms > 1e-14, norms, 1.0)
    spins = spins / norms

    j_local, d_local, b_local = get_local_coefficients(tau_magnetic, atomic, params)
    h_eff = np.zeros_like(spins)

    for n in range(N_SITES):
        if n < N_SITES - 1:
            sr = spins[n + 1]
            jr = j_local[n]
            dr = d_local[n]
            h_eff[n, 0] += jr * sr[0] - dr * sr[1]
            h_eff[n, 1] += jr * sr[1] + dr * sr[0]
            h_eff[n, 2] += jr * sr[2]

        if n > 0:
            sl = spins[n - 1]
            jl = j_local[n - 1]
            dl = d_local[n - 1]
            h_eff[n, 0] += jl * sl[0] + dl * sl[1]
            h_eff[n, 1] += jl * sl[1] - dl * sl[0]
            h_eff[n, 2] += jl * sl[2]

        h_eff[n, 2] += params.beta_field - 2.0 * b_local[n] * spins[n, 2]

    precession = -np.cross(spins, h_eff)
    damping = -ALPHA_PROFILE[:, None] * np.cross(spins, np.cross(spins, h_eff))
    dstates = precession + damping

    return dstates.ravel()


# =============================================================================
# --- ЭНЕРГИЯ И ДИАГНОСТИКИ ---
# =============================================================================


def compute_energy_snapshot(
    spins: np.ndarray,
    tau_magnetic: float,
    atomic: AtomicBreatherData,
    params: RegimeParameters,
) -> float:
    j_local, d_local, b_local = get_local_coefficients(tau_magnetic, atomic, params)

    sx = spins[:, 0]
    sy = spins[:, 1]
    sz = spins[:, 2]
    energy = 0.0

    for n in range(N_SITES - 1):
        exchange = -j_local[n] * (sx[n] * sx[n + 1] + sy[n] * sy[n + 1] + sz[n] * sz[n + 1])
        dmi = d_local[n] * (sx[n] * sy[n + 1] - sy[n] * sx[n + 1])
        energy += float(exchange + dmi)

    energy += float(np.sum(b_local * sz ** 2))
    energy += float(-params.beta_field * np.sum(sz))
    return energy



def dominant_frequency(signal: np.ndarray, time_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centered = signal - np.mean(signal)
    dt = float(time_grid[1] - time_grid[0])
    fft_vals = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered), dt)
    if len(freqs) <= 1:
        return freqs, fft_vals, 0.0
    peak_idx = int(np.argmax(fft_vals[1:]) + 1)
    return freqs, fft_vals, float(freqs[peak_idx])


# =============================================================================
# --- ОСНОВНОЙ РАСЧЕТ ---
# =============================================================================


def solve_magnetic_dynamics(atomic: AtomicBreatherData, params: RegimeParameters) -> tuple[np.ndarray, MagneticSolution]:
    profile = refine_stationary_profile(params)
    y0 = autonomous_lab_spins(profile, tau_value=0.0, params=params).ravel()

    tau_grid = np.linspace(0.0, TAU_END, MAGNETIC_TIME_SAMPLES)
    sol = solve_ivp(
        fun=lambda t, y: magnetic_rhs_ll(t, y, atomic, params),
        t_span=(tau_grid[0], tau_grid[-1]),
        y0=y0,
        t_eval=tau_grid,
        method="DOP853",
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"Магнитный решатель завершился с ошибкой: {sol.message}")

    states = sol.y.T.reshape(len(tau_grid), N_SITES, 3)
    norms = np.linalg.norm(states, axis=2, keepdims=True)
    norms = np.where(norms > 1e-14, norms, 1.0)
    states = states / norms

    sx = states[:, :, 0].T
    sy = states[:, :, 1].T
    sz = states[:, :, 2].T
    s_perp = np.sqrt(sx ** 2 + sy ** 2)
    s_rot_real, s_rot_imag, s_rot_abs = rotating_frame_fields(sx, sy, tau_grid, params)

    energy = np.zeros(len(tau_grid), dtype=float)
    ipr = np.zeros(len(tau_grid), dtype=float)
    local_b_center = np.zeros(len(tau_grid), dtype=float)
    local_j_center = np.zeros(len(tau_grid), dtype=float)
    local_d_center = np.zeros(len(tau_grid), dtype=float)

    center = N_SITES // 2
    for i, tau in enumerate(tau_grid):
        spins_i = states[i]
        energy[i] = compute_energy_snapshot(spins_i, tau, atomic, params)

        amp = s_perp[:, i]
        denom = float(np.sum(amp ** 2))
        ipr[i] = float(np.sum(amp ** 4) / (denom ** 2)) if denom > 1e-14 else 0.0

        j_local, d_local, b_local = get_local_coefficients(tau, atomic, params)
        local_b_center[i] = b_local[center]
        local_j_center[i] = j_local[center]
        local_d_center[i] = d_local[center]

    center_signal = s_rot_real[center]

    solution = MagneticSolution(
        tau=tau_grid,
        sx=sx,
        sy=sy,
        sz=sz,
        s_perp=s_perp,
        s_rot_real=s_rot_real,
        s_rot_imag=s_rot_imag,
        s_rot_abs=s_rot_abs,
        energy=energy,
        ipr=ipr,
        center_signal=center_signal,
        local_b_center=local_b_center,
        local_j_center=local_j_center,
        local_d_center=local_d_center,
    )
    return profile, solution


# =============================================================================
# --- ВИЗУАЛИЗАЦИИ ---
# =============================================================================


def select_time_indices(time_grid: np.ndarray, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, len(time_grid) - 1, count, dtype=int))



def save_profile_plot(profile: np.ndarray, folder: Path, params: RegimeParameters) -> None:
    n = np.arange(N_SITES)
    plt.figure(figsize=(10.2, 5.6))
    plt.plot(n, profile, "o-", markersize=3, linewidth=1.3)
    plt.title(f"Автономный профиль s_n в rotating frame | {params.name}")
    plt.xlabel("Номер узла n")
    plt.ylabel("s_n")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(folder / "01_autonomous_profile.png", dpi=PLOT_DPI)
    plt.close()



def save_reference_orientation_plot(profile: np.ndarray, folder: Path, params: RegimeParameters) -> None:
    """
    Главная картинка для проверки статьи.

    Здесь не нужно смотреть на отдельные Sx и Sy. Нужно смотреть на:
    - S_perp = sqrt(Sx^2 + Sy^2): поперечная компонента;
    - Sz: продольная компонента вдоль цепочки.
    """
    spins0 = autonomous_lab_spins(profile, tau_value=0.0, params=params)
    s_perp = np.sqrt(spins0[:, 0] ** 2 + spins0[:, 1] ** 2)
    sz = spins0[:, 2]
    n = np.arange(N_SITES)

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.6), constrained_layout=True)

    axes[0].plot(n, s_perp, linewidth=2.0, label=r"$S_\perp=\sqrt{S_x^2+S_y^2}$")
    axes[0].plot(n, sz, linewidth=2.0, label=r"$S_z$")
    axes[0].set_title(
        "Проверка геометрии режима: Z -- ось цепочки, Sx/Sy -- поперечные компоненты"
    )
    axes[0].set_xlabel("Номер узла n")
    axes[0].set_ylabel("Амплитуда")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(n, profile, linewidth=2.0, label="s_n в rotating frame")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title(
        f"Ожидание по статье | {params.article_hint}"
    )
    axes[1].set_xlabel("Номер узла n")
    axes[1].set_ylabel("s_n")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()

    plt.savefig(folder / "02_reference_orientation.png", dpi=PLOT_DPI)
    plt.close(fig)



def save_article_like_snapshots(profile: np.ndarray, folder: Path, params: RegimeParameters) -> None:
    """
    Snapshot'ы точной автономной орбиты статьи в lab frame.
    Это удобнее, чем смотреть произвольные Sx/Sy из driven расчета.
    """
    tau_samples = np.linspace(0.0, 2.0 * math.pi / max(abs(omega_lab(params)), 1e-12), 4, endpoint=False)
    n = np.arange(N_SITES)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    for ax, tau_value in zip(axes.ravel(), tau_samples):
        spins = autonomous_lab_spins(profile, tau_value=tau_value, params=params)
        s_perp = np.sqrt(spins[:, 0] ** 2 + spins[:, 1] ** 2)
        sz = spins[:, 2]
        ax.plot(n, s_perp, label=r"$S_\perp$", linewidth=1.9)
        ax.plot(n, sz, label=r"$S_z$", linewidth=1.9)
        ax.set_title(rf"Автономная орбита, $\tau={tau_value:.3f}$")
        ax.set_xlabel("n")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle(f"Article-like snapshot'ы: судим по S_perp и Sz | {params.name}", fontsize=14)
    plt.savefig(folder / "03_article_like_snapshots.png", dpi=PLOT_DPI)
    plt.close(fig)



def save_component_heatmaps(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> None:
    tau = solution.tau
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.2), constrained_layout=True)

    fields = [
        (solution.s_perp, r"$S_\perp(n,\tau)$", "inferno"),
        (solution.sz, r"$S_z(n,\tau)$", "viridis"),
        (solution.s_rot_real, r"Re$\,s_n$ в rotating frame", "coolwarm"),
        (solution.s_rot_imag, r"Im$\,s_n$ в rotating frame", "coolwarm"),
    ]

    for ax, (data, title, cmap) in zip(axes.flat, fields):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[tau[0], tau[-1], 0, N_SITES - 1],
        )
        ax.set_title(title)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("n")
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"Карты, по которым нужно проверять dark / bright режим | {params.name}", fontsize=14)
    plt.savefig(folder / "04_main_heatmaps.png", dpi=PLOT_DPI)
    plt.close(fig)



def save_raw_xy_heatmaps(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> None:
    tau = solution.tau
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.6), constrained_layout=True)

    for ax, data, title in [
        (axes[0], solution.sx, "Sx(n, tau)"),
        (axes[1], solution.sy, "Sy(n, tau)"),
    ]:
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            extent=[tau[0], tau[-1], 0, N_SITES - 1],
        )
        ax.set_title(title + "  (только вспомогательно)")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("n")
        fig.colorbar(im, ax=ax)

    plt.savefig(folder / "05_raw_sx_sy_heatmaps.png", dpi=PLOT_DPI)
    plt.close(fig)



def save_center_timeseries(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> None:
    center = N_SITES // 2
    tau = solution.tau

    plt.figure(figsize=(12.6, 7.0))
    plt.plot(tau, solution.s_perp[center], label=r"$S_\perp$(center)")
    plt.plot(tau, solution.sz[center], label=r"$S_z$(center)")
    plt.plot(tau, solution.s_rot_real[center], label=r"Re$\,s_{center}$")
    plt.plot(tau, solution.s_rot_imag[center], label=r"Im$\,s_{center}$")
    plt.title(f"Центральный узел: физически осмысленные величины | {params.name}")
    plt.xlabel(r"$\tau$")
    plt.ylabel("Амплитуда")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "06_center_timeseries.png", dpi=PLOT_DPI)
    plt.close()



def save_diagnostics(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> None:
    tau = solution.tau
    freqs, fft_vals, peak_freq = dominant_frequency(solution.center_signal, tau)

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.2), constrained_layout=True)

    axes[0, 0].plot(tau, solution.ipr)
    axes[0, 0].set_title("IPR(t) по S_perp")
    axes[0, 0].set_xlabel(r"$\tau$")
    axes[0, 0].grid(True, alpha=0.35)

    axes[0, 1].plot(tau, 1.0 / np.maximum(solution.ipr, 1e-12))
    axes[0, 1].set_title(r"$N_{eff}(t)=1/IPR$")
    axes[0, 1].set_xlabel(r"$\tau$")
    axes[0, 1].grid(True, alpha=0.35)

    axes[1, 0].plot(tau, solution.energy)
    axes[1, 0].set_title("Энергия магнитной подсистемы")
    axes[1, 0].set_xlabel(r"$\tau$")
    axes[1, 0].grid(True, alpha=0.35)

    axes[1, 1].plot(freqs, fft_vals)
    axes[1, 1].axvline(peak_freq, color="red", linestyle="--", label=f"peak={peak_freq:.3f}")
    axes[1, 1].set_title("Спектр Re s_center в rotating frame")
    axes[1, 1].set_xlabel(r"Частота в единицах $1/\tau$")
    axes[1, 1].set_xlim(0.0, min(5.0, freqs[-1] if len(freqs) else 5.0))
    axes[1, 1].grid(True, alpha=0.35)
    axes[1, 1].legend()

    fig.suptitle(f"Диагностики локализации и спектра | {params.name}", fontsize=14)
    plt.savefig(folder / "07_diagnostics.png", dpi=PLOT_DPI)
    plt.close(fig)



def save_spatial_snapshots(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> None:
    snap_ids = select_time_indices(solution.tau, 6)
    n = np.arange(N_SITES)

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.0), constrained_layout=True)
    axes = axes.ravel()

    for ax, idx in zip(axes, snap_ids):
        ax.plot(n, solution.s_perp[:, idx], label=r"$S_\perp$")
        ax.plot(n, solution.sz[:, idx], label=r"$S_z$")
        ax.plot(n, solution.s_rot_real[:, idx], label=r"Re$\,s_n$", linestyle="--")
        ax.set_title(rf"$\tau={solution.tau[idx]:.1f}$")
        ax.set_xlabel("n")
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    fig.suptitle(f"Пространственные профили в driven расчете | {params.name}", fontsize=14)
    plt.savefig(folder / "08_spatial_snapshots.png", dpi=PLOT_DPI)
    plt.close(fig)



def save_driver_maps(atomic: AtomicBreatherData, folder: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), constrained_layout=True)

    items = [
        (atomic.q_period, "Атомные смещения q_n(t_a)", "coolwarm"),
        (atomic.bond_eta_norm, "Bond strain", "coolwarm"),
        (atomic.site_eta_norm, "Site strain", "coolwarm"),
    ]

    t_axis = np.linspace(0.0, atomic.period, atomic.q_period.shape[1], endpoint=False)
    for ax, (data, title, cmap) in zip(axes, items):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[t_axis[0], t_axis[-1], 0, data.shape[0] - 1],
        )
        ax.set_title(title)
        ax.set_xlabel(r"$t_a$")
        ax.set_ylabel("n")
        fig.colorbar(im, ax=ax)

    plt.savefig(folder / "09_atomic_driver_maps.png", dpi=PLOT_DPI)
    plt.close(fig)



def save_parameter_tracks(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> None:
    tau = solution.tau
    plt.figure(figsize=(12.4, 6.2))
    plt.plot(tau, solution.local_j_center, label="j_center(t)")
    plt.plot(tau, solution.local_d_center, label="d_center(t)")
    plt.plot(tau, solution.local_b_center, label="b_center(t)")
    plt.title(f"Локальные магнитные коэффициенты в центре | {params.name}")
    plt.xlabel(r"$\tau$")
    plt.ylabel("Значение")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "10_local_coefficients_center.png", dpi=PLOT_DPI)
    plt.close()



def rescale_gif_frame_duration(path: Path, slowdown_factor: int) -> None:
    """
    Замедляет уже сохраненный GIF через увеличение duration у кадров.

    Это дешевле, чем дублировать кадры во время рендера.
    """
    if slowdown_factor <= 1:
        return

    image = Image.open(path)
    frames = []
    durations = []

    try:
        while True:
            frames.append(image.copy())
            durations.append(image.info.get("duration", 100))
            image.seek(image.tell() + 1)
    except EOFError:
        pass

    if not frames:
        return

    slowed_durations = [max(20, int(duration * slowdown_factor)) for duration in durations]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=slowed_durations,
        loop=image.info.get("loop", 0),
        disposal=2,
    )

def save_stacked_components_animation(
    solution: MagneticSolution,
    atomic: AtomicBreatherData,
    folder: Path,
    params: RegimeParameters,
) -> str:
    """
    Анимация из трех профилей:
    1) Sx и Sy на одном графике,
    2) Sz,
    3) |s^+| и атомный бризер q_n(t_a) на одном графике.

    Атомный бризер масштабируется ОДНИМ глобальным коэффициентом,
    чтобы его амплитуда была сопоставима с |s^+| по всему GIF.
    Это удобнее для визуальной оценки синфазности, чем покадровая нормировка.
    """
    frame_ids = np.unique(np.linspace(0, len(solution.tau) - 1, ANIMATION_MAX_FRAMES, dtype=int))
    n = np.arange(N_SITES)

    # Глобальный масштаб атомного бризера под уровень |s^+|.
    atomic_abs_max = max(float(np.max(np.abs(atomic.q_period))), 1e-12)
    splus_abs_max = max(float(np.max(solution.s_perp)), 1e-12)
    atomic_scale = 0.92 * splus_abs_max / atomic_abs_max

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 11.2), sharex=True)

    # --- Панель 1: Sx и Sy вместе ---
    line_sx, = axes[0].plot([], [], linewidth=1.8, label="Sx")
    line_sy, = axes[0].plot([], [], linewidth=1.8, label="Sy")
    axes[0].set_xlim(0, N_SITES - 1)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Sx, Sy", fontsize=ANIMATION_LABEL_FONTSIZE)
    axes[0].tick_params(axis="both", labelsize=ANIMATION_TICK_FONTSIZE)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=ANIMATION_TICK_FONTSIZE)

    # --- Панель 2: Sz ---
    line_sz, = axes[1].plot([], [], linewidth=1.8, label="Sz")
    axes[1].set_xlim(0, N_SITES - 1)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("Sz", fontsize=ANIMATION_LABEL_FONTSIZE)
    axes[1].tick_params(axis="both", labelsize=ANIMATION_TICK_FONTSIZE)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=ANIMATION_TICK_FONTSIZE)

    # --- Панель 3: |s^+| и атомный бризер ---
    line_splus, = axes[2].plot([], [], linewidth=2.0, label=r"|s$^+$| = $S_\perp$")
    line_atomic, = axes[2].plot([], [], "--", linewidth=1.8, label=r"scaled $q_n(t_a)$")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[2].set_xlim(0, N_SITES - 1)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_ylabel(r"|s$^+$| и $q_n$", fontsize=ANIMATION_LABEL_FONTSIZE)
    axes[2].set_xlabel("Номер узла n", fontsize=ANIMATION_LABEL_FONTSIZE)
    axes[2].tick_params(axis="both", labelsize=ANIMATION_TICK_FONTSIZE)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=ANIMATION_TICK_FONTSIZE)

    title = fig.suptitle("", fontsize=ANIMATION_TITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def update(frame_idx: int):
        idx = int(frame_ids[frame_idx])
        tau_mag = float(solution.tau[idx])
        tau_atomic = RHO_TIME_SCALING * tau_mag

        q_atomic = periodic_sample(atomic.q_period, atomic.period, tau_atomic)
        q_atomic_scaled = atomic_scale * q_atomic

        line_sx.set_data(n, solution.sx[:, idx])
        line_sy.set_data(n, solution.sy[:, idx])
        line_sz.set_data(n, solution.sz[:, idx])
        line_splus.set_data(n, solution.s_perp[:, idx])
        line_atomic.set_data(n, q_atomic_scaled)

        title.set_text(
            "Профили Sx/Sy, Sz и |s⁺| + атомный бризер"
            f" | {params.name} | tau={tau_mag:.2f}"
        )
        return line_sx, line_sy, line_sz, line_splus, line_atomic, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    path = folder / "13_profiles_stacked_components.gif"
    animation.save(path, writer=PillowWriter(fps=ANIMATION_FPS))
    plt.close(fig)
    return str(path)

def save_stacked_components_animation_old(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> str:
    """
    Возвращает анимацию из четырех профилей один под другим.

    Здесь оставляем именно те четыре величины, которые были нужны пользователю:
    Sx, Sy, Sz и |s^+|. Последняя величина по модулю совпадает с S_perp.
    """
    frame_ids = np.unique(np.linspace(0, len(solution.tau) - 1, ANIMATION_MAX_FRAMES, dtype=int))
    n = np.arange(N_SITES)

    fig, axes = plt.subplots(4, 1, figsize=(11.2, 12.8), sharex=True)
    components = [
        (solution.sx, "Sx", -1.05, 1.05),
        (solution.sy, "Sy", -1.05, 1.05),
        (solution.sz, "Sz", -1.05, 1.05),
        (solution.s_perp, r"|s$^+$| = $S_\perp$", -0.05, 1.05),
    ]

    lines = []
    for ax, (_data, label, y_min, y_max) in zip(axes, components):
        line, = ax.plot([], [], linewidth=1.8)
        lines.append(line)
        ax.set_xlim(0, N_SITES - 1)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(label, fontsize=ANIMATION_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=ANIMATION_TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Номер узла n", fontsize=ANIMATION_LABEL_FONTSIZE)
    title = fig.suptitle("", fontsize=ANIMATION_TITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def update(frame_idx: int):
        idx = int(frame_ids[frame_idx])
        for line, (data, _label, _y_min, _y_max) in zip(lines, components):
            line.set_data(n, data[:, idx])
        title.set_text(
            "Профили Sx, Sy, Sz и |s⁺| по узлам"
            f" | {params.name} | tau={solution.tau[idx]:.2f}"
        )
        return (*lines, title)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    path = folder / "13_profiles_stacked_components.gif"
    animation.save(path, writer=PillowWriter(fps=ANIMATION_FPS))
    plt.close(fig)
    return str(path)


def save_spin_3d_animation(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> str:
    """
    Крупная 3D-анимация ориентаций всех спинов.

    В исправленной версии геометрия согласована с тем, что обсуждали:
    ось Z является осью цепочки. Поэтому узлы располагаются вдоль оси Z,
    а компоненты Sx и Sy остаются поперечными.
    """
    frame_indices = np.unique(np.linspace(0, len(solution.tau) - 1, ANIMATION_MAX_FRAMES, dtype=int))
    node_positions = np.arange(N_SITES, dtype=float)

    fig = plt.figure(figsize=SPIN_3D_FIGSIZE)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.84], projection="3d")

    def draw_frame(frame_idx: int) -> None:
        ax.cla()

        sx = solution.sx[:, frame_idx]
        sy = solution.sy[:, frame_idx]
        sz = solution.sz[:, frame_idx]

        # Геометрическая ось цепочки теперь совпадает с осью Z.
        ax.plot(
            np.zeros_like(node_positions),
            np.zeros_like(node_positions),
            node_positions,
            color="black",
            linewidth=1.6,
            alpha=0.65,
        )

        # Цвет кодирует продольную компоненту Sz.
        colors = plt.cm.plasma((sz + 1.0) / 2.0)

        ax.quiver(
            np.zeros_like(node_positions),
            np.zeros_like(node_positions),
            node_positions,
            sx,
            sy,
            sz,
            colors=colors,
            length=0.9,
            normalize=True,
            arrow_length_ratio=0.22,
            linewidths=1.0,
        )

        center = N_SITES // 2
        ax.scatter(
            [0.0], [0.0], [float(center)],
            s=60,
            color="red",
            depthshade=False,
            label="центр цепочки",
        )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-2.0, N_SITES + 1.0)
        ax.set_xlabel("Sx", fontsize=SPIN_3D_LABEL_FONTSIZE, labelpad=12)
        ax.set_ylabel("Sy", fontsize=SPIN_3D_LABEL_FONTSIZE, labelpad=12)
        ax.set_zlabel("Ось цепочки Z (номер узла)", fontsize=SPIN_3D_LABEL_FONTSIZE, labelpad=18)
        ax.tick_params(labelsize=SPIN_3D_TICK_FONTSIZE, pad=3)
        ax.view_init(elev=18, azim=-62)
        ax.set_box_aspect((2.6, 2.6, N_SITES / 10.0))
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        ax.set_title(
            f"3D-ориентации спинов вдоль оси цепочки Z | {params.name}",
            fontsize=SPIN_3D_TITLE_FONTSIZE,
            pad=10,
        )
        fig.suptitle(
            f"Стрелки показывают направление (Sx, Sy, Sz), tau = {solution.tau[frame_idx]:.3f}",
            fontsize=SPIN_3D_SUPTITLE_FONTSIZE,
            y=0.955,
        )

    def update(frame_number: int):
        frame_idx = int(frame_indices[frame_number])
        draw_frame(frame_idx)
        return []

    draw_frame(int(frame_indices[0]))
    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    path = folder / "14_spin_3d_orientation.gif"
    animation.save(path, writer=PillowWriter(fps=ANIMATION_FPS))
    plt.close(fig)
    rescale_gif_frame_duration(path, SPIN_3D_SLOWDOWN_FACTOR)
    return str(path)

def save_requested_animations(
    solution: MagneticSolution,
    atomic: AtomicBreatherData,
    folder: Path,
    params: RegimeParameters,
) -> None:
    if not SAVE_GIFS:
        return

    save_stacked_components_animation(solution, atomic, folder, params)
    save_spin_3d_animation(solution, folder, params)

def save_requested_animations_old(solution: MagneticSolution, folder: Path, params: RegimeParameters) -> None:
    if not SAVE_GIFS:
        return

    save_stacked_components_animation(solution, folder, params)
    save_spin_3d_animation(solution, folder, params)



def save_summary_json(profile: np.ndarray, solution: MagneticSolution, folder: Path, params: RegimeParameters) -> None:
    center = N_SITES // 2
    freqs, fft_vals, peak_freq = dominant_frequency(solution.center_signal, solution.tau)

    edge_mean_perp = 0.5 * (float(solution.s_perp[0, 0]) + float(solution.s_perp[-1, 0]))
    edge_mean_sz = 0.5 * (float(solution.sz[0, 0]) + float(solution.sz[-1, 0]))

    summary = {
        "regime": asdict(params),
        "n_sites": N_SITES,
        "atomic_omega": ATOMIC_OMEGA,
        "rho_time_scaling": RHO_TIME_SCALING,
        "kappa_j": KAPPA_J,
        "kappa_d": KAPPA_D,
        "b_a": B_A,
        "use_normalized_drive": USE_NORMALIZED_DRIVE,
        "enable_j_drive": ENABLE_J_DRIVE,
        "enable_d_drive": ENABLE_D_DRIVE,
        "enable_a_drive": ENABLE_A_DRIVE,
        "tau_end": TAU_END,
        "tau_ramp": TAU_RAMP,
        "alpha_bulk": ALPHA_BULK,
        "alpha_edge_max": ALPHA_EDGE_MAX,
        "profile_max_abs": float(np.max(np.abs(profile))),
        "profile_center": float(profile[center]),
        "profile_edge_left": float(profile[0]),
        "profile_edge_right": float(profile[-1]),
        "s_perp_center_initial": float(solution.s_perp[center, 0]),
        "sz_center_initial": float(solution.sz[center, 0]),
        "s_perp_edge_mean_initial": edge_mean_perp,
        "sz_edge_mean_initial": edge_mean_sz,
        "ipr_mean_last_half": float(np.mean(solution.ipr[len(solution.ipr) // 2:])),
        "dominant_center_frequency": peak_freq,
        "energy_mean_last_half": float(np.mean(solution.energy[len(solution.energy) // 2:])),
        "energy_std_last_half": float(np.std(solution.energy[len(solution.energy) // 2:])),
    }

    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =============================================================================
# --- ЗАПУСК ОДНОГО РЕЖИМА ---
# =============================================================================


def run_one_regime(atomic: AtomicBreatherData, params: RegimeParameters) -> None:
    folder = RESULTS_ROOT / params.name
    folder.mkdir(parents=True, exist_ok=True)

    profile, solution = solve_magnetic_dynamics(atomic, params)

    save_profile_plot(profile, folder, params)
    save_reference_orientation_plot(profile, folder, params)
    save_article_like_snapshots(profile, folder, params)
    save_component_heatmaps(solution, folder, params)
    save_raw_xy_heatmaps(solution, folder, params)
    save_center_timeseries(solution, folder, params)
    save_diagnostics(solution, folder, params)
    save_spatial_snapshots(solution, folder, params)
    save_driver_maps(atomic, folder)
    save_parameter_tracks(solution, folder, params)
    save_requested_animations(solution, atomic, folder, params)
    #save_requested_animations(solution, folder, params)
    save_summary_json(profile, solution, folder, params)

    if SAVE_NUMPY_ARCHIVE:
        np.savez_compressed(
            folder / "solution_data.npz",
            profile=profile,
            tau=solution.tau,
            sx=solution.sx,
            sy=solution.sy,
            sz=solution.sz,
            s_perp=solution.s_perp,
            s_rot_real=solution.s_rot_real,
            s_rot_imag=solution.s_rot_imag,
            s_rot_abs=solution.s_rot_abs,
            energy=solution.energy,
            ipr=solution.ipr,
            center_signal=solution.center_signal,
            local_b_center=solution.local_b_center,
            local_j_center=solution.local_j_center,
            local_d_center=solution.local_d_center,
        )


# =============================================================================
# --- MAIN ---
# =============================================================================


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic = simulate_atomic_breather()

    for mode_name in RUN_MODES:
        if mode_name not in REGIMES:
            raise KeyError(f"Неизвестный режим: {mode_name}")
        run_one_regime(atomic, REGIMES[mode_name])

    readme_text = f"""
Результаты сохранены в подпапках:
{', '.join(str((RESULTS_ROOT / name).name) for name in RUN_MODES)}

Что смотреть в первую очередь:
1. 02_reference_orientation.png
   Главная проверка геометрии статьи.
   Если это bottom_dark, то на краях должен быть большой S_perp и малый S_z,
   а в центре наоборот.
   Если это bottom_bright, то на краях должен быть большой S_z и малый S_perp,
   а отклонение должно сидеть около центра.

2. 03_article_like_snapshots.png
   Snapshot'ы точной автономной орбиты статьи в lab frame.

3. 04_main_heatmaps.png
   Карты driven-динамики в тех величинах, по которым физически надо судить
   о dark / bright режиме: S_perp, S_z и s_n в rotating frame.

4. 05_raw_sx_sy_heatmaps.png
   Sx и Sy оставлены только как вспомогательные карты. По ним одним судить
   о геометрии бризера нельзя.

5. 13_profiles_stacked_components.gif
   Возвращенная анимация четырех профилей: Sx, Sy, Sz и |s^+| по узлам.

6. 14_spin_3d_orientation.gif
   Крупная 3D-анимация, где ось цепочки совпадает с осью Z.

Ключевые параметры текущего расчета:
- ATOMIC_OMEGA = {ATOMIC_OMEGA}
- RHO_TIME_SCALING = {RHO_TIME_SCALING}
- KAPPA_J = {KAPPA_J}
- KAPPA_D = {KAPPA_D}
- B_A = {B_A}
- USE_NORMALIZED_DRIVE = {USE_NORMALIZED_DRIVE}
- TAU_RAMP = {TAU_RAMP}
- TAU_END = {TAU_END}
""".strip()

    with open(RESULTS_ROOT / "README_results.txt", "w", encoding="utf-8") as f:
        f.write(readme_text)


if __name__ == "__main__":
    main()
