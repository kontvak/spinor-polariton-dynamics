"""M0–M3 polariton fields and reservoirs; units: um, ps, meV."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(profile="F1", **overrides):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    config = copy.deepcopy(raw["base"])
    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                merge(target.setdefault(key, {}), value)
            else:
                target[key] = value
    merge(config, raw["profiles"][profile])
    merge(config, overrides)
    parameters(config)
    return config


def parameters(config):
    p = dict(config["physical"])
    p.update(config["reservoirs"][config["reservoir"]])
    p.setdefault("beta", 0.0)
    p["model"] = config["model"]
    p["spins"] = 2 if p["model"] in ("M2", "M3") else 1
    if p["model"] not in ("M0", "M1", "M2", "M3"):
        raise ValueError("Unknown model")
    if p["model"] == "M2" and p["J"] != 0:
        raise ValueError("M2 requires J=0")
    p["K"] = 3.80998212e-5 / p["mass_over_me"]
    if not np.isfinite(p["beta"]) or abs(p["beta"]) >= p["K"]:
        raise ValueError("TE-TM requires finite |beta| < K")
    if p["spins"] != 2 and p["beta"] != 0:
        raise ValueError("TE-TM requires a spinor field")
    p["Pth"] = p["gammaC"] * p["gammaA"] / p["R"]
    p["nAth"] = p["gammaC"] / p["R"]
    p["tauI"] = 1 / (p["gammaI"] + p["W"])
    p["injection_ratio"] = (p["gammaI"] + p["W"]) / p["W"]
    positive = ("hbar", "mass_over_me", "W", "R")
    if any(p[k] <= 0 for k in positive) or min(p["GammaA"], p["GammaI"], p["J"], p["gammaC"], p["gammaA"], p["gammaI"]) < 0:
        raise ValueError("Invalid rates or coupling")
    if not np.isclose(p["gI"] / p["W"], p["gR"] * p["eta"], rtol=1e-12):
        raise ValueError("gI/W must equal gR*eta")
    if abs(config["pump"]["epsilon"]) > 1:
        raise ValueError("Pump polarization must lie in [-1,1]")
    return p


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    k2: np.ndarray
    dx: float
    da: float


def make_grid(config):
    n, length = int(config["space"]["N"]), float(config["space"]["L"])
    if n < 8 or n % 2 or length <= 0:
        raise ValueError("Use a positive domain and an even grid with N>=8")
    dx = length / n
    x = -length / 2 + np.arange(n) * dx
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    return Grid(x, x.copy(), k, k.copy(), k[:, None]**2 + k[None, :]**2, dx, dx**2)


def pump_profile(config, grid):
    kind = config["pump"]["kind"]
    if kind == "uniform":
        return np.ones(grid.k2.shape)
    radius = np.hypot(grid.x[None, :], grid.y[:, None])
    offset = config["pump"]["r0"] if kind == "ring" else 0.0
    if kind not in ("ring", "gaussian"):
        raise ValueError("Unknown pump profile")
    return np.exp(-((radius - offset) / config["pump"]["d"])**2)


def injection(config, grid=None):
    p = parameters(config)
    scale = config["pump"]["s"] if grid is None or config["pump"]["kind"] == "uniform" else config["pump"]["P0"]
    values = np.array([scale * p["Pth"]])
    if p["spins"] == 2:
        values = scale * p["Pth"] * (1 + np.array([1., -1.]) * config["pump"]["epsilon"])
    return values if grid is None else values[:, None, None] * pump_profile(config, grid)


def seed_waves(config):
    rng = np.random.default_rng(config["initial"]["seed"])
    return np.vstack((rng.normal(0, 1., 24), rng.normal(0, 1., 24),
                      rng.uniform(0, 2*np.pi, 24)))


def initial_state(config, grid=None, waves=None):
    p = parameters(config)
    theta = config["initial"]["theta"]
    if grid is None:
        field = np.array([np.sqrt(config["initial"]["density_0d"])], complex)
    else:
        kind = config["initial"].get("kind", "plane_waves")
        if kind in ("radial", "radial_perturbed"):
            field = np.ones(grid.k2.shape, complex)
            if kind == "radial_perturbed":
                # A continuous real perturbation, identical on overlapping nodes
                # of different domains; no imposed phase winding or current.
                epsilon=float(config["initial"].get("perturbation",1e-3))
                if not 0 <= epsilon < .1:raise ValueError("Seed perturbation must be smaller than 0.1")
                rng=np.random.default_rng(config["initial"].get("perturb_seed",20261025))
                xx,yy=grid.x[None,:],grid.y[:,None]
                z=(xx+1j*yy)/np.sqrt(xx*xx+yy*yy+2.5**2)
                noise=sum(np.real(z**m*np.exp(1j*phase)) for m,phase in enumerate(rng.uniform(0,2*np.pi,8),1))/2
                field *= 1+epsilon*noise
        elif kind == "plane_waves":
            waves = seed_waves(config) if waves is None else waves
            field = np.zeros(grid.k2.shape, complex)
            for kx, ky, phase in waves.T:
                field += np.exp(1j*(kx*grid.x[None, :] + ky*grid.y[:, None] + phase)) / np.sqrt(24)
        else:
            raise ValueError("Unknown seed kind")
        field *= np.sqrt(config["initial"]["density_2d"] * pump_profile(config, grid))
        field = field[None, ...]
    if p["spins"] == 2:
        phases = np.exp(1j*np.array([theta/2, -theta/2]))
        field = phases.reshape((2,) + (1,)*(field.ndim-1)) * field / np.sqrt(2)
    return field, np.zeros(field.shape), np.zeros(field.shape)


def local_rhs_scalar_one_reservoir(psi, active, inactive, pump, p):
    density = abs(psi)**2
    energy = p["gC"]*density + p["gR"]*(active + p["eta"]*pump)
    return (((p["R"]*active-p["gammaC"])/2 - 1j*energy/p["hbar"])*psi,
            pump-(p["gammaA"]+p["R"]*density)*active, np.zeros_like(inactive))


def local_rhs_scalar_two_reservoir(psi, active, inactive, pump, p):
    density = abs(psi)**2
    energy = p["gC"]*density + p["gA"]*active + p["gI"]*inactive
    return (((p["R"]*active-p["gammaC"])/2 - 1j*energy/p["hbar"])*psi,
            p["W"]*inactive-(p["gammaA"]+p["R"]*density)*active,
            p["injection_ratio"]*pump-(p["gammaI"]+p["W"])*inactive)


def local_rhs_spinor(psi, active, inactive, pump, p):
    density = abs(psi)**2
    energy = p["alpha1"]*density + p["alpha2"]*density[::-1] + p["gA"]*active + p["gI"]*inactive
    field = (((p["R"]*active-p["gammaC"])/2 - 1j*energy/p["hbar"])*psi
             + 1j*p["J"]/p["hbar"]*psi[::-1])
    da = (p["W"]*inactive-(p["gammaA"]+p["R"]*density)*active
          + p["GammaA"]*(active[::-1]-active))
    di = (p["injection_ratio"]*pump-(p["gammaI"]+p["W"])*inactive
          + p["GammaI"]*(inactive[::-1]-inactive))
    return field, da, di


def local_rhs(state, pump, p):
    function = {"M0": local_rhs_scalar_one_reservoir, "M1": local_rhs_scalar_two_reservoir,
                "M2": local_rhs_spinor, "M3": local_rhs_spinor}[p["model"]]
    return function(*state, pump, p)


def scalar_stationary(pump, p):
    nc = max(0., (pump-p["Pth"])/p["gammaC"])
    active = p["nAth"] if pump > p["Pth"] else pump/p["gammaA"]
    return nc, active, pump/p["W"]


def spin_stationary(s, epsilon, p):
    di = (p["gammaI"]+p["W"])/(p["gammaI"]+p["W"]+2*p["GammaI"])
    if s*(1-abs(epsilon)*di) <= 1:
        raise ValueError("Both-component stationary formula is not valid")
    inactive = s*p["Pth"]/p["W"]*(1+np.array([1., -1.])*epsilon*di)
    active = np.full(2, p["nAth"])
    density = p["gammaA"]/p["R"]*(s*(1+np.array([1., -1.])*epsilon*di)-1)
    energy = p["alpha1"]*density+p["alpha2"]*density[::-1]+p["gA"]*active+p["gI"]*inactive
    return density, active, inactive, energy


def stokes(psi, axis=0):
    """Unnormalized S0..S3; S2=2 Im(conj(psi_plus)*psi_minus).

    observables_0d()['legacy_s1'/'legacy_s2'] retain the normalized older
    convention for reproducing earlier reports. S0..S3 always use this one.
    """
    plus, minus = np.moveaxis(np.asarray(psi), axis, 0)
    nplus, nminus = abs(plus)**2, abs(minus)**2
    coherence = np.conj(plus)*minus
    return np.stack((nplus+nminus, 2*coherence.real,
                     2*coherence.imag, nplus-nminus))


def normalized_stokes(psi, relative_mask=1e-3, axis=0):
    S = stokes(psi, axis)
    valid = S[0] >= relative_mask*S[0].max()
    valid &= S[0] > 0
    return np.divide(S[1:], S[0], out=np.full_like(S[1:], np.nan),
                     where=valid), valid


def report3_config(case="B0", geometry="spot", pump_scale=4., **overrides):
    """Independent controlled starts for the new TE-TM series only."""
    rows = {"B0": (0., 0., 0.), "A1": (.3, 0., 0.), "A0": (.3, 0., 0.),
            "T0": (0., 0., .0119), "T1": (.3, 0., .0119),
            "J1": (.3, .010, 0.), "C1": (.3, .010, .0119)}
    if case not in rows or geometry not in ("spot", "ring"):
        raise ValueError("Unknown report3 case or geometry")
    epsilon, J, beta = rows[case]
    config = load_config("report3", physical={"J": J, "beta": beta},
                         pump={"kind": "gaussian" if geometry == "spot" else "ring",
                               "P0": float(pump_scale), "epsilon": epsilon})
    if case == "A0": config["physical"]["alpha2"] = 0.0
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict): merge(a.setdefault(k, {}), v)
            else: a[k] = v
    merge(config, overrides)
    parameters(config)
    return config
