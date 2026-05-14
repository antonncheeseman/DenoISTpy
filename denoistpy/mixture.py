from __future__ import annotations

import numpy as np


def _init_pis(n_inits: int | np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if np.isscalar(n_inits):
        return rng.uniform(np.finfo(np.float32).eps, 0.5, int(n_inits)).astype(np.float32)
    pis = np.asarray(n_inits, dtype=np.float32)
    if np.any(pis <= 0) or np.any(pis > 1):
        raise ValueError("All initial values for pi must be in the range (0, 1].")
    return pis


def _poisson_logpmf_numpy(x: np.ndarray, rate: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    rate = np.maximum(rate, np.finfo(np.float32).tiny)
    return x * np.log(rate) - rate - gammaln(x + 1)


def solve_poisson_mixture_numpy(
    x: np.ndarray,
    s: np.ndarray,
    *,
    max_iter: int = 5000,
    tol: float = 1e-6,
    n_inits: int | np.ndarray = 10,
    posterior_cutoff: float = 0.6,
    random_state: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Faithful NumPy implementation of the R per-cell Poisson mixture."""

    x = np.asarray(x, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    n = x.size
    if np.sum(x) == 0:
        return {
            "memberships": np.ones(n, dtype=np.int8),
            "posterior": np.ones(n, dtype=np.float32),
            "lambda1": np.nan,
            "lambda2": np.nan,
            "pi": np.nan,
            "log_lik": np.nan,
            "n_iter": 0,
            "status": "zero_count",
        }

    keep = s > 0
    if not np.any(keep):
        return {
            "memberships": np.ones(n, dtype=np.int8),
            "posterior": np.ones(n, dtype=np.float32),
            "lambda1": np.nan,
            "lambda2": np.nan,
            "pi": np.nan,
            "log_lik": np.nan,
            "n_iter": 0,
            "status": "zero_offset",
        }

    xf = x[keep]
    sf = s[keep]
    rng = np.random.default_rng(random_state)
    best: dict[str, np.ndarray | float] | None = None
    best_log_lik = -np.inf

    for pi_init in _init_pis(n_inits, rng):
        lambda1 = float(np.mean(xf) / np.mean(sf))
        lambda2 = float(np.mean(xf) / (2 * np.mean(sf)))
        pi = float(pi_init)
        log_lik = float(
            np.sum(
                np.logaddexp(
                    np.log(pi) + _poisson_logpmf_numpy(xf, sf * lambda1),
                    np.log1p(-pi) + _poisson_logpmf_numpy(xf, sf * lambda2),
                )
            )
        )
        n_iter = 0
        gamma = np.ones_like(xf)

        for n_iter in range(1, max_iter + 1):
            log_tau1 = np.log(pi) + _poisson_logpmf_numpy(xf, sf * lambda1)
            log_tau2 = np.log1p(-pi) + _poisson_logpmf_numpy(xf, sf * lambda2)
            gamma = 1 / (1 + np.exp(log_tau2 - log_tau1))

            lambda1 = float(np.sum(gamma * xf) / np.sum(gamma * sf))
            lambda2 = float(np.sum((1 - gamma) * xf) / np.sum((1 - gamma) * sf))
            pi = float(np.mean(gamma))
            pi = min(max(pi, np.finfo(np.float32).eps), 1 - np.finfo(np.float32).eps)

            new_log_lik = float(
                np.sum(
                    np.logaddexp(
                        np.log(pi) + _poisson_logpmf_numpy(xf, sf * lambda1),
                        np.log1p(-pi) + _poisson_logpmf_numpy(xf, sf * lambda2),
                    )
                )
            )
            if not np.isfinite(new_log_lik) or abs(new_log_lik - log_lik) < tol:
                break
            log_lik = new_log_lik

        if log_lik > best_log_lik:
            best_log_lik = log_lik
            if abs(lambda1 - lambda2) <= 1e-2:
                gamma = np.ones_like(xf)
            best = {
                "gamma": gamma.astype(np.float32),
                "lambda1": lambda1,
                "lambda2": lambda2,
                "pi": pi,
                "log_lik": log_lik,
                "n_iter": n_iter,
            }

    assert best is not None
    memberships = (best["gamma"] >= posterior_cutoff).astype(np.int8)
    if np.all(memberships == 0):
        memberships[:] = 1

    full_memberships = np.ones(n, dtype=np.int8)
    full_posterior = np.ones(n, dtype=np.float32)
    full_memberships[keep] = memberships
    full_posterior[keep] = best["gamma"]
    return {
        "memberships": full_memberships,
        "posterior": full_posterior,
        "lambda1": best["lambda1"],
        "lambda2": best["lambda2"],
        "pi": best["pi"],
        "log_lik": best["log_lik"],
        "n_iter": best["n_iter"],
        "status": "fit",
    }


def solve_poisson_mixture_torch(
    x: np.ndarray,
    s: np.ndarray,
    *,
    max_iter: int = 5000,
    tol: float = 1e-6,
    n_inits: int | np.ndarray = 10,
    posterior_cutoff: float = 0.6,
    device: str = "auto",
    random_state: int | None = None,
) -> dict[str, np.ndarray]:
    """Batched PyTorch EM for dense genes x batch_cells chunks."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install denoistpy[gpu] to use the PyTorch backend.") from exc
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    rng = np.random.default_rng(random_state)
    pi_inits = _init_pis(n_inits, rng)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    st = torch.as_tensor(s, dtype=torch.float32, device=device)
    zero_count = xt.sum(dim=0) == 0
    keep = st > 0
    xt = torch.where(keep, xt, torch.zeros_like(xt))
    st = torch.where(keep, st, torch.zeros_like(st))

    eps = torch.finfo(torch.float32).eps
    tiny = torch.finfo(torch.float32).tiny
    valid_counts = keep.sum(dim=0).clamp_min(1)
    mean_x = xt.sum(dim=0) / valid_counts
    mean_s = st.sum(dim=0).clamp_min(tiny) / valid_counts

    best_log_lik = torch.full((xt.shape[1],), -torch.inf, device=device)
    best_gamma = torch.ones_like(xt)
    best_lambda1 = torch.full((xt.shape[1],), torch.nan, device=device)
    best_lambda2 = torch.full((xt.shape[1],), torch.nan, device=device)
    best_pi = torch.full((xt.shape[1],), torch.nan, device=device)
    best_iter = torch.zeros((xt.shape[1],), dtype=torch.int32, device=device)

    def logpmf(obs: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        rate = rate.clamp_min(tiny)
        return obs * torch.log(rate) - rate - torch.lgamma(obs + 1)

    for pi_init in pi_inits:
        lambda1 = mean_x / mean_s.clamp_min(tiny)
        lambda2 = mean_x / (2 * mean_s.clamp_min(tiny))
        pi = torch.full_like(lambda1, float(pi_init)).clamp(eps, 1 - eps)
        log_lik = torch.full_like(lambda1, -torch.inf)
        gamma = torch.ones_like(xt)
        current_iter = 0

        for current_iter in range(1, max_iter + 1):
            rate1 = st * lambda1.unsqueeze(0)
            rate2 = st * lambda2.unsqueeze(0)
            log_tau1 = torch.log(pi).unsqueeze(0) + logpmf(xt, rate1)
            log_tau2 = torch.log1p(-pi).unsqueeze(0) + logpmf(xt, rate2)
            gamma = torch.sigmoid(log_tau1 - log_tau2)
            gamma = torch.where(keep, gamma, torch.ones_like(gamma))

            lambda1 = (gamma * xt).sum(dim=0) / (gamma * st).sum(dim=0).clamp_min(tiny)
            lambda2 = ((1 - gamma) * xt).sum(dim=0) / ((1 - gamma) * st).sum(dim=0).clamp_min(tiny)
            pi = (gamma * keep).sum(dim=0) / valid_counts
            pi = pi.clamp(eps, 1 - eps)

            rate1 = st * lambda1.unsqueeze(0)
            rate2 = st * lambda2.unsqueeze(0)
            log_tau1 = torch.log(pi).unsqueeze(0) + logpmf(xt, rate1)
            log_tau2 = torch.log1p(-pi).unsqueeze(0) + logpmf(xt, rate2)
            new_log_lik = torch.logaddexp(log_tau1, log_tau2)
            new_log_lik = torch.where(keep, new_log_lik, torch.zeros_like(new_log_lik)).sum(dim=0)
            converged = torch.isfinite(new_log_lik) & (torch.abs(new_log_lik - log_lik) < tol)
            log_lik = new_log_lik
            if bool(torch.all(converged).item()):
                break

        collapsed = torch.abs(lambda1 - lambda2) <= 1e-2
        gamma = torch.where(collapsed.unsqueeze(0), torch.ones_like(gamma), gamma)
        improved = log_lik > best_log_lik
        best_log_lik = torch.where(improved, log_lik, best_log_lik)
        best_gamma = torch.where(improved.unsqueeze(0), gamma, best_gamma)
        best_lambda1 = torch.where(improved, lambda1, best_lambda1)
        best_lambda2 = torch.where(improved, lambda2, best_lambda2)
        best_pi = torch.where(improved, pi, best_pi)
        best_iter = torch.where(
            improved,
            torch.full_like(best_iter, current_iter),
            best_iter,
        )

    memberships = (best_gamma >= posterior_cutoff).to(torch.int8)
    all_zero = memberships.sum(dim=0) == 0
    memberships[:, all_zero] = 1
    memberships = torch.where(keep, memberships, torch.ones_like(memberships))
    posterior = torch.where(keep, best_gamma, torch.ones_like(best_gamma))
    memberships[:, zero_count] = 1
    posterior[:, zero_count] = 1
    best_lambda1[zero_count] = torch.nan
    best_lambda2[zero_count] = torch.nan
    best_pi[zero_count] = torch.nan
    best_log_lik[zero_count] = torch.nan
    best_iter[zero_count] = 0

    status = np.full((xt.shape[1],), "fit", dtype=object)
    status[zero_count.cpu().numpy()] = "zero_count"

    return {
        "memberships": memberships.cpu().numpy(),
        "posterior": posterior.cpu().numpy().astype(np.float32),
        "lambda1": best_lambda1.cpu().numpy(),
        "lambda2": best_lambda2.cpu().numpy(),
        "pi": best_pi.cpu().numpy(),
        "log_lik": best_log_lik.cpu().numpy(),
        "n_iter": best_iter.cpu().numpy(),
        "status": status,
    }
