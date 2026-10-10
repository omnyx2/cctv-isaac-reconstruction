"""Vanishing point of the static edges by a Gaussian process over the ray signal.

Signal of a point P: rays from P downwards (0..180 deg: road lines lie below the
vanishing point); along each ray the edge-sum image of tools/edge_accumulate.py is
summed as line evidence: at each point of the ray, the image value minus the mean of
the values SIDE px to its left and right (perpendicular to the ray), not below 0 -
large only where the ray runs along a thin line, 0 in blobs and wide areas (grass,
queues). R(P, theta) is then the length of line the ray follows; the signal is
sum_theta R(P, theta)^2: a ray that runs along a line for long counts much, the short
evidence of rays merely crossing lines (everywhere about the same) little. The vanishing
point is where the signal is strongest. (Earlier tries: the plain sum of the image
along the rays is the same for every P; sum R^2 rewards one heavy line; prominence
against neighbouring angles counts any blob that looks narrow from far away, so it
grows with the distance of P.)

The signal is not evaluated everywhere: a Gaussian process (Matern 5/2) is fitted to
the points evaluated so far and the next point is the one with the highest expected
improvement (Bayesian optimization), in two stages:
  1. coarse: quarter resolution, blurred, 1 deg rays, the whole domain (the image and
     above it - the point can lie outside the image),
  2. fine: half resolution, 0.25 deg rays, around each of the CANDIDATES highest
     separate hills of stage 1's GP mean (a narrow peak may be between its
     evaluations; the mean still rises there); the strongest result wins.
For checking, stage 1's signal is also evaluated on a dense grid (heat map).

Writes into <dir> (an edge_accumulate blur folder):
  vp_stage1.png   dense signal (heat map), the GP's evaluations in order, the best
  vp_stage2.png   the GP mean around it, its evaluations, the vanishing point
  vp_profile.png  R(theta) at the vanishing point
  vp_rays.png     the strongest rays through it on the static edges
  vp_overview.jpg
  vp.json         the vanishing point (full-resolution pixels)

Usage:
    C:\\isaacsim\\python.bat tools/vp_gaussian_process.py --dir data/results/edge_accum/<video>/blur9
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.edge_accumulate import overview  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN = ((-0.5, 1.5), (-0.6, 0.8))     # x, y of the search domain, in image widths / heights
STAGES = (  # scale of the image, blur sigma and side offset (px at that scale), ray step deg, start points, iterations
    {"scale": 0.25, "sigma": 4.0, "side": 10.0, "step_deg": 1.0, "start": 40, "iterations": 60},
    {"scale": 0.5, "sigma": 1.0, "side": 6.0, "step_deg": 0.25, "start": 10, "iterations": 30},
)
FINE_HALF_WIDTH = 0.08                   # stage 2 domain: best +- this x image width
RAY_RANGE_DEG = (10.0, 170.0)            # downwards (y grows down), not along the horizon: bridges, horizon edges
CANDIDATES = 3                           # stage 2 runs around this many separate hills of stage 1's GP mean
CANDIDATE_SEPARATION = 0.15              # ... at least this x image width apart


# ---------------------------------------------------------------- the signal
class RaySignal:
    def __init__(self, image: np.ndarray, scale: float, sigma: float, side: float, step_deg: float):
        h, w = image.shape
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if sigma > 0:
            small = cv2.GaussianBlur(small, (0, 0), sigma)
        self.scale, self.h, self.w = scale, small.shape[0], small.shape[1]
        self.img = torch.from_numpy(small.astype(np.float32)).to(DEVICE)[None, None]
        self.theta = torch.arange(RAY_RANGE_DEG[0], RAY_RANGE_DEG[1], step_deg, device=DEVICE) * math.pi / 180
        self.side = side
        reach = math.hypot(self.w, self.h) * 2.2                  # from a point outside, across the image
        self.r = torch.arange(1.0, reach, 1.0, device=DEVICE)

    def rays(self, pts: torch.Tensor) -> torch.Tensor:
        """R(P, theta): line evidence along each ray, points (N, 2) in full-resolution pixels -> (N, angles)."""
        p = pts * self.scale
        out = []
        for chunk in torch.split(p, max(1, 2_000_000 // (len(self.theta) * len(self.r) // 8 + 1)) or 1):
            dx, dy = torch.cos(self.theta), torch.sin(self.theta)
            x = chunk[:, 0, None, None] + self.r[None, None, :] * dx[None, :, None]
            y = chunk[:, 1, None, None] + self.r[None, None, :] * dy[None, :, None]
            img = self.img.expand(len(chunk), -1, -1, -1)

            def sample(ox, oy):
                grid = torch.stack([(x + ox) / (self.w - 1) * 2 - 1, (y + oy) / (self.h - 1) * 2 - 1], -1)
                return F.grid_sample(img, grid, align_corners=True, padding_mode="zeros")[:, 0]

            def inside(ox, oy):
                return (x + ox >= 0) & (x + ox <= self.w - 1) & (y + oy >= 0) & (y + oy <= self.h - 1)

            nx, ny = (-dy * self.side)[None, :, None], (dx * self.side)[None, :, None]
            line = sample(0, 0) - 0.5 * (sample(nx, ny) + sample(-nx, -ny))
            valid = inside(nx, ny) & inside(-nx, -ny)            # both sides seen: no evidence at the border
            out.append((line.clamp_min(0) * valid).sum(-1))
        return torch.cat(out) / self.scale                       # in full-resolution pixel units

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        return (self.rays(pts) ** 2).sum(-1)


# ---------------------------------------------------------------- Gaussian process
def matern52(a: torch.Tensor, b: torch.Tensor, length: float) -> torch.Tensor:
    d = torch.cdist(a, b) / length
    s = math.sqrt(5) * d
    return (1 + s + s * s / 3) * torch.exp(-s)


class GP:
    """Zero-mean GP on standardized targets; length scale and noise by marginal likelihood
    (all candidate pairs at once: one batched Cholesky on the GPU)."""

    LENGTHS = np.logspace(-2.3, -0.3, 14)
    NOISES = (1e-4, 1e-3, 1e-2)

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        self.x, self.mu, self.sd = x, y.mean(), y.std().clamp_min(1e-9)
        z = (y - self.mu) / self.sd
        length = torch.tensor(np.repeat(self.LENGTHS, len(self.NOISES)), device=x.device, dtype=x.dtype)
        noise = torch.tensor(np.tile(self.NOISES, len(self.LENGTHS)), device=x.device, dtype=x.dtype)
        s5 = math.sqrt(5) * torch.cdist(x, x)[None] / length[:, None, None]
        k = (1 + s5 + s5 * s5 / 3) * torch.exp(-s5) + noise[:, None, None] * torch.eye(len(x), device=x.device, dtype=x.dtype)
        chol, info = torch.linalg.cholesky_ex(k)
        zb = z[None, :, None].expand(len(length), -1, -1)
        alpha = torch.cholesky_solve(zb, chol)
        nll = 0.5 * (zb * alpha).sum((1, 2)) + torch.log(torch.diagonal(chol, dim1=1, dim2=2).abs()).sum(1)
        nll = torch.where(info == 0, nll, torch.full_like(nll, float("inf")))
        best = int(torch.argmin(nll))
        self.length, self.noise = float(length[best]), float(noise[best])
        self.chol, self.alpha = chol[best], alpha[best]
        return self

    def predict(self, q: torch.Tensor):
        k = matern52(q, self.x, self.length)
        mean = (k @ self.alpha)[:, 0]
        v = torch.cholesky_solve(k.T, self.chol)
        var = (1 - (k * v.T).sum(1)).clamp_min(1e-12)
        return mean * self.sd + self.mu, var.sqrt() * self.sd


def expected_improvement(mean, std, best):
    z = (mean - best) / std
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return (mean - best) * cdf + std * pdf


def optimize(signal: RaySignal, lo: np.ndarray, hi: np.ndarray, start: int, iterations: int, rng):
    """Bayesian optimization of log(signal) over the box lo..hi (full-res pixels)."""
    lo_t = torch.tensor(lo, device=DEVICE, dtype=torch.float64)
    span = torch.tensor(hi - lo, device=DEVICE, dtype=torch.float64)

    def evaluate(u):                                             # u in [0, 1]^2
        return torch.log(signal((lo_t + u * span).float()).double() + 1e-9)

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(int(rng.integers(2 ** 31)))
    nx = max(1, int(round(math.sqrt(start * 1.6))))              # start points: a regular grid covering the box
    ny = max(1, int(math.ceil(start / nx)))                       # (random ones can miss a narrow hill)
    gx, gy = torch.meshgrid((torch.arange(nx, device=DEVICE, dtype=torch.float64) + 0.5) / nx,
                            (torch.arange(ny, device=DEVICE, dtype=torch.float64) + 0.5) / ny, indexing="xy")
    u = torch.stack([gx.ravel(), gy.ravel()], 1)
    y = evaluate(u)
    gp = GP()
    for _ in range(iterations):
        gp.fit(u, y)
        cand = torch.cat([torch.rand((20000, 2), generator=gen, device=DEVICE, dtype=torch.float64),
                          (u[torch.argmax(y)] + 0.02 * torch.randn((2000, 2), generator=gen, device=DEVICE,
                                                                   dtype=torch.float64)).clamp(0, 1)])
        mean, std = gp.predict(cand)
        nxt = cand[torch.argmax(expected_improvement(mean, std, y.max()))][None]
        u, y = torch.cat([u, nxt]), torch.cat([y, evaluate(nxt)])
    gp.fit(u, y)
    pts = (lo_t + u * span).cpu().numpy()
    return pts, y.cpu().numpy(), gp, (lo_t, span)


def find_vanishing_point(image: np.ndarray, rng, domain=DOMAIN):
    """Stages 1 and 2 without pictures: the vanishing point (full-resolution px) and
    its log signal at stage 2."""
    return find_vanishing_points(image, rng, domain)[0]


def find_vanishing_points(image: np.ndarray, rng, domain=DOMAIN):
    """Every hill of stage 1, refined at stage 2: [(point, log signal)], strongest first.
    In some scenes hills are near ties (TUM s06: 14.605 / 14.602 / 14.596) - the caller
    decides among them with other evidence."""
    h, w = image.shape
    s1 = RaySignal(image, **{k: STAGES[0][k] for k in ("scale", "sigma", "side", "step_deg")})
    lo = np.array([domain[0][0] * w, domain[1][0] * h])
    hi = np.array([domain[0][1] * w, domain[1][1] * h])
    pts1, y1, gp1, _ = optimize(s1, lo, hi, STAGES[0]["start"], STAGES[0]["iterations"], rng)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 150))
    q = torch.tensor(np.c_[gx.ravel(), gy.ravel()], device=DEVICE)
    mean1 = gp1.predict((q - torch.tensor(lo, device=DEVICE)) / torch.tensor(hi - lo, device=DEVICE))[0].cpu().numpy()
    hills = [pts1[np.argmax(y1)]]
    for k in np.argsort(-mean1):
        c = np.array([gx.ravel()[k], gy.ravel()[k]])
        if all(np.hypot(*(c - o)) >= CANDIDATE_SEPARATION * w for o in hills):
            hills.append(c)
        if len(hills) == CANDIDATES:
            break
    s2 = RaySignal(image, **{k: STAGES[1][k] for k in ("scale", "sigma", "side", "step_deg")})
    half = FINE_HALF_WIDTH * w
    found = []
    for c in hills:
        pts2, y2, _, _ = optimize(s2, c - half, c + half, STAGES[1]["start"], STAGES[1]["iterations"], rng)
        found.append(refine(s2, pts2[np.argmax(y2)]))     # each hill refined: compared at their true peaks
    return sorted(found, key=lambda f: -f[1])


REFINE = ((32.0, 4.0), (4.0, 1.0))      # deterministic grids after the GP: (half width, step) px


def refine(signal: RaySignal, p0: np.ndarray):
    """The GP decides which hill; the point on it is found deterministically: a grid of
    the signal around it, then a finer one around its best, then a quadratic fit through
    the best and its neighbours (sub-pixel). Removes the scatter of the GP's random
    candidates from the result."""
    p = np.asarray(p0, float)
    for half, step in REFINE:
        offs = np.arange(-half, half + step / 2, step)
        gx, gy = np.meshgrid(p[0] + offs, p[1] + offs)
        pts = torch.tensor(np.c_[gx.ravel(), gy.ravel()], device=DEVICE).float()
        v = torch.cat([signal(c) for c in torch.split(pts, 16)]).cpu().numpy().reshape(gx.shape)
        i, j = np.unravel_index(np.argmax(v), v.shape)
        p = np.array([gx[i, j], gy[i, j]])
        if 0 < i < v.shape[0] - 1 and 0 < j < v.shape[1] - 1:
            dx = 0.5 * (v[i, j - 1] - v[i, j + 1]) / (v[i, j - 1] - 2 * v[i, j] + v[i, j + 1] + 1e-12)
            dy = 0.5 * (v[i - 1, j] - v[i + 1, j]) / (v[i - 1, j] - 2 * v[i, j] + v[i + 1, j] + 1e-12)
            sub_px = np.clip([dx, dy], -0.5, 0.5) * step
        else:
            sub_px = np.zeros(2)
    p = p + sub_px
    return p, float(np.log(float(signal(torch.tensor(p[None], device=DEVICE).float())[0]) + 1e-9))


# ---------------------------------------------------------------- pictures
def heat(values: np.ndarray) -> np.ndarray:
    v = (values - values.min()) / max(values.max() - values.min(), 1e-12)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def to_canvas(pt, lo, hi, size):
    return (int((pt[0] - lo[0]) / (hi[0] - lo[0]) * (size[0] - 1)), int((pt[1] - lo[1]) / (hi[1] - lo[1]) * (size[1] - 1)))


def draw_points(img, pts, lo, hi, best):
    size = (img.shape[1], img.shape[0])
    for k, p in enumerate(pts):
        c = to_canvas(p, lo, hi, size)
        cv2.circle(img, c, 5, (255, 255, 255), 1)
        cv2.putText(img, str(k), (c[0] + 5, c[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)
    cv2.drawMarker(img, to_canvas(best, lo, hi, size), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
    return img


def image_frame(img, lo, hi, w, h, color=(0, 255, 255)):
    """The image rectangle in a domain canvas."""
    size = (img.shape[1], img.shape[0])
    cv2.rectangle(img, to_canvas((0, 0), lo, hi, size), to_canvas((w, h), lo, hi, size), color, 1)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="edge_accumulate blur folder (ratio.png, static.png)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-dense", action="store_true", help="skip the dense check of stage 1")
    args = parser.parse_args()
    folder = Path(args.dir)
    image = cv2.imread(str(folder / "ratio.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    static = cv2.imread(str(folder / "static.png"), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    rng = np.random.default_rng(args.seed)
    tiles = []

    t = time.time()                                                           # stage 1
    s1 = RaySignal(image, **{k: STAGES[0][k] for k in ("scale", "sigma", "side", "step_deg")})
    lo = np.array([DOMAIN[0][0] * w, DOMAIN[1][0] * h])
    hi = np.array([DOMAIN[0][1] * w, DOMAIN[1][1] * h])
    pts1, y1, gp1, _ = optimize(s1, lo, hi, STAGES[0]["start"], STAGES[0]["iterations"], rng)
    best1 = pts1[np.argmax(y1)]
    t1 = time.time() - t
    gw, gh = 400, int(400 * (hi[1] - lo[1]) / (hi[0] - lo[0]))
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], gw), np.linspace(lo[1], hi[1], gh))
    q = torch.tensor(np.c_[gx.ravel(), gy.ravel()], device=DEVICE)
    lo_t, span = torch.tensor(lo, device=DEVICE), torch.tensor(hi - lo, device=DEVICE)
    mean1 = gp1.predict((q - lo_t) / span)[0].cpu().numpy().reshape(gh, gw)
    if not args.no_dense:
        t = time.time()
        dense = torch.cat([torch.log(s1(c.float()) + 1e-9) for c in torch.split(q, 64)]).cpu().numpy().reshape(gh, gw)
        dense_best = np.array([gx.ravel()[np.argmax(dense)], gy.ravel()[np.argmax(dense)]])
        print(f"stage 1 dense check: {gw * gh} points in {time.time() - t:.0f} s, max at "
              f"({dense_best[0]:.0f}, {dense_best[1]:.0f})")
        img = image_frame(draw_points(heat(dense), pts1, lo, hi, best1), lo, hi, w, h)
        cv2.drawMarker(img, to_canvas(dense_best, lo, hi, (gw, gh)), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
        tiles.append((f"1. signal, dense (check); GP's {len(pts1)} evaluations, green best, cyan dense max", img))
    img = image_frame(draw_points(heat(mean1), pts1, lo, hi, best1), lo, hi, w, h)
    tiles.append((f"1. GP mean after {len(pts1)} evaluations ({t1:.1f} s)", img))
    cv2.imwrite(str(folder / "vp_stage1.png"), np.vstack([tl[1] for tl in tiles]))

    t = time.time()                                                           # stage 2
    s2 = RaySignal(image, **{k: STAGES[1][k] for k in ("scale", "sigma", "side", "step_deg")})
    half = FINE_HALF_WIDTH * w
    order = np.argsort(-mean1.ravel())
    hills = [best1]
    for k in order:
        c = np.array([gx.ravel()[k], gy.ravel()[k]])
        if all(np.hypot(*(c - o)) >= CANDIDATE_SEPARATION * w for o in hills):
            hills.append(c)
        if len(hills) == CANDIDATES:
            break
    runs = []
    for c in hills:
        lo2, hi2 = c - half, c + half
        pts2, y2, gp2, _ = optimize(s2, lo2, hi2, STAGES[1]["start"], STAGES[1]["iterations"], rng)
        runs.append((float(y2.max()), pts2, y2, gp2, lo2, hi2, c))
    for value, pts, ys, _, _, _, c in runs:
        found = pts[np.argmax(ys)]
        print(f"  stage 2 around ({c[0]:.0f}, {c[1]:.0f}): ({found[0]:.0f}, {found[1]:.0f}) log signal {value:.3f}")
    _, pts2, y2, gp2, lo2, hi2, _ = max(runs, key=lambda run: run[0])
    vp = pts2[np.argmax(y2)]
    t2 = time.time() - t
    gx2, gy2 = np.meshgrid(np.linspace(lo2[0], hi2[0], 300), np.linspace(lo2[1], hi2[1], 300))
    q2 = torch.tensor(np.c_[gx2.ravel(), gy2.ravel()], device=DEVICE)
    mean2 = gp2.predict((q2 - torch.tensor(lo2, device=DEVICE)) / torch.tensor(hi2 - lo2, device=DEVICE))[0]
    img = draw_points(heat(mean2.cpu().numpy().reshape(300, 300)), pts2, lo2, hi2, vp)
    cv2.imwrite(str(folder / "vp_stage2.png"), img)
    tiles.append((f"2. GP mean around it, {len(pts2)} evaluations ({t2:.1f} s): VP ({vp[0]:.0f}, {vp[1]:.0f})", img))

    r = s2.rays(torch.tensor(vp[None], device=DEVICE).float())[0].cpu().numpy()   # profile + rays
    theta = s2.theta.cpu().numpy() * 180 / math.pi
    prof = np.full((400, 1440, 3), 20, np.uint8)
    xs = ((theta - RAY_RANGE_DEG[0]) / (RAY_RANGE_DEG[1] - RAY_RANGE_DEG[0]) * 1439).astype(int)
    ys = (380 - r / r.max() * 360).astype(int)
    for a, b, c, d in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        cv2.line(prof, (a, b), (c, d), (230, 230, 230), 1)
    cv2.putText(prof, f"R(theta) at the vanishing point, {RAY_RANGE_DEG[0]:.0f}..{RAY_RANGE_DEG[1]:.0f} deg", (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(folder / "vp_profile.png"), prof)
    tiles.append(("3. line evidence R(theta) along the rays from the VP (downwards)", prof))
    peaks = [k for k in range(len(r)) if r[k] >= r[k - 1] and r[k] >= r[(k + 1) % len(r)] and r[k] > 0.15 * r.max()]
    canvas_lo = np.minimum([0, 0], vp - 40)
    canvas_hi = np.maximum([w, h], vp + 40)
    cw = int(canvas_hi[0] - canvas_lo[0])
    ch = int(canvas_hi[1] - canvas_lo[1])
    canvas = np.zeros((ch, cw, 3), np.uint8)
    ox, oy = int(-canvas_lo[0]), int(-canvas_lo[1])
    canvas[oy:oy + h, ox:ox + w] = cv2.cvtColor(static // 2, cv2.COLOR_GRAY2BGR)
    c = (int(vp[0]) + ox, int(vp[1]) + oy)
    for k in sorted(peaks, key=lambda k: -r[k])[:20]:
        a = math.radians(theta[k])
        L = 3 * max(w, h)
        cv2.line(canvas, c, (int(c[0] + L * math.cos(a)), int(c[1] + L * math.sin(a))), (0, 220, 255), 2)
    cv2.circle(canvas, c, 14, (255, 0, 255), 3)
    cv2.rectangle(canvas, (ox, oy), (ox + w - 1, oy + h - 1), (0, 255, 255), 1)
    cv2.imwrite(str(folder / "vp_rays.png"), canvas)
    tiles.append((f"4. strongest rays through the VP on the static edges", canvas))
    cv2.imwrite(str(folder / "vp_overview.jpg"), overview(tiles, width=900), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (folder / "vp.json").write_text(json.dumps({"vanishing_point": [round(float(vp[0]), 2), round(float(vp[1]), 2)],
                                                "image_size": [w, h], "log_signal": round(float(y2.max()), 4)}))
    print(f"stage 1: {len(pts1)} evaluations {t1:.1f} s, best ({best1[0]:.0f}, {best1[1]:.0f}); "
          f"stage 2: {len(pts2)} evaluations {t2:.1f} s -> vanishing point ({vp[0]:.1f}, {vp[1]:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
