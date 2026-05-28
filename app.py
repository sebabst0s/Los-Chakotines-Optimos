from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import math
import os
import time
import sys

sys.setrecursionlimit(5000)

app = Flask(__name__)

# Pre-warm matplotlib font/layout engine at startup to avoid cold-start overhead
# and prevent any deferred initialisation from happening inside a request.
try:
    _warmup_fig, _warmup_ax = plt.subplots(1, 1, figsize=(2, 2))
    _warmup_ax.plot([0], [0])
    _warmup_buf = io.BytesIO()
    plt.savefig(_warmup_buf, format='png', dpi=72, bbox_inches='tight')
    plt.close(_warmup_fig)
    del _warmup_fig, _warmup_ax, _warmup_buf
except Exception:
    pass

# ── Safe evaluation namespace ────────────────────────────────────────────────

SAFE_BUILTINS = {
    'abs': abs, 'round': round, 'min': min, 'max': max,
    'sum': sum, 'float': float, 'int': int, 'bool': bool,
    'len': len, 'range': range, 'enumerate': enumerate,
    'zip': zip, 'list': list, 'tuple': tuple, 'pow': pow,
}

SAFE_NS = {
    '__builtins__': SAFE_BUILTINS,
    'np': np, 'numpy': np, 'math': math,
    'pi': math.pi, 'e': math.e, 'inf': float('inf'),
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'exp': np.exp, 'log': np.log, 'log2': np.log2, 'log10': np.log10,
    'sqrt': np.sqrt, 'arcsin': np.arcsin, 'arccos': np.arccos,
    'arctan': np.arctan, 'arctan2': np.arctan2,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
}


def eval_func(expr, x):
    ns = SAFE_NS.copy()
    ns['x'] = np.asarray(x, dtype=float)
    return float(eval(expr, ns))  # noqa: S307


# ── Numerical differentiation ────────────────────────────────────────────────

def gradient(f, x, h=1e-6):
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def hessian(f, x, h=1e-4):
    n = len(x)
    f0 = f(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h; xm[i] -= h
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h ** 2
            else:
                xpp = x.copy(); xpp[i] += h; xpp[j] += h
                xpm = x.copy(); xpm[i] += h; xpm[j] -= h
                xmp = x.copy(); xmp[i] -= h; xmp[j] += h
                xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
                v = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h ** 2)
                H[i, j] = H[j, i] = v
    return H


# ── Wolfe line search (Nocedal & Wright Algorithm 3.5 / 3.6) ─────────────────

def _zoom(f, gradf, x, d, a_lo, a_hi, f0, dphi0, c1, c2, phi_lo):
    for _ in range(30):
        a = (a_lo + a_hi) / 2
        phi = f(x + a * d)
        if phi > f0 + c1 * a * dphi0 or phi >= phi_lo:
            a_hi = a
        else:
            dphi = float(np.dot(gradf(x + a * d), d))
            if abs(dphi) <= c2 * abs(dphi0):
                return a
            phi_lo = phi
            if dphi * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a
        if abs(a_hi - a_lo) < 1e-14:
            break
    return a


def wolfe_line_search(f, gradf, x, d, alpha_init=1.0, c1=1e-4, c2=0.9):
    f0 = f(x)
    g0 = gradf(x)
    dphi0 = float(np.dot(g0, d))
    if dphi0 >= 0:
        return None

    a_prev, phi_prev = 0.0, f0
    a = alpha_init

    for i in range(50):
        phi = f(x + a * d)
        if phi > f0 + c1 * a * dphi0 or (i > 0 and phi >= phi_prev):
            result = _zoom(f, gradf, x, d, a_prev, a, f0, dphi0, c1, c2, phi_prev)
            return result if result and result > 1e-15 else None

        dphi = float(np.dot(gradf(x + a * d), d))
        if abs(dphi) <= c2 * abs(dphi0):
            return a

        if dphi >= 0:
            result = _zoom(f, gradf, x, d, a, a_prev, f0, dphi0, c1, c2, phi)
            return result if result and result > 1e-15 else None

        a_prev, phi_prev = a, phi
        a = min(a * 2.0, alpha_init * 100)

    return a if a > 1e-15 else None


# ── Optimization methods ──────────────────────────────────────────────────────

def _record(history, k, x, f, gn):
    history.append({'k': k, 'x': x.tolist(), 'f': float(f), 'grad_norm': float(gn)})


def steepest_descent(f, x0, max_iter, tol, c1, c2, alpha_init):
    x = x0.copy()
    history = []
    gradf = lambda xk: gradient(f, xk)

    for k in range(max_iter):
        g = gradf(x)
        gn = float(np.linalg.norm(g))
        _record(history, k, x, f(x), gn)
        if gn < tol:
            break
        d = -g
        a = wolfe_line_search(f, gradf, x, d, alpha_init, c1, c2)
        if a is None:
            break
        x = x + a * d

    return x, history


def fletcher_reeves(f, x0, max_iter, tol, c1, c2, alpha_init):
    x = x0.copy()
    n = len(x)
    history = []
    gradf = lambda xk: gradient(f, xk)

    g = gradf(x)
    d = -g.copy()

    for k in range(max_iter):
        gn = float(np.linalg.norm(g))
        _record(history, k, x, f(x), gn)
        if gn < tol:
            break

        a = wolfe_line_search(f, gradf, x, d, alpha_init, c1, c2)
        if a is None:
            d = -g
            a = wolfe_line_search(f, gradf, x, d, alpha_init, c1, c2)
            if a is None:
                break

        x_new = x + a * d
        g_new = gradf(x_new)

        gg = float(np.dot(g, g))
        beta = float(np.dot(g_new, g_new)) / max(gg, 1e-300)
        d = -g_new + beta * d
        x, g = x_new, g_new

        if (k + 1) % max(n, 1) == 0:
            d = -g

    return x, history


def newton(f, x0, max_iter, tol, c1, c2, alpha_init):
    x = x0.copy()
    history = []
    gradf = lambda xk: gradient(f, xk)

    for k in range(max_iter):
        g = gradf(x)
        gn = float(np.linalg.norm(g))
        _record(history, k, x, f(x), gn)
        if gn < tol:
            break

        H = hessian(f, x)
        H = (H + H.T) / 2

        d = None
        reg = 1e-8
        for _ in range(20):
            try:
                H_reg = H + reg * np.eye(len(x))
                np.linalg.cholesky(H_reg)
                d_cand = -np.linalg.solve(H_reg, g)
                if np.dot(g, d_cand) < 0:
                    d = d_cand
                    break
            except np.linalg.LinAlgError:
                pass
            reg *= 10

        if d is None:
            d = -g

        a = wolfe_line_search(f, gradf, x, d, alpha_init, c1, c2)
        if a is None:
            break
        x = x + a * d

    return x, history


# ── Plot helpers ─────────────────────────────────────────────────────────────

_PLOT_THEME = dict(
    BG='#f4f2ff', BG2='#ffffff', GRID='#e8e5f5',
    MUTED='#6b7280', TEXT='#2d2b55', SPINE='#ddd6fe',
)


def _style_axes(ax):
    t = _PLOT_THEME
    ax.set_facecolor(t['BG2'])
    ax.tick_params(colors=t['MUTED'], labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(t['SPINE'])
    ax.grid(True, color=t['GRID'], alpha=1.0, linestyle='--', linewidth=0.7)


def _savefig(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=115, bbox_inches='tight',
                facecolor=_PLOT_THEME['BG'])
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ── Single convergence plot ──────────────────────────────────────────────────

def make_plot(history, method_name):
    iters  = [h['k']         for h in history]
    gnorms = [h['grad_norm'] for h in history]
    fvals  = [h['f']         for h in history]
    t = _PLOT_THEME

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(t['BG'])
    _style_axes(ax1); _style_axes(ax2)

    every = max(1, len(iters) // 60)
    C_GRAD, C_FUNC = '#7c3aed', '#f97316'
    C_F1,   C_F2   = '#c4b5fd', '#fed7aa'

    ax1.semilogy(iters, gnorms, color=C_GRAD, lw=2.2,
                 marker='o', ms=2.8, markevery=every,
                 markerfacecolor=C_GRAD, markeredgewidth=0, zorder=3)
    pos_gnorms = [max(g, 1e-300) for g in gnorms]
    ax1.fill_between(iters, pos_gnorms, min(pos_gnorms),
                     color=C_F1, alpha=0.30, zorder=2)
    ax1.set_xlabel('Iteración', color=t['MUTED'], fontsize=10)
    ax1.set_ylabel('‖∇f(x)‖',  color=t['MUTED'], fontsize=10)
    ax1.set_title('Norma del gradiente', color=t['TEXT'],
                  fontsize=11, fontweight='bold', pad=10)

    ax2.plot(iters, fvals, color=C_FUNC, lw=2.2,
             marker='o', ms=2.8, markevery=every,
             markerfacecolor=C_FUNC, markeredgewidth=0, zorder=3)
    ax2.fill_between(iters, fvals, min(fvals), color=C_F2, alpha=0.35, zorder=2)
    ax2.set_xlabel('Iteración', color=t['MUTED'], fontsize=10)
    ax2.set_ylabel('f(x)',      color=t['MUTED'], fontsize=10)
    ax2.set_title('Valor de la función', color=t['TEXT'],
                  fontsize=11, fontweight='bold', pad=10)

    fig.suptitle(f'Convergencia — {method_name}',
                 color=t['TEXT'], fontsize=13, fontweight='bold', y=0.98)
    return _savefig(fig)


# ── Comparison convergence plot ──────────────────────────────────────────────

_CMP_COLORS = {'gradient': '#7c3aed', 'conjugate': '#0ea5e9', 'newton': '#f43f5e'}
_CMP_LABELS = {'gradient': 'Steepest Descent', 'conjugate': 'Fletcher-Reeves', 'newton': 'Newton'}


def make_comparison_plot(results):
    t = _PLOT_THEME
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(t['BG'])
    _style_axes(ax1); _style_axes(ax2)

    leg_kw = dict(fontsize=8.5, framealpha=0.9, facecolor='#fafafa',
                  edgecolor=t['SPINE'], labelcolor=t['TEXT'])

    for key in ('gradient', 'conjugate', 'newton'):
        res = results.get(key, {})
        hist = res.get('history', [])
        if not hist:
            continue
        iters  = [h['k'] for h in hist]
        gnorms = [max(h['grad_norm'], 1e-300) for h in hist]
        fvals  = [h['f'] for h in hist]
        c      = _CMP_COLORS[key]
        lbl    = _CMP_LABELS[key]
        every  = max(1, len(iters) // 60)

        ax1.semilogy(iters, gnorms, color=c, lw=2.2, label=lbl,
                     marker='o', ms=2.5, markevery=every,
                     markerfacecolor=c, markeredgewidth=0, alpha=0.9, zorder=3)
        ax2.plot(iters, fvals, color=c, lw=2.2, label=lbl,
                 marker='o', ms=2.5, markevery=every,
                 markerfacecolor=c, markeredgewidth=0, alpha=0.9, zorder=3)

    ax1.set_xlabel('Iteración', color=t['MUTED'], fontsize=10)
    ax1.set_ylabel('‖∇f(x)‖',  color=t['MUTED'], fontsize=10)
    ax1.set_title('Norma del gradiente', color=t['TEXT'],
                  fontsize=11, fontweight='bold', pad=10)
    ax1.legend(**leg_kw)

    ax2.set_xlabel('Iteración', color=t['MUTED'], fontsize=10)
    ax2.set_ylabel('f(x)',      color=t['MUTED'], fontsize=10)
    ax2.set_title('Valor de la función', color=t['TEXT'],
                  fontsize=11, fontweight='bold', pad=10)
    ax2.legend(**leg_kw)

    fig.suptitle('Comparación de métodos — Convergencia',
                 color=t['TEXT'], fontsize=13, fontweight='bold', y=0.98)
    return _savefig(fig)


# ── Routes ────────────────────────────────────────────────────────────────────

METHOD_NAMES = {
    'gradient':  'Gradiente descendente (Steepest Descent)',
    'conjugate': 'Gradiente conjugado (Fletcher-Reeves)',
    'newton':    'Método de Newton (Hessiano numérico)',
}

METHOD_FNS = {
    'gradient':  steepest_descent,
    'conjugate': fletcher_reeves,
    'newton':    newton,
}


def _parse_common(data):
    """Parse and validate shared params. Returns (func_str, n, x0, max_iter, tol, c1, c2, alpha) or raises."""
    func_str = (data.get('function') or '').strip()
    if not func_str:
        raise ValueError('Ingresa una función objetivo.')

    try:
        n = int(data.get('n', 2))
        assert 1 <= n <= 20
    except Exception:
        raise ValueError('n debe ser un entero entre 1 y 20.')

    try:
        x0 = np.array([float(v) for v in str(data.get('x0', '')).split(',')], dtype=float)
    except Exception:
        raise ValueError('Punto de partida inválido. Usa formato: 1, 2, 3')

    if len(x0) != n:
        raise ValueError(f'El punto de partida tiene {len(x0)} valor(es), se esperaban {n}.')

    try:
        max_iter   = max(1, min(int(data.get('max_iter', 1000)), 10000))
        tol        = float(data.get('tol', 1e-6))
        c1         = float(data.get('c1', 1e-4))
        c2         = float(data.get('c2', 0.9))
        alpha_init = float(data.get('alpha_init', 1.0))
    except Exception as exc:
        raise ValueError(f'Parámetro inválido: {exc}')

    if not (0 < c1 < c2 < 1):
        raise ValueError('Los parámetros de Wolfe deben cumplir 0 < c1 < c2 < 1.')
    if alpha_init <= 0:
        raise ValueError('alpha_init debe ser positivo.')

    return func_str, n, x0, max_iter, tol, c1, c2, alpha_init


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json(force=True)

        try:
            func_str, n, x0, max_iter, tol, c1, c2, alpha_init = _parse_common(data)
        except ValueError as e:
            return jsonify(error=str(e)), 400

        method = data.get('method', 'gradient')
        if method not in METHOD_FNS:
            return jsonify(error='Método no reconocido.'), 400

        def f(x):
            return eval_func(func_str, x)

        try:
            v0 = f(x0)
            if not np.isfinite(v0):
                return jsonify(
                    error='La función devuelve un valor no finito en el punto de partida.'
                ), 400
        except Exception as exc:
            return jsonify(error=f'Error evaluando la función: {exc}'), 400

        x_opt, history = METHOD_FNS[method](f, x0, max_iter, tol, c1, c2, alpha_init)

        if not history:
            return jsonify(error='El optimizador no produjo resultados.'), 500

        f_opt     = float(f(x_opt))
        final_gn  = history[-1]['grad_norm']
        converged = final_gn < tol

        return jsonify(
            success     = True,
            x_opt       = [round(v, 10) for v in x_opt.tolist()],
            f_opt       = f_opt,
            iterations  = len(history),
            final_gn    = final_gn,
            converged   = converged,
            method_name = METHOD_NAMES[method],
            plot        = make_plot(history, METHOD_NAMES[method]),
            history     = history[:200],
        )

    except Exception as exc:
        import traceback
        return jsonify(error=str(exc), detail=traceback.format_exc()), 500


@app.route('/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json(force=True)

        try:
            func_str, n, x0, max_iter, tol, c1, c2, alpha_init = _parse_common(data)
        except ValueError as e:
            return jsonify(error=str(e)), 400

        def f(x):
            return eval_func(func_str, x)

        try:
            v0 = f(x0)
            if not np.isfinite(v0):
                return jsonify(
                    error='La función devuelve un valor no finito en el punto de partida.'
                ), 400
        except Exception as exc:
            return jsonify(error=f'Error evaluando la función: {exc}'), 400

        results = {}
        for key, method_fn in METHOD_FNS.items():
            t0 = time.perf_counter()
            x_opt, history = method_fn(f, x0.copy(), max_iter, tol, c1, c2, alpha_init)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if not history:
                results[key] = {'error': 'No produjo resultados'}
                continue

            f_opt     = float(f(x_opt))
            final_gn  = history[-1]['grad_norm']
            converged = final_gn < tol

            results[key] = {
                'x_opt':       [round(v, 10) for v in x_opt.tolist()],
                'f_opt':       f_opt,
                'iterations':  len(history),
                'final_gn':    final_gn,
                'converged':   converged,
                'time_ms':     elapsed_ms,
                'method_name': METHOD_NAMES[key],
                'history':     history[:200],
            }

        plot = make_comparison_plot(results)
        return jsonify(success=True, results=results, plot=plot)

    except Exception as exc:
        import traceback
        return jsonify(error=str(exc), detail=traceback.format_exc()), 500


@app.route('/contour', methods=['POST'])
def contour():
    try:
        data = request.get_json(force=True)

        func_str = (data.get('function') or '').strip()
        if not func_str:
            return jsonify(error='Función vacía.'), 400

        x_min = float(data.get('x_min', -5))
        x_max = float(data.get('x_max',  5))
        y_min = float(data.get('y_min', -5))
        y_max = float(data.get('y_max',  5))
        res   = max(20, min(int(data.get('resolution', 80)), 150))

        xs = np.linspace(x_min, x_max, res)
        ys = np.linspace(y_min, y_max, res)
        Z  = np.full((res, res), np.nan)

        for i, yv in enumerate(ys):
            for j, xv in enumerate(xs):
                try:
                    v = eval_func(func_str, np.array([xv, yv]))
                    Z[i, j] = v if np.isfinite(v) else np.nan
                except Exception:
                    pass

        finite = Z[np.isfinite(Z)]
        if len(finite) > 0:
            p5, p95 = np.percentile(finite, 5), np.percentile(finite, 95)
            span    = max(abs(p95 - p5), 1e-10)
            Z = np.where(np.isfinite(Z),
                         np.clip(Z, p5 - span * 2.0, p95 + span * 2.0),
                         np.nan)

        zmax   = float(np.nanmax(Z)) if len(finite) > 0 else 0.0
        Z_out  = np.where(np.isfinite(Z), Z, zmax).tolist()

        return jsonify(
            success=True,
            xs=xs.tolist(), ys=ys.tolist(), Z=Z_out,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
        )

    except Exception as exc:
        import traceback
        return jsonify(error=str(exc), detail=traceback.format_exc()), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
