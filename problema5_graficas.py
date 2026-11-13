"""
Problema 5 - Trabajo y dependencia del camino (Pareja 5)
Calculo Vectorial - Graficas y verificacion numerica

Como usarlo:
  1. Instala las librerias (una sola vez):   pip install numpy scipy matplotlib
  2. Ejecuta:                                python problema5_graficas.py
  Tambien funciona en Google Colab: pega todo en una celda y ejecutala.

El programa imprime los resultados numericos de cada inciso y abre 4 figuras
(ademas las guarda como PNG en la misma carpeta).
"""

import numpy as np
import matplotlib.pyplot as plt


def quad(func, a, b):
    """Integrate a smooth function numerically with Gauss-Legendre quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    points = (a + b) / 2 + (b - a) * nodes / 2
    value = (b - a) / 2 * sum(weight * func(point) for point, weight in zip(points, weights))
    return value, 0.0


def dblquad(func, a, b, lower, upper):
    """Integrate func(y, x) over a rectangular region by Gauss-Legendre quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x_points = (a + b) / 2 + (b - a) * nodes / 2
    y_points = (lower + upper) / 2 + (upper - lower) * nodes / 2
    inner_values = [
        (upper - lower) / 2 * sum(weight * func(y, x) for y, weight in zip(y_points, weights))
        for x in x_points
    ]
    value = (b - a) / 2 * sum(weight * inner for weight, inner in zip(weights, inner_values))
    return value, 0.0

# ---------------------------------------------------------------
# Campos vectoriales del problema
# ---------------------------------------------------------------
def F(x, y):
    """F(x, y) = <y, -x>  (inciso a y reto)"""
    return y, -x

def H(x, y):
    """H(x, y) = <e^x sen y + 2x, e^x cos y>  (incisos b, c, d)"""
    return np.exp(x) * np.sin(y) + 2 * x, np.exp(x) * np.cos(y)

def f_potencial(x, y):
    """Funcion potencial de H:  f(x, y) = e^x sen y + x^2"""
    return np.exp(x) * np.sin(y) + x**2

# ---------------------------------------------------------------
# Trabajo W = integral de linea de un campo sobre una curva r(t)
#   W = integral_a^b  campo(r(t)) . r'(t) dt
# ---------------------------------------------------------------
def trabajo(campo, r, dr, a, b):
    def integrando(t):
        x, y = r(t)
        dx, dy = dr(t)
        P, Q = campo(x, y)
        return P * dx + Q * dy
    valor, _ = quad(integrando, a, b)
    return valor

# ===============================================================
# (a) Trabajo de F por dos caminos distintos de (0,0) a (1,1)
# ===============================================================
# C1: segmento y = x      ->  r(t) = (t, t),    0 <= t <= 1
# C2: parabola y = x^2    ->  r(t) = (t, t^2),  0 <= t <= 1
W_C1 = trabajo(F, lambda t: (t, t),    lambda t: (1, 1),     0, 1)
W_C2 = trabajo(F, lambda t: (t, t**2), lambda t: (1, 2 * t), 0, 1)

print("=== (a) Trabajo de F = <y, -x> ===")
print(f"  Por C1 (y = x)  : W = {W_C1:.6f}   (exacto: 0)")
print(f"  Por C2 (y = x^2): W = {W_C2:.6f}   (exacto: -1/3 = {-1/3:.6f})")
print("  Dan distinto -> el trabajo depende del camino -> F NO es conservativo\n")

# ===============================================================
# (b) H es conservativo: dP/dy = dQ/dx
# ===============================================================
# Comprobacion numerica en algunos puntos con derivadas por diferencias finitas
h = 1e-6
print("=== (b) Prueba de que H es conservativo ===")
for (x0, y0) in [(0.3, 0.7), (-1.0, 2.0), (1.5, -0.4)]:
    dPdy = (H(x0, y0 + h)[0] - H(x0, y0 - h)[0]) / (2 * h)
    dQdx = (H(x0 + h, y0)[1] - H(x0 - h, y0)[1]) / (2 * h)
    print(f"  En ({x0}, {y0}):  dP/dy = {dPdy:.6f}   dQ/dx = {dQdx:.6f}")
print("  Son iguales en todo R^2 -> H es conservativo")
print("  Funcion potencial: f(x, y) = e^x sen y + x^2 + C\n")

# ===============================================================
# (c) Teorema fundamental + verificacion por dos caminos
# ===============================================================
W_TF = f_potencial(1, np.pi / 2) - f_potencial(0, 0)   # = e + 1

# Camino 1: segmento recto de (0,0) a (1, pi/2) -> r(t) = (t, (pi/2) t)
W_H1 = trabajo(H, lambda t: (t, np.pi / 2 * t),
                  lambda t: (1, np.pi / 2), 0, 1)
# Camino 2: parabola y = (pi/2) x^2           -> r(t) = (t, (pi/2) t^2)
W_H2 = trabajo(H, lambda t: (t, np.pi / 2 * t**2),
                  lambda t: (1, np.pi * t), 0, 1)

# Camino en L (el que se integra a mano en la presentacion):
#   Tramo a: (0,0) -> (1,0),     r(t) = (t, 0),  0 <= t <= 1     -> H.r' = 2t
#   Tramo b: (1,0) -> (1, pi/2), r(t) = (1, t),  0 <= t <= pi/2  -> H.r' = e cos t
W_La = trabajo(H, lambda t: (t, 0.0), lambda t: (1, 0), 0, 1)              # = 1
W_Lb = trabajo(H, lambda t: (1.0, t), lambda t: (0, 1), 0, np.pi / 2)      # = e
W_L = W_La + W_Lb

print("=== (c) Trabajo de H de (0,0) a (1, pi/2) ===")
print(f"  Camino en L, tramo a: {W_La:.6f}   (exacto: 1)")
print(f"  Camino en L, tramo b: {W_Lb:.6f}   (exacto: e = {np.e:.6f})")
print(f"  Camino en L, total  : {W_L:.6f}   (exacto: e + 1)")
print(f"  Teorema fundamental: f(1,pi/2) - f(0,0) = e + 1 = {W_TF:.6f}")
print(f"  Camino 1 (segmento recto)    : W = {W_H1:.6f}")
print(f"  Camino 2 (parabola y=(pi/2)x^2): W = {W_H2:.6f}\n")

# ===============================================================
# (d) Trabajo de H sobre una curva cerrada (circunferencia unitaria)
# ===============================================================
circ   = lambda t: (np.cos(t), np.sin(t))
d_circ = lambda t: (-np.sin(t), np.cos(t))
W_H_cerrada = trabajo(H, circ, d_circ, 0, 2 * np.pi)
print("=== (d) Trabajo de H sobre la circunferencia unitaria (cerrada) ===")
print(f"  W = {W_H_cerrada:.2e}   (exacto: 0)\n")

# ===============================================================
# Reto: integral de y dx - x dy sobre la circunferencia unitaria
# ===============================================================
W_reto = trabajo(F, circ, d_circ, 0, 2 * np.pi)
print("=== Reto (teorema de Green) ===")
print(f"  Integral de linea directa : {W_reto:.6f}")
# Lado de Green en coordenadas polares: integral de 0 a 2pi, de 0 a 1, de (-2) r dr dtheta
W_green, _ = dblquad(lambda r, th: -2 * r, 0, 2 * np.pi, 0, 1)
print(f"  Lado de Green (polares)   : {W_green:.6f}   (= -2 pi = {-2 * np.pi:.6f})\n")

# ===============================================================
# GRAFICAS
# ===============================================================
plt.rcParams.update({"font.size": 12})
X, Y = np.meshgrid(np.linspace(-0.2, 1.3, 16), np.linspace(-0.2, 1.3, 16))
t = np.linspace(0, 1, 200)

# ---- Figura 1: campo F y los caminos C1, C2 (inciso a)
U, V = F(X, Y)
fig1, ax = plt.subplots(figsize=(7, 6))
ax.quiver(X, Y, U, V, color="gray", alpha=0.6)
ax.plot(t, t, color="tab:blue", lw=3, label=f"C1: y = x   (W = {W_C1:.3f})")
ax.plot(t, t**2, color="tab:orange", lw=3, label=f"C2: y = x²  (W = {W_C2:.3f})")
ax.plot([0, 1], [0, 1], "ko")
ax.annotate("(0,0)", (0, 0), textcoords="offset points", xytext=(-30, -15))
ax.annotate("(1,1)", (1, 1), textcoords="offset points", xytext=(8, 5))
ax.set_title("(a) Campo F = <y, -x> y dos caminos")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
ax.legend(loc="lower right")
fig1.tight_layout(); fig1.savefig("fig1_campo_F.png", dpi=150)

# ---- Figura 2: superficie de la funcion potencial f (inciso b)
xs = np.linspace(-1, 1.5, 60); ys = np.linspace(-0.5, 2.5, 60)
XS, YS = np.meshgrid(xs, ys)
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(projection="3d")
ax2.plot_surface(XS, YS, f_potencial(XS, YS), cmap="viridis", alpha=0.85)
ax2.scatter([0, 1], [0, np.pi / 2], [f_potencial(0, 0), f_potencial(1, np.pi / 2)],
            color="red", s=60)
ax2.set_title("(b) Potencial f(x, y) = e^x sen y + x²")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("f")
fig2.tight_layout(); fig2.savefig("fig2_potencial.png", dpi=150)

# ---- Figura 3: campo H, curvas de nivel de f y los dos caminos (inciso c)
X3, Y3 = np.meshgrid(np.linspace(-0.2, 1.3, 15), np.linspace(-0.2, 1.9, 15))
U3, V3 = H(X3, Y3)
N = np.hypot(U3, V3)
fig3, ax3 = plt.subplots(figsize=(7, 7))
cs = ax3.contour(XS, YS, f_potencial(XS, YS), levels=15, cmap="viridis")
ax3.clabel(cs, fontsize=8)
ax3.quiver(X3, Y3, U3 / N, V3 / N, color="gray", alpha=0.6)
ax3.plot(t, np.pi / 2 * t, color="tab:blue", lw=3, label=f"Camino 1 (W = {W_H1:.4f})")
ax3.plot(t, np.pi / 2 * t**2, color="tab:orange", lw=3, label=f"Camino 2 (W = {W_H2:.4f})")
ax3.plot([0, 1, 1], [0, 0, np.pi / 2], color="tab:green", lw=3, ls="--", label=f"Camino en L (W = {W_L:.4f})")
ax3.plot([0, 1], [0, np.pi / 2], "ro")
ax3.set_xlim(-0.2, 1.3); ax3.set_ylim(-0.2, 1.9)
ax3.set_title("(c) Campo H, curvas de nivel de f y tres caminos")
ax3.set_xlabel("x"); ax3.set_ylabel("y")
ax3.legend(loc="lower right")
fig3.tight_layout(); fig3.savefig("fig3_campo_H.png", dpi=150)

# ---- Figura 4: reto, campo F sobre la circunferencia unitaria
s = np.linspace(0, 2 * np.pi, 300)
X4, Y4 = np.meshgrid(np.linspace(-1.4, 1.4, 15), np.linspace(-1.4, 1.4, 15))
U4, V4 = F(X4, Y4)
fig4, ax4 = plt.subplots(figsize=(7, 7))
ax4.quiver(X4, Y4, U4, V4, color="gray", alpha=0.6)
ax4.fill(np.cos(s), np.sin(s), color="tab:blue", alpha=0.12, label="Region D (area = pi)")
ax4.plot(np.cos(s), np.sin(s), color="tab:blue", lw=3, label="C: circunferencia antihoraria")
ax4.annotate("", xy=(0, 1), xytext=(0.15, 0.99),
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.5))
ax4.set_title("Reto: F = <y, -x> gira en sentido horario\n"
              f"integral de linea = {W_reto:.4f} = -2 pi")
ax4.set_aspect("equal"); ax4.set_xlabel("x"); ax4.set_ylabel("y")
ax4.legend(loc="lower right")
fig4.tight_layout(); fig4.savefig("fig4_reto_green.png", dpi=150)

plt.show()
