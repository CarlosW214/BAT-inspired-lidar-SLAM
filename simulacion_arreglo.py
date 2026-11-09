import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


N_senores = 15
alcance_max = 10 #[m]
R0 = 0.03 #[m]


sensores_lidar = {
    "TFmini_S": dict(
        a_v= 2.3, a_h= 2.0, rho_min=0.10, f_max=500, exact = 0.04, pitch =0.02),
    "TF_NOVA": dict(
        a_v= 1, a_h= 14, rho_min=0.10, f_max=900, exact = 0.05, pitch =0.024
    )
}

# Simulando un entorno interior controlado. 
R_ROI = 6; altura_techo = 3
R_CERCA = 2.0; Z_CERCA = 0.5
S_OBJ = 0.10; D_OBJ = 6.0

p_ciego = np.degrees(2*np.arctan(S_OBJ/(2*D_OBJ)))
altura_Marreglo = 1.5

D = 0.02
nr, nz = int(R_ROI/D), int(altura_techo/D)
r = (np.arange(nr) + 0.5)*D
z = (np.arange(nz) + 0.5)*D

RR, ZZ = np.meshgrid(r, z)
W = RR.copy()
CERCA = (RR <= R_CERCA) & (ZZ <= Z_CERCA)
W1,W2 = W / W.sum(), (W*CERCA) / (W*CERCA).sum()

Angulos = np.round(np.arange(-85,85,0.05),2)
Q = int(np.ceil((altura_Marreglo + 0.25)/D))


def tabla_mask(s):
    zz = (np.arange(-Q, nz) + 0.5) * D
    Rr, Zr = np.meshgrid(r, zz)
    phi = np.degrees(np.arctan2(Zr, Rr - R0))
    rho = np.hypot(Rr - R0, Zr)

    condicion = (Rr >= R0) & (rho >= s["rho_min"]) & (rho <= alcance_max) 
    M = np.empty((len(Angulos), len(zz), int(np.ceil(nr/8))), np.uint8)
    for k, a in enumerate(Angulos):
        M[k] = np.packbits(condicion & (np.abs(phi - a) <= s["a_v"]/2), axis=1)
    return M

def angulos(x):
    return x[0] + np.concatenate([[0], np.cumsum(x[1:N_senores])])

def altura_o(h,s):
    return h + (np.arange(N_senores) - (N_senores-1)/2)*s["pitch"]

def cobertura(x,s,M):
    th, h = angulos(x), x[N_senores]
    idx = np.clip(np.round((th - Angulos[0]) / 0.05).astype(int), 0, len(Angulos)-1)
    m = np.round(altura_o(h,s)/D).astype(int)
    acc = np.zeros(M.shape[1:], np.uint8)[:nz]

    for k, mi in zip(idx, m):
        acc |= M[k, Q - mi:Q - mi + nz]
    cub = np.unpackbits(acc, axis=1)[:, :nr].astype(bool)

    return cub, (W1 * cub).sum(), (W2 * cub).sum()

def d_suelo(x,s):
    z1 = altura_o(x[N_senores],s)[0]
    e = angulos(x)[0] - s["a_v"]/2

    return z1 / np.tan(np.radians(-e)) if e < 0 else np.inf

def resolver(s,M, eps=None, h_fijo=None, seed=0):
    def obj(x):
        _, f1, _ = cobertura(x,s,M)
        pen = 0.0
        if eps is not None:
            z1 = altura_o(x[N_senores],s)[0]
            e_req = -np.degrees(np.arctan(z1/eps))
            pen = max(0.0, (x[0] -  s["a_v"]/2) - e_req)
        return -f1 + 0.2 * pen

    col = (N_senores - 1) / 2 * s["pitch"]
    hb = (col + 0.05, altura_Marreglo) if h_fijo is None else (h_fijo, h_fijo + 1e-8)
    bounds = [(-60, 30)] + [(0, s["a_v"] + p_ciego )] * (N_senores - 1) + [hb]
    resolucion = differential_evolution(obj, bounds, seed=seed, popsize=15, maxiter=350, tol=1e-9, polish=False, init="sobol")

    _,f1,f2 = cobertura(resolucion.x,s,M)
    return resolucion.x, f1, d_suelo(resolucion.x,s)

EPS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
D_CAP = 6.0

def pareto_frente(s, M):
    P = []
    for e in EPS:
        x, f1, d = resolver(s, M, eps=e)
        if e is None or d <= e +0.02:
            P.append((f1,-min(d, D_CAP),x))

    P = [p for p in P if not any ((q[0] >= p[0] and q[1] >= p[1]) and (q[0] > p[0] or q[1] > p[1]) for q in P)]

    unicos = {}
    for p in P:
        unicos.setdefault((round(p[0], 3), round(p[1], 2)), p)
    return sorted(unicos.values(), key=lambda p: -p[1])

def rodilla(P):
    f = np.array([[p[0], p[1]] for p in P])
    if np.ptp(f[:,0]) < 0.005:
        return P[int(np.argmax(f[:,1] + f[:,1]))]
    if len(f) < 3:
        return P[int(np.argmax(f[:,0] + f[:,1]))]

    fn = (f - f.min(0)) / np.ptp(f,0).clip(1e-12)
    a,b = fn[0], fn[-1]
    u,v = b - a, fn - a
    d = np.abs(u[0] * v[:,1] - u[1] * v[:,0]) / np.linalg.norm(u)
    return P[int(np.argmax(d))]
    
def zonas_ciegas(x, s, cub):
    th, h = angulos(x), x[N_senores]
    zs = altura_o(h, s)
    inf = th[0] - s["a_v"] / 2
    sup = th[-1] + s["a_v"] / 2
    d_pis = zs[0] / np.tan(np.radians(-inf)) if inf < 0 else np.inf
    d_techo = (altura_techo - zs[-1]) / np.tan(np.radians(sup)) if sup > 0 else np.inf
    huecos = np.clip(np.diff(th) - s["a_v"], 0, None)
    hmax = huecos.max()
    return {
        "zona ciega cercana [m]": s["rho_min"],
        "piso visible desde [m]": d_pis,
        "techo visible desde [m]": d_techo,
        "hueco max entre haces [deg]": hmax,
        "objeto que se cuela a 6 m [cm]": 100 * 2 * 6 * np.tan(np.radians(hmax / 2)),
        "volumen ciego del salon [%]": 100 * (W1 * ~cub).sum(),
        "volumen ciego zona cercana [%]": 100 * (W2 * ~cub).sum(),
        "giro max sin huecos horiz. [rev/s]": s["f_max"] * s["a_h"] / 360,
        "giro max p/ 10 cm a 6 m [rev/s]": s["f_max"] * min(s["a_h"], p_ciego) / 360,
        "error angular peor caso a 6 m [cm]": 100 * 6 * np.tan(np.radians(max(s["a_v"], s["a_h"]) / 2)),
        "cumple 500 Hz": s["f_max"] >= 500,
    }

#Optimizacion del arreglo. 

def sensor_opt(nombre):
    s = sensores_lidar[nombre]
    M = tabla_mask(s)
    P = pareto_frente(s, M)
    f1, mf2, x = rodilla(P)
    eps_star = max(-mf2, 0.5)
    col = (N_senores - 1) / 2 * s["pitch"]
    curva_h = []
    for h in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5]:
        if h >= col + 0.05:
            xh, fh, dh = resolver(s, M, eps=eps_star, h_fijo=h)
            curva_h.append((h, fh, dh, angulos(xh)))
    cub, _, _ = cobertura(x, s, M)
    return nombre, dict(P=P, x=x, cub=cub, curva_h=curva_h, eps=eps_star)



if __name__ == "__main__":
    from multiprocessing import Pool
    print(f"hueco max permitido = {p_ciego:.3f} deg (objeto {S_OBJ*100:.0f} cm a {D_OBJ} m)")
    with Pool(3) as pool:
        resultados = dict(pool.map(sensor_opt, list(sensores_lidar)))

    for nombre, R in resultados.items():
        s, x = sensores_lidar[nombre], R["x"]
        th = angulos(x)
        print(f"\n================ {nombre} ================")
        print("frente de Pareto:  radio zona ciega inferior [m] | cobertura salon")
        for p in R["P"]:
            print(f"   {-p[1]:5.2f}{'+' if -p[1] >= D_CAP else ' '} | {100*p[0]:5.1f} %")
        print(f"RODILLA -> altura del centro del arreglo h = {x[N_senores]:.3f} m")
        print("angulos de elevacion [deg] (abajo -> arriba):", np.round(th, 2))
        print("separaciones [deg]:", np.round(np.diff(th), 2))
        print(f"campo de vision vertical: {th[0]-s['a_v']/2:.2f} a "
              f"{th[-1]+s['a_v']/2:.2f} deg = {th[-1]-th[0]+s['a_v']:.2f} deg")
        for k, v in zonas_ciegas(x, s, R["cub"]).items():
            print(f"   {k:38s} {v}" if isinstance(v, (bool, np.bool_)) else f"   {k:38s} {v:.3f}")
        print(f"cobertura optima vs altura (zona ciega inferior <= {R['eps']:.2f} m):")
        for h, fh, dh, th_h in R["curva_h"]:
            print(f"   h = {h:4.2f} m -> {100*fh:5.1f} %  (piso desde {dh:.2f} m)  "
                  f"angulos: {np.array2string(np.round(th_h, 1), max_line_width=200)}")

    colores = {"TFmini_S": "#2a78d6", "TF_NOVA": "#eb6834"}
    marcas = {"TFmini_S": "o","TF_NOVA": "s"}

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for n, R in resultados.items():
        f = np.array([[-p[1], 100 * p[0]] for p in R["P"]])
        ax[0].plot(f[:, 0], f[:, 1], "-", marker=marcas[n], c=colores[n], lw=2, ms=8, label=n)
        k = [p for p in R["P"] if np.array_equal(p[2], R["x"])][0]
        ax[0].plot(-k[1], 100 * k[0], "*", c="#222", ms=16, zorder=5)
        ch = np.array([c[:2] for c in R["curva_h"]])
        ax[1].plot(ch[:, 0], 100 * ch[:, 1], "-", marker=marcas[n], c=colores[n],
                   lw=2, ms=8, label=f"{n} (piso ≤ {R['eps']:.2f} m)")
    ax[0].set_xlabel("radio de la zona ciega inferior [m]  (6 = piso no visible)")
    ax[0].set_ylabel("cobertura volumétrica del salón [%]")
    ax[0].set_title("Frentes de Pareto (★ = punto de rodilla)")
    ax[1].set_xlabel("altura del centro del arreglo h [m]")
    ax[1].set_ylabel("cobertura volumétrica óptima [%]")
    ax[1].set_title("Cobertura óptima según altura de montaje")
    for a in ax:
        a.grid(alpha=0.25); a.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig("pareto.png", dpi=140)

    fig, axs = plt.subplots(1, 2, figsize=(16, 4.6), sharey=True)
    for a, (n, R) in zip(axs, resultados.items()):
        a.imshow(R["cub"], origin="lower", extent=[0, R_ROI, 0, altura_techo],
                 cmap="Greys_r", vmin=-0.6, vmax=1.0, aspect="auto")
        th, zs = angulos(R["x"]), altura_o(R["x"][N_senores], sensores_lidar[n])
        for t, z0 in zip(th, zs):
            a.plot([R0, R0 + alcance_max * np.cos(np.radians(t))],
                   [z0, z0 + alcance_max * np.sin(np.radians(t))], lw=0.5, c=colores[n])
        a.set_xlim(0, R_ROI); a.set_ylim(0, altura_techo)
        a.set_title(f"{n}: claro = visto, oscuro = zona ciega", fontsize=10)
        a.set_xlabel("distancia horizontal r [m]")
    axs[0].set_ylabel("altura z [m]")
    plt.tight_layout(); plt.savefig("cortes.png", dpi=140)