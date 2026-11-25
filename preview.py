import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import scenes

def box_tris(cx, cy, cz, sx, sy, sz):
    x0, x1 = cx - sx / 2, cx + sx / 2; y0, y1 = cy - sy / 2, cy + sy / 2; z0, z1 = cz - sz / 2, cz + sz / 2
    P = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    F = [(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),(3,2,6),(3,6,7),(0,3,7),(0,7,4),(1,5,6),(1,6,2)]
    return P, np.array(F)

def render(meshes, opts, views, out):
    V, T, C = [], [], []
    off = 0
    for v, tri, kind, smask in meshes:
        if kind == 'pt':
            col = np.where(smask[:, None], [[0.12, 0.2, 0.35]], [[0.72, 0.52, 0.40]])
        else:
            yc = v[tri].mean(1)[:, 1]
            top = yc > yc.min() + (yc.max() - yc.min()) * 0.83
            col = np.where(top[:, None], [[0.72, 0.52, 0.40]], [[0.55, 0.75, 0.9]])
            col[smask] = [0.2, 0.3, 0.5]
        V.append(v); T.append(tri + off); C.append(col); off += len(v)
    if opts.get('bed'):
        bw, bt, bl = scenes.BED['size']
        P, F = box_tris(0, scenes.BED_H - bt / 2, 0, bw, bt, bl)
        V.append(P); T.append(F + off); C.append(np.tile([0.85, 0.85, 0.8], (len(F), 1))); off += len(P)
        P, F = box_tris(0, scenes.BED_H / 2 - bt / 2, 0, bw * 0.9, scenes.BED_H - bt, bl * 0.9)
        V.append(P); T.append(F + off); C.append(np.tile([0.6, 0.55, 0.5], (len(F), 1))); off += len(P)
    V = np.vstack(V); T = np.vstack(T); C = np.vstack(C)
    fig, axs = plt.subplots(1, len(views), figsize=(6 * len(views), 5))
    if len(views) == 1: axs = [axs]
    for ax, (name, d) in zip(axs, views):
        d = np.array(d, float); d /= np.linalg.norm(d)
        up = np.array([0, 1, 0]); r = np.cross(up, d); r /= np.linalg.norm(r); u = np.cross(d, r)
        P2 = np.c_[V @ r, V @ u]; depth = V @ d
        tri = V[T]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= (np.linalg.norm(n, axis=1)[:, None] + 1e-9)
        light = np.array([0.4, 0.8, 0.45]); light /= np.linalg.norm(light)
        sh = 0.35 + 0.65 * np.abs(n @ light)
        order = np.argsort(depth[T].mean(1))
        polys = P2[T[order]]
        ax.add_collection(PolyCollection(polys, facecolors=np.clip(C[order] * sh[order, None], 0, 1), edgecolors='none'))
        ax.set_xlim(P2[:, 0].min() - 1, P2[:, 0].max() + 1); ax.set_ylim(P2[:, 1].min() - 1, P2[:, 1].max() + 1)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(name)
    plt.tight_layout(); plt.savefig(out, dpi=70); plt.close()

if __name__ == '__main__':
    name = sys.argv[1]
    meshes, opts = getattr(scenes, 'scene_' + name)()
    render(meshes, opts, [('side', (1, 0.12, 0)), ('3/4', (1.3, 0.7, 0.9)), ('end', (0.2, 0.3, 1))], '/tmp/pv-%s.png' % name)
    print('ok')
