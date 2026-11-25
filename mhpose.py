"""จัดท่าหุ่น MakeHuman ด้วย linear blend skinning (ตามสกิล makehuman-pose-render)
หน่วยโมเดล = เดซิเมตร, Y ขึ้น, หน้า = +Z, ซ้ายคน (.L) = +X
"""
import json, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MH = os.path.join(HERE, 'mh')
_base = np.load(os.path.join(MH, 'base.npz'))
_skel = json.load(open(os.path.join(MH, 'default.mhskel')))
_w = json.load(open(os.path.join(MH, 'default_weights.mhw')))['weights']
_tg = np.load(os.path.join(MH, 'targets_asian.npz'))

# ชื่อ face group
_fg = bytes(_base['fgstr']).decode()
_idx = list(_base['fgidx']) + [len(_fg)]
GROUPS = [_fg[_idx[i]:_idx[i + 1]] for i in range(len(_idx) - 1)]


def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == 'X': return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'Y': return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R(*pairs):
    """R(('X',30),('Z',10)) หมุนตัวแรกก่อน"""
    M = np.eye(3)
    for ax, d in pairs: M = rot(ax, d) @ M
    return M


class Human:
    def __init__(self, target='asian-male-young', weight=1.0):
        co = _base['coord'].astype(np.float64).copy()
        k = 'targets/macrodetails/' + target
        co[_tg[k + '.index']] += weight * _tg[k + '.vector'].astype(np.float64) * 1e-3
        self.rest = co
        fv = _base['fvert']; grp = _base['group']
        body = GROUPS.index('body')
        tri = []
        for f, g in zip(fv, grp):
            if g != body: continue
            a, b, c, d = [int(x) for x in f]
            for t in ((a, b, c), (a, c, d)):
                if len(set(t)) == 3: tri.append(t)
        self.tri = np.array(tri)
        self.joints = {n: co[v].mean(0) for n, v in _skel['joints'].items()}
        self.bones = _skel['bones']
        self.order = []
        seen = set()
        def add(b):
            if b in seen: return
            p = self.bones[b]['parent']
            if p: add(p)
            seen.add(b); self.order.append(b)
        for b in self.bones: add(b)
        # weights เป็นเมทริกซ์ sparse ต่อ bone
        self.W = {b: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for b, v in _w.items()}

    def head(self, b): return self.joints[self.bones[b]['head']]
    def tail(self, b): return self.joints[self.bones[b]['tail']]

    def pose(self, rots):
        """rots: {bone: 3x3} แกนโลกของ rest pose หมุนรอบ head ของ bone"""
        G = {}
        for b in self.order:
            A = np.eye(4)
            if b in rots:
                Rm = rots[b]; h = self.head(b)
                A[:3, :3] = Rm; A[:3, 3] = h - Rm @ h
            p = self.bones[b]['parent']
            G[b] = (G[p] @ A) if p else A
        self.G = G
        n = len(self.rest)
        acc = np.zeros((n, 3)); ws = np.zeros(n)
        hv = np.c_[self.rest, np.ones(n)]
        for b, (vi, w) in self.W.items():
            if b not in G: continue
            acc[vi] += w[:, None] * (hv[vi] @ G[b].T)[:, :3]
            ws[vi] += w
        out = self.rest.copy()
        m = ws > 0
        out[m] = acc[m] / ws[m][:, None]
        self.v = out
        return out

    def jpos(self, b, end='head'):
        p = self.head(b) if end == 'head' else self.tail(b)
        return (self.G[b] @ np.r_[p, 1])[:3]

    def shorts_mask(self):
        c = self.rest[self.tri].mean(1)
        return (c[:, 1] > -1.7) & (c[:, 1] < 1.1) & (np.abs(c[:, 0]) < 2.6)


def place(v, Rm=np.eye(3), t=(0, 0, 0)):
    return v @ Rm.T + np.array(t)


def skel_G(h, rots):
    G = {}
    for b in h.order:
        A = np.eye(4)
        if b in rots:
            Rm = rots[b]; hd = h.head(b)
            A[:3, :3] = Rm; A[:3, 3] = hd - Rm @ hd
        p = h.bones[b]['parent']
        G[b] = (G[p] @ A) if p else A
    return G


def jp(h, G, b, end='head'):
    p = h.head(b) if end == 'head' else h.tail(b)
    return (G[b] @ np.r_[p, 1])[:3]


def solve_arm(h, rots, side, target_world, world_R, world_t, end='hand', guess=(-60, 0, 0, -30), elbow_lim=(-150, 0)):
    """หามุมไหล่ (X flex, Z abd, Y rot) + ศอก (X) ให้ข้อมือ/มือไปถึงเป้า (พิกัดโลก, dm)"""
    from scipy.optimize import minimize
    sg = 1 if side == 'L' else -1
    tgt = np.asarray(target_world, float)
    ua, la = 'upperarm01.' + side, 'lowerarm01.' + side
    hand = 'wrist.' + side if 'wrist.' + side in h.bones else 'hand.' + side
    def f(x):
        r = dict(rots)
        r[ua] = R(('Z', -44 * sg), ('X', x[0]), ('Z', x[1] * sg), ('Y', x[2]))
        r[la] = R(('X', x[3]))
        G = skel_G(h, r)
        p = jp(h, G, hand, 'tail' if end == 'hand' else 'head')
        pw = p @ world_R.T + world_t
        return np.sum((pw - tgt) ** 2) + 1e-4 * (x[3] ** 2) * 0
    bnds = [(-170, 60), (-30, 80), (-90, 90), elbow_lim]
    res = min((minimize(f, np.array(g, float), method='L-BFGS-B', bounds=bnds) for g in
               [guess, (-80, 20, 0, -60), (-40, 10, 30, -10), (-100, 30, -30, -90)]), key=lambda r: r.fun)
    x = res.x
    return R(('Z', -44 * sg), ('X', x[0]), ('Z', x[1] * sg), ('Y', x[2])), R(('X', x[3])), np.sqrt(res.fun), x
