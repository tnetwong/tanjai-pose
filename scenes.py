"""ฉากท่าตรวจ: คืนค่า list ของ mesh (verts dm ในพิกัดโลก, tri, kind) + กล่องอุปกรณ์"""
import numpy as np
from mhpose import Human, R, rot, skel_G, jp, solve_arm

BED_H = 6.6     # ความสูงเตียง (dm)
BED = dict(size=(7.0, 0.8, 19.0))  # กว้าง x หนา x ยาว (dm)

_cache = {}
def human(t):
    if t not in _cache: _cache[t] = Human(t)
    return _cache[t]

ARMS_DOWN = lambda: {'upperarm01.L': R(('Z', -44)), 'upperarm01.R': R(('Z', 44))}
# แขนแนบลำตัว ฝ่ามือวางบนเตียง (ตอนนอน)
ARMS_SIDE = lambda: {'upperarm01.L': R(('Z', -44), ('X', 22)), 'upperarm01.R': R(('Z', 44), ('X', 22))}


def lay_supine(h, rots):
    """นอนหงายบนเตียง หัวไป -Z เท้าไป +Z ด้านหลังแตะผิวเตียง"""
    Rw = rot('X', -90)
    v = h.pose(rots) @ Rw.T
    back = h.pose({})  # ใช้ rest เพื่อหาระดับหลัง
    vb = back @ Rw.T
    t = np.array([0, BED_H - vb[:, 1].min(), -vb[:, 2].mean()])
    return Rw, t


def scene_slr():
    pt = human('asian-male-old')
    r = ARMS_SIDE()
    r['upperleg01.R'] = R(('X', -45))          # ยกขาขวาเหยียดตรง 45 องศา
    Rw, t = lay_supine(pt, r)
    vp = pt.pose(r) @ Rw.T + t
    G = pt.G
    heel = jp(pt, G, 'foot.R', 'head') @ Rw.T + t
    knee = jp(pt, G, 'lowerleg01.R', 'head') @ Rw.T + t
    heel_under = heel + np.array([0, -0.6, 0.3])
    ld = (heel - knee) / np.linalg.norm(heel - knee)          # แนวขา
    nf = np.cross(ld, [1, 0, 0]); nf = nf if nf[1] > 0 else -nf  # ด้านหน้าหน้าแข้ง
    knee_front = knee + nf * 0.65
    # ผู้ตรวจ ยืนข้างขวาผู้ป่วย หันหน้าเข้าหาเตียง (+X)
    ex = human('asian-female-young')
    er = ARMS_DOWN()
    er['spine03'] = R(('X', 10)); er['spine02'] = R(('X', 8))
    Rx = rot('Y', 90)
    ex.pose({})
    foot_y = (ex.rest @ Rx.T)[:, 1].min()
    tx = np.array([-(BED['size'][0] / 2 + 2.2), -foot_y, knee[2] + 1.5])
    ua, la, e1, _ = solve_arm(ex, er, 'R', heel_under, Rx, tx, guess=(-70, 10, 0, -40))
    er['upperarm01.R'], er['lowerarm01.R'] = ua, la
    ua, la, e2, _ = solve_arm(ex, er, 'L', knee_front, Rx, tx, guess=(-80, 0, 0, -60))
    er['upperarm01.L'], er['lowerarm01.L'] = ua, la
    ve = ex.pose(er) @ Rx.T + tx
    print('slr hand error dm: heel %.2f knee %.2f' % (e1, e2))
    return [(vp, pt.tri, 'pt', pt.shorts_mask()), (ve, ex.tri, 'ex', ex.shorts_mask())], dict(bed=True)
