"""ฉากท่าตรวจ: คืน parts (ชื่อ, verts dm พิกัดโลก, tri, mask, สี) + อุปกรณ์ + กล้อง 2 มุม"""
import numpy as np
from mhpose import Human, R, rot, skel_G, jp, solve_arm2, cloth_masks

BED_H = 6.6                       # ความสูงพื้นเตียง (dm)
BED = (7.0, 0.8, 19.0)            # กว้าง x หนา x ยาว (dm)
SKIN = (0.72, 0.52, 0.40); BEDTOP = (0.78, 0.82, 0.88); SHORTS = (0.12, 0.20, 0.35)
SCRUB = (0.45, 0.68, 0.85); PANTS = (0.16, 0.24, 0.40)
EXAMINER = (0.29, 0.42, 0.52)     # ผู้ตรวจสีเดียวทั้งตัว (slate) ให้เห็นจุดที่มือสัมผัสชัด

_cache = {}
def human(t):
    if t not in _cache: _cache[t] = Human(t)
    return _cache[t]

ARMS_DOWN = lambda: {'upperarm01.L': R(('Z', -44)), 'upperarm01.R': R(('Z', 44))}
ARMS_SIDE = lambda: {'upperarm01.L': R(('Z', -44), ('X', 22)), 'upperarm01.R': R(('Z', 44), ('X', 22))}


def patient_parts(h, v):
    m = cloth_masks(h)
    return [('pt_skin', v, h.tri, ~m['shorts'], SKIN), ('pt_shorts', v, h.tri, m['shorts'], SHORTS)]


def examiner_parts(h, v):
    return [('ex_body', v, h.tri, np.ones(len(h.tri), bool), EXAMINER)]


def supine_world(h):
    """นอนหงายบนเตียง หัวไป -Z เท้าไป +Z หลังแตะผิวเตียง"""
    Rw = rot('X', -90)
    vb = h.rest @ Rw.T
    return Rw, np.array([0, BED_H - vb[:, 1].min() - 0.15, -vb[:, 2].mean()])


def standing_world(h, yaw, x, z):
    Rw = rot('Y', yaw)
    return Rw, np.array([x, -(h.rest @ Rw.T)[:, 1].min(), z])


def scene_slr():
    pt = human('asian-male-old')
    r = ARMS_SIDE(); r['upperleg01.R'] = R(('X', -45))        # ยกขาขวาเหยียดตรง 45 องศา
    Rw, t = supine_world(pt)
    G = skel_G(pt, r)
    W = lambda b, e='head': jp(pt, G, b, e) @ Rw.T + t
    knee, ankle = W('lowerleg01.R'), W('foot.R')
    ld = (ankle - knee) / np.linalg.norm(ankle - knee)               # แนวขา (ไปทางปลายเท้า)
    nf = np.cross(ld, [1, 0, 0]); nf = nf if nf[1] > 0 else -nf      # ด้านหน้าหน้าแข้ง
    lat = np.array([-1.0, 0, 0])                                      # ด้านข้างขาขวา (ฝั่งผู้ตรวจ)
    ex = human('asian-female-young')
    er = ARMS_DOWN(); er['spine03'] = R(('X', 8)); er['spine02'] = R(('X', 6))
    hip = W('upperleg01.R')
    Ex, tx = standing_world(ex, 90, -(BED[0] / 2 + 1.6), knee[2] - 0.6)          # ยืนระดับเข่า
    # มือขวาประคองใต้ส้นเท้า: ข้อมือใต้น่องเหนือส้น ปลายนิ้วอ้อมไปใต้ส้นเท้า
    heel = ankle - nf * 0.5
    arm, e1, _ = solve_arm2(ex, er, 'R', [('wrist.*', 'head', heel - ld * 1.0 - nf * 0.25, 1),
                                          ('finger3-3.*', 'tail', heel + ld * 0.45 - nf * 0.35, 1)], Ex, tx)
    er.update(arm)
    # มือซ้ายวางทาบหน้าเข่า: ข้อมือด้านนอกเข่า (ฝั่งผู้ตรวจ) ปลายนิ้วพาดข้ามลูกสะบ้าไปด้านใน
    kf = knee + nf * 0.55
    sh = jp(ex, skel_G(ex, er), 'upperarm01.L') @ Ex.T + tx
    elb = sh + np.array([0.9, -1.8, -1.3])                            # ศอกกางออกหน้าลำตัว ไม่ให้แขนจมเข้าอก
    arm, e2, _ = solve_arm2(ex, er, 'L', [('wrist.*', 'head', kf + lat * 1.0 - ld * 0.1, 1),
                                          ('finger3-3.*', 'tail', kf - lat * 0.5, 1),
                                          ('lowerarm01.*', 'head', elb, 0.5)], Ex, tx, guess=(-80, 0, 0, -60, 40, 0, 0))
    er.update(arm)
    print('slr hand error (dm): heel %.2f knee %.2f' % (e1, e2))
    vp = pt.pose(r) @ Rw.T + t
    ve = ex.pose(er) @ Ex.T + tx
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), bed=True,
                cams=[('side', (1, 0.15, 0.05)), ('34', (1.2, 0.75, 0.9))])
