"""ฉากท่าตรวจ: คืน parts (ชื่อ, verts dm พิกัดโลก, tri, mask, สี) + อุปกรณ์ + กล้อง 2 มุม"""
import numpy as np
from mhpose import Human, R, rot, skel_G, jp, solve_arm2, solve_bones, cloth_masks

BED_H = 6.6                       # ความสูงพื้นเตียง (dm)
BED = (7.0, 0.8, 19.0)            # กว้าง x หนา x ยาว (dm)
SKIN = (0.72, 0.52, 0.40); BEDTOP = (0.78, 0.82, 0.88); SHORTS = (0.12, 0.20, 0.35)
SCRUB = (0.45, 0.68, 0.85); PANTS = (0.16, 0.24, 0.40)
EXAMINER = (0.353, 0.373, 0.40)   # ผู้ตรวจเทาเข้ม #5A5F66 สีเดียวทั้งตัว ให้เห็นจุดที่มือสัมผัสชัด
FURN = (0.835, 0.847, 0.863)       # เตียง/เก้าอี้ เทาอ่อน #D5D8DC (ไม่ชนสีผิว/กางเกง)
EX_MODEL = 'asian-male-young'          # ผู้ตรวจชาย (ต้นขอ: หุ่นหญิงสีเดียวดูไม่เหมาะ)
FURN_BASE = (0.74, 0.75, 0.77)

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
    vb = (h.rest @ Rw.T)[np.unique(h.tri)]
    return Rw, np.array([0, BED_H - vb[:, 1].min() - 0.15, -vb[:, 2].mean()])


def standing_world(h, yaw, x, z):
    Rw = rot('Y', yaw)
    return Rw, np.array([x, -(h.rest[np.unique(h.tri)] @ Rw.T)[:, 1].min(), z])


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
    ex = human(EX_MODEL)
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


def plinth(seat_h, front_z, x0=-2.6, x1=9.4):
    """เตียงตรวจให้นั่งขอบเตียง หันหน้าไป +Z ขอบเตียงอยู่ที่ front_z ผู้ป่วยนั่งค่อนไปทางปลายเตียง x0 (เหลือที่ให้ผู้ตรวจยืนฝั่ง -X)"""
    cx, w = (x0 + x1) / 2, x1 - x0
    return [((w, 0.8, 6.0), (cx, seat_h - 0.4, front_z - 3.0), FURN),
            ((w - 1.0, seat_h - 0.8, 5.4), (cx, (seat_h - 0.8) / 2, front_z - 3.0), FURN_BASE)]


def sit_world(h, r, yaw=0, x=0, z=0, seat_h=None):
    """นั่ง: ถ้าไม่ระบุ seat_h ให้เท้าแตะพื้น (เก้าอี้) ถ้าระบุ = นั่งขอบเตียงสูง seat_h ขาห้อย คืน Rw, t, ความสูงที่นั่ง (dm)"""
    Rw = rot('Y', yaw)
    v = (h.pose(r) @ Rw.T)[np.unique(h.tri)]                       # เฉพาะ vertex ของผิว (ไม่เอา helper)
    t = np.array([x, -v[:, 1].min(), z]); v = v + t
    G = skel_G(h, r)
    hipL = jp(h, G, 'upperleg01.L') @ Rw.T + t; hipR = jp(h, G, 'upperleg01.R') @ Rw.T + t
    hy = (hipL[1] + hipR[1]) / 2
    near = (np.abs(v[:, 0] - (hipL[0] + hipR[0]) / 2) < 1.5) & (v[:, 1] < hy) & (np.abs(v[:, 2] - hipL[2]) < 1.2)
    seat = v[near, 1].min()
    if seat_h is not None:
        t = t + [0, seat_h - seat, 0]; seat = seat_h
    return Rw, t, seat


def scene_cxrot():
    """หมุนคอไปขวา (active rotation ROM) นั่งเก้าอี้ ผู้ตรวจยืนหลัง มือ 2 ข้างกดไหล่ไว้กันลำตัวหมุนตาม"""
    pt = human('asian-male-old')
    r = {'upperleg01.L': R(('X', -90)), 'upperleg01.R': R(('X', -90)),
         'lowerleg01.L': R(('X', 90)), 'lowerleg01.R': R(('X', 90))}
    for b, a in (('neck01', -12), ('neck02', -12), ('neck03', -16), ('head', -30)):   # รวม 70 องศาไปขวา
        r[b] = R(('Y', a))
    Rw, t, seat = sit_world(pt, r)
    G = skel_G(pt, r)
    W = lambda b, e='head': jp(pt, G, b, e) @ Rw.T + t
    for side in 'LR':                                                   # มือวางบนต้นขา
        hp, kn = W('upperleg01.' + side), W('lowerleg01.' + side)
        arm, e, _ = solve_arm2(pt, r, side, [('wrist.*', 'head', hp + (kn - hp) * 0.7 + [0, 0.9, 0], 1),
                                             ('finger3-3.*', 'tail', hp + (kn - hp) * 1.15 + [0, 0.6, 0], 1)], Rw, t)
        r.update(arm); print('cxrot pt hand %s %.2f' % (side, e))
    G = skel_G(pt, r)
    ex = human(EX_MODEL)
    er = ARMS_DOWN()
    back = W('spine03')[2]
    Ex, tx = standing_world(ex, 0, 0, back - 3.0)
    Gx = skel_G(ex, er)
    for side, sg in (('L', 1), ('R', -1)):
        top = W('upperarm01.' + side) + np.array([-sg * 0.2, 0.85, 0])        # บนไหล่ (acromion)
        sh = jp(ex, Gx, 'upperarm01.' + side) @ Ex.T + tx
        arm, e, _ = solve_arm2(ex, er, side, [('wrist.*', 'head', top + np.array([0, 0.35, -1.2]), 1),
                                              ('finger3-3.*', 'tail', top + np.array([0, 0.05, 0.6]), 1),
                                              ('lowerarm01.*', 'head', sh + np.array([sg * 1.0, -2.0, 0.4]), 0.4)], Ex, tx)
        er.update(arm); print('cxrot hand %s error %.2f' % (side, e))
    vp = pt.pose(r) @ Rw.T + t
    ve = ex.pose(er) @ Ex.T + tx
    foc = ((W('neck01') + W('head')) / 2 + [0, -0.6, 0]) / 10      # โฟกัสคอ-ไหล่ (เมตร)
    foc = foc.tolist()
    focus = (W('neck01') + W('upperarm01.L') / 2 + W('upperarm01.R') / 2) / 2 + [0, 0.6, 0]   # คอ-ไหล่
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), focus=focus.tolist(), focus_size=5.0,
                boxes=[((4.2, seat, 4.0), (0, seat / 2, W('upperleg01.L')[2] + 0.6), FURN)],
                cams=[('front', (0.45, 0.3, 1)), ('right', (-1.0, 0.35, 0.45))])


GONIO = (0.95, 0.72, 0.15)        # โกนิโอมิเตอร์สีเหลืองอำพัน


def gonio(c, d1, d2, lat, L=2.2):
    """โกนิโอมิเตอร์: จุดหมุนที่ c แขน 2 ข้างตามทิศ d1, d2 แป้นกลมตั้งฉาก lat"""
    u = lambda d: np.asarray(d, float) / np.linalg.norm(d)
    c = np.asarray(c, float); lat = u(lat)
    return [(c.tolist(), (c + u(d1) * L).tolist(), 0.09, GONIO), (c.tolist(), (c + u(d2) * L).tolist(), 0.09, GONIO),
            ((c - lat * 0.03).tolist(), (c + lat * 0.06).tolist(), 0.42, GONIO)]


def hands(ex, er, Ex, tx, specs, tag):
    """วางมือผู้ตรวจ specs: [(side, จุดข้อมือ, จุดปลายนิ้วกลาง, เป้าศอก|None)] เป้าศอกเป็นพิกัดในกรอบตัวผู้ตรวจ (ขวา/ซ้าย, ขึ้น, หน้า) จากไหล่"""
    for side, wp, tp, eo in specs:
        sg = 1 if side == 'L' else -1
        sh = jp(ex, skel_G(ex, er), 'upperarm01.' + side) @ Ex.T + tx
        eo = eo if eo is not None else (0.9, -2.0, 0.5)
        elb = sh + Ex @ np.array([sg * eo[0], eo[1], eo[2]])
        tg = [('wrist.*', 'head', wp, 1), ('finger3-3.*', 'tail', tp, 1), ('lowerarm01.*', 'head', elb, 0.3)]
        arm, e, _ = solve_arm2(ex, er, side, tg, Ex, tx)
        er.update(arm); print('%s hand %s error %.2f' % (tag, side, e))
    return er


def seated(seat=None):
    return {'upperleg01.L': R(('X', -90)), 'upperleg01.R': R(('X', -90)),
            'lowerleg01.L': R(('X', 90)), 'lowerleg01.R': R(('X', 90))}


def scene_sher():
    """ER ไหล่ขวา แขนแนบลำตัว ศอกงอ 90 (active ROM) ผู้ตรวจยืนเฉียงหลัง-ขวา มือหนึ่งประคองศอกแนบลำตัว อีกมือถือโกนิโอใต้ศอก"""
    pt = human('asian-male-old')
    r = seated(); r['lowerleg01.L'] = R(('X', 75)); r['lowerleg01.R'] = R(('X', 75))
    Rw, t, seat = sit_world(pt, r, seat_h=BED_H)                       # นั่งขอบเตียง ขาห้อย
    G = skel_G(pt, r); W = lambda b, e='head': jp(pt, G, b, e) @ Rw.T + t
    hp, kn = W('upperleg01.L'), W('lowerleg01.L')
    arm, e, _ = solve_arm2(pt, r, 'L', [('wrist.*', 'head', hp + (kn - hp) * 0.7 + [0, 0.9, 0], 1),
                                        ('finger3-3.*', 'tail', hp + (kn - hp) * 1.15 + [0, 0.6, 0], 1)], Rw, t)
    r.update(arm)
    sh = W('upperarm01.R'); el = sh + [-0.1, -2.2, 0.1]
    d = np.array([-np.sin(np.radians(60)), 0, np.cos(np.radians(60))])        # ปลายแขนหมุนออก 60 องศา
    arm, e, _ = solve_arm2(pt, r, 'R', [('lowerarm01.*', 'head', el, 1), ('wrist.*', 'head', el + d * 2.1, 1),
                                        ('finger3-3.*', 'tail', el + d * 3.9, 0.5)], Rw, t)
    r.update(arm); print('sher pt arm error %.2f' % e)
    G = skel_G(pt, r)
    elbow, wrist = W('lowerarm01.R'), W('wrist.R')
    ex = human(EX_MODEL); er = ARMS_DOWN()
    Ex, tx = standing_world(ex, 70, elbow[0] - 2.9, elbow[2] - 1.4)          # ยืนข้างขวาค่อนหลัง หันเข้าหาศอกผู้ป่วย
    below = elbow + [0, -0.55, 0]
    er = hands(ex, er, Ex, tx, [
        ('L', elbow + [-0.8, 1.5, -0.3], elbow + [-0.55, 0.0, 0.2], None),         # มือซ้ายกดศอกด้านนอก ให้แนบลำตัว
        ('R', below + [-1.0, -0.4, -1.0], below + [-0.2, -0.3, 0.3], None)], 'sher')    # มือขวาจับแป้นโกนิโอใต้ศอก   # มือขวาถือโกนิโอใต้ศอก
    rods = gonio(below, wrist - elbow, [0, 0, 1], [0, 1, 0])               # แกนหมุนใต้ศอก แขนหนึ่งชี้หน้า อีกแขนตามปลายแขน
    vp = pt.pose(r) @ Rw.T + t; ve = ex.pose(er) @ Ex.T + tx
    focus = (elbow + wrist) / 2 + [0, 0.5, 0]
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), rods=rods, focus=focus.tolist(), focus_size=6.0,
                boxes=plinth(BED_H, W('lowerleg01.L')[2] - 0.2),
                cams=[('front', (-0.35, 0.45, 1)), ('top', (-0.1, 1.0, 0.35))])


def sit_plinth_patient(rest_hand_L=True, rest_hand_R=True):
    """ผู้ป่วยนั่งขอบเตียง ขาห้อย มือวางบนต้นขา"""
    pt = human('asian-male-old')
    r = seated(); r['lowerleg01.L'] = R(('X', 75)); r['lowerleg01.R'] = R(('X', 75))
    Rw, t, seat = sit_world(pt, r, seat_h=BED_H)
    W = lambda b, e='head': jp(pt, skel_G(pt, r), b, e) @ Rw.T + t
    for side, on in (('L', rest_hand_L), ('R', rest_hand_R)):
        if not on: continue
        hp, kn = W('upperleg01.' + side), W('lowerleg01.' + side)
        arm, e, _ = solve_arm2(pt, r, side, [('wrist.*', 'head', hp + (kn - hp) * 0.7 + [0, 0.9, 0], 1),
                                             ('finger3-3.*', 'tail', hp + (kn - hp) * 1.15 + [0, 0.6, 0], 1)], Rw, t)
        r.update(arm)
    return pt, r, Rw, t, W


def table(cx, cz, w, d, top_y):
    """โต๊ะ: แผ่นบน + ขา 4 ขา (rod) ให้เข่าคนนั่งสอดใต้โต๊ะได้"""
    box = [((w, 0.4, d), (cx, top_y - 0.2, cz), FURN)]
    legs = [([cx + sx * (w / 2 - 0.3), 0, cz + sz * (d / 2 - 0.3)], [cx + sx * (w / 2 - 0.3), top_y - 0.4, cz + sz * (d / 2 - 0.3)], 0.18, FURN_BASE)
            for sx in (-1, 1) for sz in (-1, 1)]
    return box, legs


def scene_apb():
    """MMT APB มือขวา: นั่งโต๊ะ หงายมือวางบนโต๊ะ กางนิ้วโป้งตั้งฉากฝ่ามือ ผู้ตรวจนั่งตรงข้าม นิ้วชี้ต้านที่ข้อแรกนิ้วโป้ง อีกมือกดข้อมือไว้"""
    pt = human('asian-male-old'); r = seated()
    Rw, t, seat = sit_world(pt, r)
    W = lambda b, e='head': jp(pt, skel_G(pt, r), b, e) @ Rw.T + t
    hpL = W('upperleg01.L'); arm, e, _ = solve_arm2(pt, r, 'L', [('wrist.*', 'head', hpL + [0.3, 1.0, 2.2], 1),
                                                               ('finger3-3.*', 'tail', hpL + [0.1, 0.8, 3.9], 1)], Rw, t); r.update(arm)
    sh = W('upperarm01.R'); top_y = sh[1] - 2.3                       # โต๊ะสูงระดับศอก
    tz0 = sh[2] + 0.6; td, tw = 4.5, 6.0
    el = np.array([sh[0] - 0.1, top_y + 0.35, sh[2] + 1.0]); w0 = np.array([sh[0] - 0.4, top_y + 0.3, sh[2] + 3.1])
    arm, e, _ = solve_arm2(pt, r, 'R', [('lowerarm01.*', 'head', el, 1), ('wrist.*', 'head', w0, 1),
                                        ('finger3-3.*', 'tail', w0 + [0.15, 0.05, 1.8], 1),
                                        ], Rw, t)
    r.update(arm); print('apb pt arm %.2f' % e)
    sup, e = solve_bones(pt, r, ['lowerarm01.R', 'wrist.R'], [('wrist.R', 'head', w0, 1), ('finger3-3.R', 'tail', w0 + [0.15, 0.05, 1.6], 1),
                                                              ('finger1-1.R', 'head', w0 + [-0.55, 0.1, 0.35], 2)], Rw, t, lim=130)
    r.update(sup); print('apb supinate %.2f' % e)                          # หงายมือ: บิดปลายแขนรอบแกน
    base = W('finger1-1.R')
    th, e = solve_bones(pt, r, ['metacarpal1.R', 'finger1-1.R'], [('finger1-3.R', 'tail', base + [-0.2, 1.4, 0.3], 1)], Rw, t)
    r.update(th); print('apb thumb %.2f' % e)
    prox = (W('finger1-2.R') + W('finger1-2.R', 'tail')) / 2
    tcz = tz0 + td / 2
    ex = human(EX_MODEL); xr = seated(); xr['spine03'] = R(('X', 10)); xr['spine02'] = R(('X', 6))
    Ex, tx, eseat = sit_world(ex, xr, yaw=90, x=0, z=0)
    XW = lambda b: jp(ex, skel_G(ex, xr), b) @ Ex.T + tx
    hipc = (XW('upperleg01.L') + XW('upperleg01.R')) / 2
    tx = tx + [(w0[0] - 4.3) - hipc[0], 0, w0[2] + 0.2 - hipc[2]]       # นั่งด้านขวาของผู้ป่วย หันเข้าหามือ เข่าสอดใต้โต๊ะ
    hipc = (XW('upperleg01.L') + XW('upperleg01.R')) / 2
    wr = W('wrist.R')
    xr = hands(ex, xr, Ex, tx, [
        ('R', prox + [-1.6, 0.5, 0.5], prox + [-0.15, 0.05, 0.0], None),          # ปลายนิ้วแตะด้านนอกข้อแรกนิ้วโป้ง ดันเข้าหาฝ่ามือ
        ('L', wr + [-1.2, 0.7, -1.0], wr + [0.3, 0.35, -0.1], None)], 'apb')      # กดข้อมือผู้ป่วยไว้กับโต๊ะ
    vp = pt.pose(r) @ Rw.T + t; ve = ex.pose(xr) @ Ex.T + tx
    tb, legs = table(w0[0] - 1.6, tcz, 7.5, td, top_y)
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), rods=legs, focus=(wr + [0, 0.6, 0.8]).tolist(), focus_size=4.0,
                boxes=tb + [((4.2, seat, 4.0), (0, seat / 2, W('upperleg01.L')[2] + 0.6), FURN),
                            ((4.0, eseat, 4.2), (hipc[0] + 0.6, eseat / 2, hipc[2]), FURN)],
                cams=[('front', (-0.1, 0.55, 1)), ('top', (-0.15, 1.0, 0.45))])


def bed_patient(prone=False, feet_off=0.0):
    """ผู้ป่วยนอนบนเตียง (หงาย/คว่ำ) feet_off = ให้เท้ายื่นพ้นปลายเตียง (dm)"""
    pt = human('asian-male-old')
    Rw = (rot('Y', 180) @ rot('X', 90)) if prone else rot('X', -90)
    vb = (pt.rest @ Rw.T)[np.unique(pt.tri)]
    lift = (vb[:, 1].min() if not prone else vb[:, 1].min()) - 0.15
    t = np.array([0, BED_H - lift, 0.0])
    zf = vb[:, 2].max()                                                 # ปลายเท้า
    t[2] = BED[2] / 2 - zf + feet_off - 0.6
    return pt, Rw, t


def scene_kflex():
    """งอเข่าขวาเต็มที่ นอนหงาย ลากส้นเท้าเข้าหาก้น ผู้ตรวจวัดโกนิโอด้านนอกเข่า"""
    pt, Rw, t = bed_patient()
    r = ARMS_SIDE(); r['upperleg01.R'] = R(('X', -65)); r['lowerleg01.R'] = R(('X', 125)); r['foot.R'] = R(('X', 20))
    G = skel_G(pt, r); W = lambda b, e='head': jp(pt, G, b, e) @ Rw.T + t
    hip, knee, ank = W('upperleg01.R'), W('lowerleg01.R'), W('foot.R')
    t[1] += 0  # ส้นเท้าวางบนเตียงโดยประมาณ
    lat = np.array([-0.75, 0, 0])
    c = knee + lat
    rods = gonio(c, hip - knee, ank - knee, [1, 0, 0], L=2.6)
    ex = human(EX_MODEL); er = ARMS_DOWN(); er['spine03'] = R(('X', 12)); er['spine02'] = R(('X', 8))
    Ex, tx = standing_world(ex, 140, -(BED[0] / 2 + 1.3), knee[2] + 3.6)   # ยืนข้างขวาค่อนไปทางปลายเท้า ไม่บังเข่า
    u = lambda v: v / np.linalg.norm(v)
    a1 = c + u(hip - knee) * 1.6; a2 = c + u(ank - knee) * 1.6
    er = hands(ex, er, Ex, tx, [('L', a1 + [-1.2, 0.3, -0.6], a1 + [0.0, 0.0, 0.3], None),
                                ('R', a2 + [-1.2, 0.3, 0.5], a2 + [0.0, 0.0, -0.3], None)], 'kflex')
    vp = pt.pose(r) @ Rw.T + t; ve = ex.pose(er) @ Ex.T + tx
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), bed=True, rods=rods,
                focus=(knee + [0, -0.5, 0]).tolist(), focus_size=7.5,
                cams=[('side', (-1, 0.55, -0.55)), ('top', (-0.35, 1.0, -0.7))])


def scene_kext():
    """เหยียดเข่าขวาเต็มที่ นอนหงาย หนุนผ้าม้วนใต้ข้อเท้า ผู้ตรวจวัดโกนิโอด้านนอกเข่า"""
    pt, Rw, t = bed_patient()
    r = ARMS_SIDE()
    G = skel_G(pt, r); W = lambda b, e='head': jp(pt, G, b, e) @ Rw.T + t
    knee0 = W('lowerleg01.R'); ank0 = W('foot.R')
    r['upperleg01.R'] = R(('X', -6)); r['lowerleg01.R'] = R(('X', 6))   # ยกส้นขึ้นบนผ้าม้วน เข่าเหยียด
    G = skel_G(pt, r)
    hip, knee, ank = W('upperleg01.R'), W('lowerleg01.R'), W('foot.R')
    roll = ([-2.6, BED_H + 0.45, ank[2] - 0.6], [2.6, BED_H + 0.45, ank[2] - 0.6], 0.55, (0.62, 0.72, 0.85))
    c = knee + [-0.75, 0, 0]
    rods = gonio(c, hip - knee, ank - knee, [1, 0, 0], L=2.6) + [roll]
    ex = human(EX_MODEL); er = ARMS_DOWN(); er['spine03'] = R(('X', 14)); er['spine02'] = R(('X', 10))
    Ex, tx = standing_world(ex, 140, -(BED[0] / 2 + 1.3), knee[2] + 3.6)
    u = lambda v: v / np.linalg.norm(v)
    a1 = c + u(hip - knee) * 1.6; a2 = c + u(ank - knee) * 1.6
    er = hands(ex, er, Ex, tx, [('L', a1 + [-1.2, 0.5, -0.3], a1 + [0.0, 0.1, 0.3], None),
                                ('R', a2 + [-1.2, 0.5, 0.3], a2 + [0.0, 0.1, -0.3], None)], 'kext')
    vp = pt.pose(r) @ Rw.T + t; ve = ex.pose(er) @ Ex.T + tx
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), bed=True, rods=rods,
                focus=knee.tolist(), focus_size=7.5,
                cams=[('side', (-1, 0.55, -0.75)), ('top', (-0.35, 1.0, -0.7))])


def scene_quad():
    """MMT quadriceps ขวา: นั่งขอบเตียง เหยียดเข่าเกือบสุด ผู้ตรวจกดเหนือข้อเท้าด้านหน้าลงล่าง อีกมือรองใต้ต้นขาส่วนปลาย"""
    pt = human('asian-male-old')
    r = seated(); r['lowerleg01.L'] = R(('X', 75)); r['lowerleg01.R'] = R(('X', 10))
    Rw, t, seat = sit_world(pt, r, seat_h=BED_H)
    W = lambda b, e='head': jp(pt, skel_G(pt, r), b, e) @ Rw.T + t
    for side in 'LR':                                                   # มือวางข้างลำตัวบนเตียง
        hp = W('upperleg01.' + side); sg = 1 if side == 'L' else -1
        arm, e, _ = solve_arm2(pt, r, side, [('wrist.*', 'head', hp + [sg * 2.1, -0.3, -0.6], 1),
                                             ('finger3-3.*', 'tail', hp + [sg * 2.3, -0.75, 0.8], 1)], Rw, t)
        r.update(arm)
    knee, ank = W('lowerleg01.R'), W('foot.R')
    ld = (ank - knee) / np.linalg.norm(ank - knee)
    up = np.cross([1, 0, 0], ld); up = up if up[1] > 0 else -up        # ด้านหน้าหน้าแข้ง
    push = ank - ld * 0.6 + up * 0.55
    under = knee - [0, 0.0, 0.9] + [0, -0.75, 0]
    ex = human(EX_MODEL); er = ARMS_DOWN(); er['spine03'] = R(('X', 10)); er['spine02'] = R(('X', 6))
    Ex, tx = standing_world(ex, 90, knee[0] - 3.2, knee[2] + 0.6)        # ยืนข้างขวา หันเข้าหาผู้ป่วย
    er = hands(ex, er, Ex, tx, [('R', push + [-1.3, 0.6, -0.2], push + [0.4, 0.05, 0.1], None),     # กดหน้าแข้งเหนือข้อเท้า
                                ('L', under + [-1.5, -0.4, -0.3], under + [0.2, 0.0, 0.1], None)], 'quad')  # รองใต้ต้นขาเหนือเข่า
    vp = pt.pose(r) @ Rw.T + t; ve = ex.pose(er) @ Ex.T + tx
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), focus=((knee + ank) / 2).tolist(), focus_size=7.5,
                boxes=plinth(BED_H, W('lowerleg01.L')[2] - 0.2),
                cams=[('front', (-0.15, 0.35, 1)), ('34', (-0.75, 0.45, 1))])


def scene_k2w():
    """knee-to-wall ขวา: ยืนก้าวขาขวาหน้าชิดผนัง ส้นเท้าติดพื้น ดันเข่าแตะผนัง วัดระยะนิ้วโป้งเท้าถึงผนัง"""
    pt = human('asian-male-old')
    r = {'upperleg01.R': R(('X', -18)), 'lowerleg01.R': R(('X', 68)),
         'upperleg01.L': R(('X', 18)), 'lowerleg01.L': R(('X', 12))}
    Rw = np.eye(3); t = np.zeros(3)
    W = lambda b, e='head': jp(pt, skel_G(pt, r), b, e) @ Rw.T + t
    # ข้อเท้าขวา: ฝ่าเท้าราบกับพื้น (หาเอง) / ขาซ้ายหลัง: ส้นยก ปลายเท้าแตะพื้น
    hip = W('upperleg01.R'); ank = W('foot.R')
    fr, e1 = solve_bones(pt, r, ['foot.R'], [('foot.R', 'tail', ank + [0, -0.75, 1.3], 1)], Rw, t)
    r.update(fr)
    ra = W('foot.R')
    lg, e3 = solve_bones(pt, r, ['upperleg01.L', 'lowerleg01.L'], [('foot.L', 'head', [1.4, ra[1] + 0.45, hip[2] - 4.6], 1)], Rw, t)
    r.update(lg); print('k2w back leg %.2f' % e3)                       # ขาหลัง: ข้อเท้าสูงกว่าเท้าหน้าเล็กน้อย (ส้นยก)
    fl, e2 = solve_bones(pt, r, ['foot.L'], [('foot.L', 'tail', W('foot.L') + [0, -1.1, 0.9], 1)], Rw, t)
    r.update(fl)
    vb = (pt.pose(r) @ Rw.T)[np.unique(pt.tri)]
    t = np.array([0, -vb[:, 1].min(), 0.0])
    kn = W('lowerleg01.R')
    kv = vb + t; wall_z = kv[np.linalg.norm(kv - kn, axis=1) < 1.0][:, 2].max()   # ผนังแตะหน้าเข่าขวา
    for side, sg in (('L', 1), ('R', -1)):                               # มือยันผนัง
        sh = W('upperarm01.' + side)
        arm, e, _ = solve_arm2(pt, r, side, [('wrist.*', 'head', [sh[0] + sg * 0.5, sh[1] - 0.8, wall_z - 0.25], 1),
                                             ('finger3-3.*', 'tail', [sh[0] + sg * 0.6, sh[1] + 0.8, wall_z - 0.2], 1)], Rw, t)
        r.update(arm)
    vb = (pt.pose(r) @ Rw.T + t)[np.unique(pt.tri)]
    toe = vb[(vb[:, 0] < 0) & (vb[:, 1] < 1.2)][:, 2].max()            # ปลายนิ้วโป้งเท้าขวา
    print('k2w toe-wall %.1f cm' % ((wall_z - toe) * 10))
    tape = ([W('foot.R')[0] - 0.9, 0.05, toe], [W('foot.R')[0] - 0.9, 0.05, wall_z - 0.02], 0.1, GONIO)
    vp = pt.pose(r) @ Rw.T + t
    return dict(parts=patient_parts(pt, vp), rods=[tape], focus=[W('foot.R')[0], 2.8, wall_z - 1.2], focus_size=6.5,
                boxes=[((14, 18, 0.6), (0, 9, wall_z + 0.3), FURN)],
                cams=[('side', (-1, 0.15, -0.25)), ('34', (-1, 0.6, -0.9))])


def scene_dfev():
    """Dorsiflexion-eversion test เท้าขวา: นอนหงาย ผู้ตรวจที่ปลายเตียง จับส้นเท้า อีกมือดันฝ่าเท้าให้กระดกขึ้น บิดออก และเหยียดนิ้วเท้า ค้างไว้"""
    pt, Rw, t = bed_patient(feet_off=1.0)
    r = ARMS_SIDE(); r['foot.R'] = R(('X', -18), ('Z', -20))             # กระดกข้อเท้า + บิดออก
    for k in range(1, 6):
        for j in (1, 2):
            b = 'toe%d-%d.R' % (k, j)
            if b in pt.bones: r[b] = R(('X', -30))                        # เหยียดนิ้วเท้า
    W = lambda b, e='head': jp(pt, skel_G(pt, r), b, e) @ Rw.T + t
    ank = W('foot.R'); heel = ank + [0, -0.4, -0.55]
    sole = (W('toe1-1.R') + W('toe5-1.R')) / 2 + [0, 0, 0.5]
    ex = human(EX_MODEL); er = ARMS_DOWN(); er['spine03'] = R(('X', 10)); er['spine02'] = R(('X', 6))
    Ex, tx = standing_world(ex, 180, ank[0] - 0.5, BED[2] / 2 + 3.2)
    er = hands(ex, er, Ex, tx, [('R', heel + [0.5, -0.9, 0.9], heel + [-0.2, -0.2, -0.4], None),       # อุ้งมือรองส้นเท้า
                                ('L', sole + [-0.6, -0.9, 0.6], sole + [0.3, 0.9, -0.1], None)], 'dfev')  # ฝ่ามือดันฝ่าเท้า/นิ้วเท้า
    vp = pt.pose(r) @ Rw.T + t; ve = ex.pose(er) @ Ex.T + tx
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), bed=True, focus=(ank + [0, 0.3, 0.6]).tolist(), focus_size=5.0,
                cams=[('side', (-1, 0.45, 0.2)), ('34', (-0.9, 0.6, -0.5))])


def settle(pt, r, Rw, t):
    """วางตัวให้แนบผิวเตียงจริงหลังจัดท่า (ใช้เฉพาะส่วนที่อยู่บนเตียง)"""
    v = (pt.pose(r) @ Rw.T + t)[np.unique(pt.tri)]
    on = (np.abs(v[:, 2]) < BED[2] / 2 - 0.3) & (np.abs(v[:, 0]) < BED[0] / 2)
    return t + [0, BED_H + 0.05 - v[on, 1].min(), 0]


def scene_heelpad():
    """คลำ heel fat pad ขวา: นอนคว่ำ เท้ายื่นพ้นปลายเตียง ผู้ตรวจกดนิ้วโป้งกลางส้นเท้า อีกมือจับข้อเท้า"""
    pt, Rw, t = bed_patient(prone=True, feet_off=1.6)
    r = {'upperarm01.L': R(('Z', -44), ('X', 12)), 'upperarm01.R': R(('Z', 44), ('X', 12))}; t = settle(pt, r, Rw, t)
    W = lambda b, e='head': jp(pt, skel_G(pt, r), b, e) @ Rw.T + t
    ank = W('foot.R')
    vb = (pt.pose(r) @ Rw.T + t)[np.unique(pt.tri)]
    near = np.linalg.norm(vb - ank, axis=1) < 1.6
    heel = vb[near][np.argmax(vb[near][:, 1] - 0.6 * vb[near][:, 2])]   # ผิวส้นเท้าด้านฝ่าเท้า (หันขึ้น/ไปปลายเตียง)
    ex = human(EX_MODEL); er = ARMS_DOWN(); er['spine03'] = R(('X', 10)); er['spine02'] = R(('X', 6))
    Ex, tx = standing_world(ex, 180, ank[0] - 0.4, BED[2] / 2 + 3.0)
    er = hands(ex, er, Ex, tx, [('R', heel + [0.2, 0.3, 1.6], heel + [0.1, 0.1, 0.3], None),
                                ('L', ank + [1.2, 0.6, 0.8], ank + [0.4, 0.5, -0.6], None)], 'heelpad')
    # นิ้วโป้งขวาแตะกลางส้น
    th, e = solve_bones(ex, er, ['metacarpal1.R', 'finger1-1.R'], [('finger1-3.R', 'tail', heel + [0, 0.05, 0.05], 1)], Ex, tx)
    er.update(th); print('heelpad thumb %.2f' % e)
    vp = pt.pose(r) @ Rw.T + t; ve = ex.pose(er) @ Ex.T + tx
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), bed=True, focus=heel.tolist(), focus_size=5.0,
                cams=[('side', (-1, 0.5, 0.25)), ('34', (-0.9, 0.7, -0.5))])


def scene_adt():
    """Anterior drawer ข้อเท้าขวา: นอนหงาย เท้ายื่นพ้นปลายเตียง ข้อเท้าจิกลง ~20 องศา มือซ้ายกดหน้าแข้งส่วนปลายไว้ มือขวาอุ้งส้นเท้าดึงขึ้น (ไปหน้า)"""
    pt, Rw, t = bed_patient(feet_off=1.4)
    r = ARMS_SIDE(); r['foot.R'] = R(('X', 15))
    W = lambda b, e='head': jp(pt, skel_G(pt, r), b, e) @ Rw.T + t
    ank = W('foot.R'); tib = W('lowerleg01.R') * 0.25 + ank * 0.75 + [0, 0.55, 0]
    heel = ank + [0, -0.55, -0.3]
    ex = human(EX_MODEL); er = ARMS_DOWN(); er['spine03'] = R(('X', 12)); er['spine02'] = R(('X', 8))
    Ex, tx = standing_world(ex, 180, ank[0] - 0.6, BED[2] / 2 + 3.0)
    er = hands(ex, er, Ex, tx, [('L', tib + [0.9, 0.6, 1.0], tib + [-0.3, 0.1, -0.4], None),        # กดหน้าแข้งเหนือข้อเท้า
                                ('R', heel + [-0.6, -0.8, 1.2], heel + [0.2, -0.2, -0.5], None)], 'adt')  # อุ้งส้นเท้า ดึงขึ้น
    arrow = ([heel[0] - 1.6, heel[1] - 0.2, heel[2]], [heel[0] - 1.6, heel[1] + 1.4, heel[2]], 0.1, GONIO)
    vp = pt.pose(r) @ Rw.T + t; ve = ex.pose(er) @ Ex.T + tx
    return dict(parts=patient_parts(pt, vp) + examiner_parts(ex, ve), bed=True, focus=(ank + [0, 0.2, 0]).tolist(), focus_size=5.5,
                cams=[('side', (-1, 0.35, 0.1)), ('34', (-0.9, 0.6, -0.5))])
