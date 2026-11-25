import sys, json, os, numpy as np
import scenes
from mhpose import write_obj
ROOT = os.path.dirname(os.path.abspath(__file__))

def export(name):
    s = getattr(scenes, 'scene_' + name)()
    d = os.path.join(ROOT, 'godot', 'shots', name); os.makedirs(d, exist_ok=True)
    meta = dict(parts=[], bed=s.get('bed', False), bed_h=scenes.BED_H / 10, bed_size=[x / 10 for x in scenes.BED], cams=s['cams'], furn=list(scenes.FURN), furn_base=list(scenes.FURN_BASE), boxes=[[[a / 10 for a in sz], [a / 10 for a in ps], list(c)] for sz, ps, c in s.get('boxes', [])])
    if 'focus' in s: meta['focus'] = [a / 10 for a in s['focus']]; meta['focus_size'] = s['focus_size'] / 10
    meta['rods'] = [[[a / 10 for a in p], [a / 10 for a in q], rr / 10, list(c)] for p, q, rr, c in s.get('rods', [])]
    allv = []
    for pname, v, tri, mask, col in s['parts']:
        write_obj(os.path.join(d, pname + '.obj'), v, tri, mask)
        meta['parts'].append(dict(name=pname, color=col)); allv.append(v[np.unique(tri[mask])] / 10)
    allv = np.vstack(allv); meta['min'] = allv.min(0).tolist(); meta['max'] = allv.max(0).tolist()
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'))

names = sys.argv[1:]
for n in names: export(n)
json.dump(names, open(os.path.join(ROOT, 'godot', 'shots', 'list.json'), 'w'))
print('exported', names)
