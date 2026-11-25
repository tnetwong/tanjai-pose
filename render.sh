#!/bin/sh
# ใช้: ./render.sh slr [ท่าอื่น...]  -> out/<ท่า>-<มุม>.png (ครอปขอบขาวแล้ว)
set -e
cd "$(dirname "$0")"
G=${GODOT:-$HOME/godot/Godot_v4.6-stable_linux.x86_64}
python3 export.py "$@"
rm -f out/"$1"-*.png; cd godot && timeout 150 $G --headless --import >/dev/null 2>&1 || true
timeout 300 xvfb-run -a -s "-screen 0 900x700x24" $G --rendering-driver opengl3 --resolution 900x700 >/dev/null 2>&1 || true
cd .. && python3 - "$@" <<'PY'
import sys, glob
from PIL import Image, ImageChops
for n in sys.argv[1:]:
    for f in glob.glob('out/%s-*.png' % n):
        if f.endswith('-crop.png'): continue
        im = Image.open(f).convert('RGB'); bg = Image.new('RGB', im.size, (255, 255, 255))
        bb = ImageChops.difference(im, bg).convert('L').point(lambda v: 255 if v > 12 else 0).getbbox()
        if bb: im = im.crop((max(0, bb[0]-20), max(0, bb[1]-20), min(im.width, bb[2]+20), min(im.height, bb[3]+20)))
        im.save(f.replace('.png', '-crop.png')); print(f, im.size)
PY
