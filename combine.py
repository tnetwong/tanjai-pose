"""รวม 2 มุมเป็นภาพเดียว: python3 combine.py <ท่า> <ไฟล์ออก> [crop|zoom]"""
import sys, glob
from PIL import Image
n, out = sys.argv[1], sys.argv[2]; kind = sys.argv[3] if len(sys.argv) > 3 else 'crop'
fs = sorted(glob.glob('out/%s-*-%s.png' % (n, kind)), key=lambda f: ('side' not in f and 'front' not in f, f))
ims = [Image.open(f) for f in fs]; h = max(i.height for i in ims)
c = Image.new('RGB', (sum(i.width for i in ims) + 20 * (len(ims) - 1), h), 'white'); x = 0
for i in ims: c.paste(i, (x, (h - i.height) // 2)); x += i.width + 20
c.save(out)
