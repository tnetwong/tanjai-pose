# tanjai-pose

ข้อมูล MakeHuman 1.3.0 (CC0) ที่ตัดมาเฉพาะส่วนที่ใช้จัดท่าหุ่น สำหรับสกิล makehuman-pose-render
จะได้ไม่ต้องอัปโหลด zip 300 MB ทุกครั้ง

- `mh/base.npz` — 3dobjs/base.npz (coord, fvert, group, fgstr/fgidx)
- `mh/default.mhskel` — rigs/default.mhskel
- `mh/default_weights.mhw` — rigs/default_weights.mhw
- `mh/targets_asian.npz` — targets.npz เฉพาะ targets/macrodetails/asian-* (key เดิม: `<name>.index`, `<name>.vector`)

ใช้: `git clone --depth 1 https://github.com/tnetwong/tanjai-pose` แล้วชี้ path ไปที่ `mh/`
