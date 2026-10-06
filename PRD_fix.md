# PRD — Fix Jitter pada Face/Object Tracking Bee dengan One Euro Filter

## 1. Tujuan

Hilangkan jitter pada virtual-camera/reframing Bee tanpa membuat kamera terasa lambat saat wajah atau objek bergerak cepat.

Target:
- Ganti smoothing EMA pada crop path dengan One Euro Filter.
- Pertahankan sampling detection yang hemat CPU.
- Pertahankan interpolasi crop pada full render FPS.
- Tambahkan dead-zone untuk micro-jitter.
- Tangani detection dropout dan shot cut.
- Pertahankan kompatibilitas crop plan JSON dan FFmpeg renderer.
- Tambahkan test kuantitatif.

Repository:
- Bee: https://github.com/SYTE001/Bee
- One Euro reference: https://github.com/jaantollander/OneEuroFilter

## 2. Kondisi Bee Saat Ini

Bee memakai backend Python dengan pipeline video dan service reframing.

File utama:
- backend/app/services/reframe.py
- backend/app/pipelines/video_pipeline.py
- backend/app/core/config.py
- backend/app/services/
- backend/tests/
- backend/requirements-ml.txt

Temuan penting pada reframe.py:
- face detection memakai MediaPipe, YuNet, atau Haar fallback;
- detection sampling default 5 FPS;
- crop plan memakai CropSample(t, x, y);
- crop path saat ini dihaluskan dengan EMA;
- FFmpeg melakukan interpolasi x/y antara keyframes;
- crop window w/h tetap agar kompatibel dengan FFmpeg crop;
- tracker dan fallback behavior sudah tersedia.

requirements-ml.txt sudah memiliki MediaPipe dan OpenCV. Tidak diperlukan dependency ML berat baru hanya untuk One Euro.

## 3. Masalah

Detection frame-per-frame/sample menghasilkan noise kecil pada koordinat target. Noise tersebut dapat masuk ke virtual-camera path dan terlihat sebagai jitter.

EMA dengan alpha konstan mengurangi noise, tetapi responsiveness selalu sama. Saat target bergerak cepat, smoothing dapat membuat camera path tertinggal.

Detection dropout dan shot cut juga harus diperlakukan berbeda dari gerakan normal.

## 4. Keputusan Arsitektur

Gunakan One Euro Filter sebagai smoothing temporal utama.

Jangan menambahkan EMA setelah One Euro sebagai default.

Jangan mengganti pipeline dengan Kalman pada fase ini.

Jangan mengganti pipeline Bee dengan AutoFlip.

Tempatkan smoothing di backend, di reframe pipeline, setelah tracked target dipilih dan sebelum CropSample final dibuat.

Pipeline:

Detection
→ tracking/association
→ target selection
→ center_x/center_y
→ One Euro Filter
→ dead-zone
→ crop position + clamp
→ CropSample
→ decimate
→ FFmpeg interpolation
→ render

Frontend tidak boleh memiliki smoothing terpisah. Preview dan render harus menggunakan crop plan yang sama.

## 5. Representasi Sinyal

Untuk single-focus:
- hitung center_x;
- hitung center_y;
- filter center_x dan center_y;
- pertahankan crop_w/crop_h konstan sesuai desain Bee.

Jangan menjadikan x1,y1,x2,y2 sebagai empat sinyal utama yang difilter independen.

Untuk split-screen:
- setiap panel mempunyai filter state sendiri;
- jangan gunakan satu filter global untuk semua speaker.

## 6. One Euro Implementation

Buat modul:
backend/app/services/tracking_smoothing.py

Modul ini bertanggung jawab atas:
- One Euro implementation;
- state initialization;
- timestamp handling;
- reset;
- parameter validation;
- optional dead-zone helper;
- type hints.

Gunakan implementasi upstream sebagai referensi algoritmik:
https://github.com/jaantollander/OneEuroFilter/blob/master/python/one_euro_filter.py

API upstream yang relevan:
OneEuroFilter(t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)

Jangan membuat Bee bergantung pada GitHub saat runtime.

Preferensi: vendorkan implementation minimal ke Bee, lalu tambahkan validation/reset sesuai architecture Bee.

## 7. Parameter Default

Baseline awal:
- detection FPS = 5
- min_cutoff = 1.0
- beta = 0.03
- d_cutoff = 1.0
- dead-zone = 3 px pada source coordinates

Expose melalui sistem configuration yang sudah ada di backend/app/core/config.py.

Konsep variable:
- REFRAME_DETECT_FPS
- REFRAME_ONE_EURO_MIN_CUTOFF
- REFRAME_ONE_EURO_BETA
- REFRAME_ONE_EURO_D_CUTOFF
- REFRAME_DEADZONE_PX
- REFRAME_SMOOTHING

Jangan membuat sistem konfigurasi kedua.

Mode smoothing:
- one_euro
- ema
- none

Default production setelah validasi: one_euro.

EMA lama harus tetap tersedia selama fase regression testing.

## 8. Timestamp

Timestamp harus berasal dari timestamp sample video yang sebenarnya.

Jangan mengasumsikan timestamp hanya berdasarkan frame index × 1/30.

One Euro membutuhkan delta waktu. Tangani:
- t <= previous_t;
- duplicate timestamps;
- non-monotonic timestamps.

Input invalid harus ditolak atau ditangani tanpa menghasilkan NaN/Inf.

## 9. Adaptive Behavior

One Euro harus mempertahankan sifat adaptive:
- target diam → smoothing lebih kuat;
- target bergerak cepat → cutoff meningkat dan filter menjadi lebih responsif.

Jangan menambahkan prediction/velocity model tambahan pada MVP.

## 10. Dead-Zone

Dead-zone diterapkan setelah One Euro.

Aturan:
- jika perpindahan kamera < threshold → pertahankan posisi kamera sebelumnya;
- jika perpindahan >= threshold → update camera position;
- default threshold = 3 px.

Dead-zone tidak boleh membuat kamera tersendat atau menghasilkan snap berulang.

## 11. Shot Cut

Pada scene/shot cut:
- reset One Euro state;
- jangan membawa posisi camera dari shot lama;
- initialize dari target pada shot baru;
- jangan melakukan long pan lintas shot.

Gunakan scene-cut metadata yang sudah ada jika tersedia.

Jika sudden displacement dipakai sebagai fallback, gunakan threshold yang eksplisit dan konservatif agar fast movement normal tidak dianggap cut.

## 12. Detection Dropout

Jangan memasukkan koordinat 0,0 atau invalid ke filter ketika detection hilang.

Aturan:
- dropout pendek → tahan posisi terakhir;
- dropout melewati batas gap existing tracker → gunakan existing fallback behavior;
- detection kembali → reacquire/reset dengan aman sesuai target identity.

Jangan menjadikan missing detection sebagai artificial movement.

## 13. Target Identity

Filter state harus terikat pada target:
- single face → satu state;
- split screen → state per panel/speaker.

Saat active target berganti:
- reset filter state;
- jangan membawa history target lama ke target baru.

Pertahankan tracker existing Bee. Perubahan ini adalah smoothing layer, bukan penggantian tracker.

## 14. Crop Position

Setelah filtered center:
1. gunakan existing _crop_position;
2. hitung x/y;
3. clamp ke source bounds;
4. buat CropSample.

Invariants:
- x >= 0
- y >= 0
- x + crop_w <= source_width
- y + crop_h <= source_height

Pertahankan crop_w/crop_h konstan untuk FFmpeg crop.

## 15. FFmpeg Compatibility

Pertahankan renderer yang sekarang:
- x/y boleh berubah per frame;
- w/h tetap;
- keyframes tetap diinterpolasi linear.

Jangan menjalankan One Euro pada semua render frames.

Detection tetap 5 FPS kecuali benchmark menunjukkan kebutuhan lain.

One Euro dijalankan pada sample trajectory dan renderer tetap menghasilkan gerakan kontinu pada 30/60 FPS.

## 16. Decimation

Urutan wajib:

raw samples
→ tracking
→ One Euro
→ dead-zone
→ clamp
→ CropSample
→ decimate

Jangan melakukan decimation sebelum smoothing.

## 17. Plan Version / Cache

Crop plans di-cache. Jika semantics crop path berubah sehingga plan lama tidak aman dipakai, naikkan PLAN_VERSION.

Jika struktur JSON tidak berubah dan hanya nilai trajectory yang dihitung ulang, tetap evaluasi apakah version bump diperlukan; jangan membiarkan cache lama mempertahankan trajectory EMA setelah default berubah ke One Euro.

## 18. Files yang Diubah

Prioritas:
1. backend/app/services/reframe.py
2. backend/app/services/tracking_smoothing.py
3. backend/app/core/config.py
4. backend/.env.example
5. backend/tests/test_reframe.py atau test terkait yang tersedia
6. backend/requirements-ml.txt hanya jika benar-benar perlu

Jangan ubah frontend kecuali parameter smoothing memang perlu diekspos ke editor.

## 19. Tests

### One Euro
- initialization;
- constant signal;
- noisy stationary signal;
- slow movement;
- fast movement;
- timestamp delta;
- invalid timestamp;
- reset.

### Dead-zone
- 1 px movement dengan threshold 3 px → suppressed;
- movement di threshold → mengikuti definisi operator yang dipilih secara konsisten;
- 10 px movement → update;
- oscillating noise ±1–2 px → tidak menghasilkan camera oscillation.

### Bounding box
- normal;
- edge of frame;
- tiny face;
- changing face size;
- missing detection;
- first detection;
- target switch.

### Crop plan
- timestamps monotonically increasing;
- x/y valid;
- crop window valid;
- JSON serialize/deserialize;
- static plan tetap static;
- dynamic plan tetap dynamic.

### Regression
Bandingkan:
- none;
- EMA baseline;
- One Euro.

Gunakan synthetic tracking sequences dan minimal 5 fixture:
1. static face;
2. slow head movement;
3. walking subject;
4. fast movement;
5. shot cut + temporary dropout.

## 20. Success Metrics

Static shot:
- RMS camera jitter turun >= 60% vs EMA baseline;
- small-direction reversal turun >= 70%;
- tidak ada visible micro-oscillation.

Fast movement:
- median lag tidak bertambah > 2–3 frames vs EMA;
- target tetap dalam safe framing;
- tidak ada oversmoothing yang jelas.

Shot cut:
- tidak ada pan lintas shot;
- state reset;
- first samples shot baru converge tanpa history shot lama.

Correctness:
- 100% generated crop samples berada dalam source bounds;
- 0 NaN/Inf;
- 0 invalid crop plans pada regression suite.

Jangan menyatakan jitter fixed hanya berdasarkan subjektif preview. Harus ada metrik atau visual regression evidence.

## 21. Observability

Saat DEBUG, catat agregat:
- detector sample count;
- valid detection count;
- filter mode;
- One Euro parameters;
- reset count;
- dropout count;
- dead-zone suppressed count;
- raw displacement statistics;
- filtered displacement statistics.

Jangan log setiap sample pada INFO production.

## 22. Performance

Target:
- tidak menambah jumlah detector invocation;
- O(1) filter computation per sample;
- memory overhead konstan;
- tidak menambah pipeline frame buffering secara signifikan;
- tidak menambah dependency berat.

Pada 5 FPS detection, biaya One Euro harus negligible dibanding MediaPipe/OpenCV/FFmpeg.

## 23. Hal yang Tidak Boleh Dilakukan

Jangan:
- smoothing di frontend;
- double smoothing tanpa alasan;
- memasukkan missing detection sebagai 0,0;
- timestamp sintetis jika timestamp actual tersedia;
- satu filter global untuk semua target;
- mengubah crop_w/h per frame pada FFmpeg crop;
- mengganti tracker existing;
- menambahkan Kalman sekaligus One Euro pada MVP;
- menghapus detector fallback;
- merusak cache semantics.

## 24. Implementation Order

1. Inspect seluruh fungsi tracking/reframe pada reframe.py dan pipeline caller.
2. Buat tracking_smoothing.py berdasarkan algoritma upstream.
3. Tambahkan configuration.
4. Bungkus smoothing lama dengan feature flag.
5. Integrasikan One Euro pada trajectory camera.
6. Tambahkan dead-zone.
7. Tambahkan reset untuk scene cut/target switch.
8. Tangani dropout.
9. Pertahankan decimation, CropPlan, serialization, dan FFmpeg interpolation.
10. Tambahkan unit tests.
11. Tambahkan regression benchmark.
12. Jalankan test suite existing.
13. Render smoke test.
14. Bandingkan metrik EMA vs One Euro.
15. Jadikan One Euro default hanya setelah regression pass.

## 25. Definition of Done

- [ ] One Euro tersedia sebagai modul internal.
- [ ] reframe.py menggunakan One Euro melalui feature flag.
- [ ] EMA baseline masih tersedia.
- [ ] Default One Euro dapat diaktifkan tanpa perubahan kode.
- [ ] Dead-zone diterapkan setelah filtering.
- [ ] Shot cut mereset state.
- [ ] Dropout tidak memasukkan koordinat invalid.
- [ ] Target filter state terisolasi.
- [ ] Crop coordinates selalu clamped.
- [ ] Existing renderer tetap kompatibel.
- [ ] Crop plan JSON tetap valid.
- [ ] Tests lulus.
- [ ] Regression metrics memenuhi target.
- [ ] CPU Docker tetap build.
- [ ] GPU Docker tetap build.
- [ ] Tidak ada dependency runtime eksternal baru yang tidak perlu.
- [ ] Cache/PLAN_VERSION ditangani dengan benar.

## 26. Expected Agent Report

Setelah implementasi, agent wajib melaporkan:
- files changed;
- One Euro defaults;
- test count/result;
- RMS jitter EMA vs One Euro;
- reversal count EMA vs One Euro;
- lag comparison;
- dropout/reset behavior;
- smoke render result;
- cache/PLAN_VERSION decision.

Jangan menggunakan klaim "production ready" tanpa test evidence.

## 27. References

Bee:
https://github.com/SYTE001/Bee

One Euro Filter:
https://github.com/jaantollander/OneEuroFilter

Reference Python:
https://github.com/jaantollander/OneEuroFilter/blob/master/python/one_euro_filter.py
