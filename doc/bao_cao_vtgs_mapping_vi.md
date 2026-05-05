# Báo Cáo Tóm Tắt: Tích Hợp View-Tied Gaussian Splatting Mapping Vào MASt3R-SLAM

## 1. Mục tiêu tích hợp

Pipeline gốc MASt3R-SLAM có khả năng tracking, chọn keyframe, tối ưu pose và
sinh pointmap dày từ mô hình MASt3R. Tuy nhiên phần map 3D dạng point cloud hoặc
Gaussian tự do có thể gặp các vấn đề:

- nhiều Gaussian bị dư, rời rạc hoặc trôi khỏi hình học thật;
- tối ưu quá nhiều tham số hình học gây tốn bộ nhớ;
- Gaussian có thể bị "floaters", blob hoặc duplicate surface nếu pose/geometry chưa ổn định;
- kết quả PLY phụ thuộc mạnh vào cách khởi tạo scale, opacity, densification và pruning.

Phương pháp mapping mới được tích hợp nhằm thử một hướng khác: **View-Tied
Gaussian Splatting (VTGS)**. Thay vì học tự do vị trí 3D của mỗi Gaussian trong
không gian world, mỗi Gaussian được **gắn với một pixel/điểm trong một keyframe
nguồn**. Vị trí world của Gaussian không phải tham số học trực tiếp, mà được suy
ra từ:

```text
anchor point trong camera nguồn + pose mới nhất của keyframe nguồn
```

Mục tiêu của bản tích hợp hiện tại là:

- giữ nguyên MASt3R-SLAM làm tracking/backend;
- thêm một mapper online mới kiểu VTGS;
- dùng `X_canon` và confidence từ MASt3R làm nguồn depth/geometry;
- xuất được cả native state `.pt` và PLY chuẩn để xem bằng SuperSplat/evaluate.

---

## 2. Cơ sở lý thuyết

### 2.1 3D Gaussian Splatting

3D Gaussian Splatting biểu diễn scene bằng tập các Gaussian trong không gian 3D.
Mỗi Gaussian thường có:

```text
mean        vị trí 3D
scale       kích thước theo các trục
rotation    hướng ellipsoid
opacity     độ trong suốt
color       màu, thường ở dạng SH coefficients
```

Khi render, các Gaussian được chiếu lên mặt phẳng ảnh, alpha-composite theo thứ
tự depth và tạo ra ảnh RGB/depth. Tối ưu 3DGS thường dùng photometric loss giữa
ảnh render và ảnh thật:

```text
L_rgb = |render_rgb - gt_rgb|
```

Trong SLAM, 3DGS hấp dẫn vì có thể tạo bản đồ dày và render đẹp, nhưng cũng khó
ổn định vì số tham số lớn. Nếu pose, depth hoặc scale khởi tạo sai, Gaussian có
thể lan rộng thành blob, duplicate surface hoặc floater.

### 2.2 MASt3R-SLAM cung cấp gì cho mapping

MASt3R-SLAM không chỉ cung cấp pose camera. Với mỗi keyframe, pipeline hiện tại
có:

- ảnh RGB đã resize/crop: `uimg`;
- pointmap trong hệ camera keyframe: `X_canon`;
- confidence map: `C / N`;
- pose camera-to-world: `T_WC`;
- intrinsics `K` nếu chạy calibrated mode.

Do đó mapper không cần tự estimate depth từ ảnh RGB. Nó có thể lấy `X_canon.z`
làm camera-depth và dùng `T_WC.act(X_canon)` để đưa điểm sang world.

### 2.3 Ý tưởng View-Tied Gaussian

Trong Gaussian tự do, `mean` là tham số được học:

```text
mean_world = learnable parameter
```

Trong VTGS, Gaussian được gắn vào một view/keyframe nguồn:

```text
anchor = X_canon[pixel] trong camera nguồn
source_kf = id của keyframe nguồn
mean_world = T_WC(source_kf).act(anchor)
```

Điểm khác biệt quan trọng:

- `anchor` không được tối ưu;
- `mean_world` không được tối ưu trực tiếp;
- khi pose keyframe được backend cập nhật, vị trí world của Gaussian tự thay đổi
  theo pose mới;
- mapper chỉ học các tham số nhẹ hơn: màu, radius đẳng hướng, opacity.

Representation hiện tại:

```text
anchor       fixed point từ X_canon
source_kf    keyframe nguồn
sh_dc        learnable color
log_radius   learnable isotropic radius
opacity      learnable opacity logit
```

Khi render/export, mapper materialize sang format 3DGS chuẩn:

```text
means  = Sim3(kf_pose_data[source_kf]).act(anchor)
scales = exp(log_radius).repeat(3)
quats  = identity quaternion
colors = sh_dc
```

### 2.4 Lợi ích kỳ vọng

VTGS có các lợi ích lý thuyết sau:

- **Ổn định hình học hơn**: Gaussian bám vào pointmap/keyframe thay vì trôi tự do.
- **Ít tham số hình học hơn**: không học mean, rotation, anisotropic scale.
- **Pose-aware**: khi backend cập nhật pose, map có thể materialize lại theo pose mới.
- **Phù hợp online mapping**: có thể chia scene thành các section, chỉ train section hiện tại.
- **Giảm nguy cơ map bị phình/blob do scale/mean tự do**.

Đổi lại, v1 hiện tại cũng có giới hạn:

- ít expressiveness hơn Gaussian tự do;
- chưa học SH degree cao/view-dependent color;
- phụ thuộc chất lượng `X_canon`;
- PLY là snapshot materialized, không còn "view-tied dynamic" sau khi export.

---

## 3. Kiến trúc hệ thống sau tích hợp

### 3.1 Các thành phần chính

```text
main.py
  |
  +-- MASt3R-SLAM frontend
  |     - đọc frame
  |     - chạy MASt3R inference/tracking
  |     - quyết định keyframe
  |
  +-- MASt3R-SLAM backend
  |     - retrieval / factor graph
  |     - global optimization
  |     - cập nhật pose trong SharedKeyframes
  |
  +-- Online mapping worker
        - standard: run_online_gs
        - view_tied: run_online_vtgs
```

Worker được chọn bằng config:

```yaml
gaussian_splat:
  online_enabled: true
  mapper_type: view_tied
```

Nếu `mapper_type: standard`, hệ thống dùng mapper online GS cũ. Nếu
`mapper_type: view_tied`, hệ thống dùng VTGS mapper mới.

### 3.2 Tách tracking và mapping

VTGS mapper là một process riêng. Nó nhận keyframe qua queue, không trực tiếp
can thiệp tracker/backend:

```text
MASt3R-SLAM tracking/backend  --->  gs_queue  --->  run_online_vtgs
```

Điều này giữ nguyên trách nhiệm:

- SLAM chịu trách nhiệm pose và pointmap;
- VTGS chịu trách nhiệm map representation và render optimization;
- hai bên đồng bộ qua `SharedKeyframes` và queue event.

### 3.3 Shared state

Các vùng dữ liệu quan trọng:

- `SharedKeyframes`: lưu keyframe, pose, pointmap, confidence, intrinsics.
- `SharedStates`: trạng thái tracking/backend, pause/terminate, current frame.
- `gs_queue`: truyền event từ main process sang mapping worker.

Event chính:

```text
new_kf      thêm keyframe mới vào mapper
terminate   SLAM kết thúc, sync pose cuối, fine refine và save map
```

---

## 4. Luồng dữ liệu toàn hệ thống

### 4.1 Đọc frame và tracking

Dataset trả về:

```text
timestamp, RGB image
```

`main.py` tạo `Frame`:

```text
frame.uimg
frame.img
frame.img_shape
frame.T_WC
```

Nếu là frame đầu, MASt3R chạy mono inference để khởi tạo:

```text
X_init, C_init = mast3r_inference_mono(model, frame)
frame.update_pointmap(X_init, C_init)
```

Nếu đang tracking, `FrameTracker` estimate pose frame mới so với keyframe hiện
tại, cập nhật pointmap/confidence khi cần, và quyết định có thêm keyframe mới
hay không.

### 4.2 Thêm keyframe vào SharedKeyframes

Khi một frame được chọn làm keyframe:

```text
keyframes.append(frame)
states.queue_global_optimization(kf_idx)
```

Keyframe lúc này chứa:

```text
uimg
X_canon
C, N
T_WC
K nếu calibrated
```

Backend sẽ tiếp tục tối ưu pose trong `SharedKeyframes`.

### 4.3 Tạo snapshot cho mapper

Ngay sau khi append keyframe, main process gửi snapshot:

```python
gs_queue.put(make_keyframe_snapshot(kf_idx, keyframes[kf_idx], use_calib))
```

Snapshot được clone sang CPU để an toàn khi đi qua multiprocessing queue.

Nội dung snapshot:

```text
uimg       ảnh RGB target
X_canon    pointmap H*W x 3 trong camera nguồn
C          confidence accumulator
N          số lần update confidence
T_WC_data  pose keyframe tại thời điểm publish
K          intrinsics nếu có
img_shape  H, W
```

### 4.4 VTGS worker nhận keyframe

Worker `run_online_vtgs` nhận event `new_kf` và gọi:

```python
mapper.insert_keyframe(event)
```

Trong `insert_keyframe`, mapper thực hiện:

1. Chuyển snapshot sang tensor CUDA.
2. Nếu calibrated mode, constrain `X_canon` theo camera ray từ `K`.
3. Chuẩn hóa confidence:

   ```text
   conf_norm = (C / N) / max(C / N)
   ```

4. Tạo depth map:

   ```text
   depth = X_canon.z * Sim3_scale
   ```

5. Tạo view matrix world-to-camera từ `T_WC`.
6. Lưu target camera data vào section.
7. Chọn mask pixel để insert Gaussian.
8. Tạo VT Gaussian params.
9. Rebuild optimizer cho section hiện tại.

### 4.5 Chia section

Map được chia thành các section:

```yaml
vt_section_size: 8
```

Cứ mỗi 8 keyframe sẽ bắt đầu một section mới. Mục đích:

- giới hạn phạm vi optimize;
- tránh phải backprop qua toàn map;
- giữ memory/training ổn định hơn;
- phù hợp với scene dài hoặc online mapping.

Mỗi `VTSection` chứa:

```text
kf_indices
viewmats
gt_images
depth_maps
conf_maps
Ks_list
anchors
source_kfs
sh_dc
log_radius
opacities
```

### 4.6 Insert Gaussian cho head frame

Frame đầu section gọi là head frame.

Head frame insert tất cả pixel hợp lệ:

```text
conf_norm > c_conf_threshold
X_canon.z > vt_min_depth
```

Số lượng bị giới hạn bởi:

```yaml
vt_head_insert_cap: 120000
```

Nếu vượt cap, giữ các pixel confidence cao nhất.

### 4.7 Insert Gaussian cho regular frame

Frame không phải head sẽ không insert toàn bộ pixel. Thay vào đó mapper render
section hiện tại vào view mới để xem vùng nào đã được cover.

Pixel mới được insert nếu:

```text
alpha < vt_silhouette_thresh
hoặc
relative_depth_error > vt_depth_cover_tol
```

Điều này có nghĩa:

- nếu vùng ảnh chưa có Gaussian nào che phủ, insert thêm;
- nếu có render nhưng depth lệch nhiều so với depth từ `X_canon`, insert thêm;
- nếu vùng đã cover tốt, không duplicate Gaussian.

Cap:

```yaml
vt_regular_insert_cap: 60000
```

### 4.8 Khởi tạo tham số VT Gaussian

Mỗi pixel được insert tạo một Gaussian:

```text
anchor = X_canon[pixel]
source_kf = kf_idx
sh_dc = (rgb - 0.5) / SH_C0
opacity = logit(conf_norm)
log_radius = log(radius)
```

Nếu có intrinsics:

```text
radius = z / mean(fx, fy)
```

Nếu không có intrinsics, radius lấy từ khoảng cách lân cận trong pointmap.

### 4.9 Online training

Sau khi insert keyframe, worker train ngay:

```text
vt_steps_head     cho head frame
vt_steps_regular  cho regular frame
```

Khi không có event mới, worker vẫn train idle:

```text
vt_steps_idle
```

Mỗi training step:

1. Chọn một target camera trong current section.
2. Lấy render context:

   ```text
   current section
   previous section
   best overlapping older section nếu có
   ```

3. Materialize Gaussian sang format renderer.
4. Render RGB/depth/alpha bằng `gsplat.rasterization`.
5. Tính loss:

   ```text
   L = vt_lambda_rgb * L_rgb
     + vt_lambda_ssim * D-SSIM
     + vt_lambda_depth * L_depth
   ```

6. Backprop chỉ vào:

   ```text
   sh_dc
   log_radius
   opacities
   ```

7. Clamp radius/opacity.

Không optimize:

```text
anchor
source pose
world-space mean
quaternion
anisotropic scale
```

### 4.10 Đồng bộ pose khi SLAM backend cập nhật

Trong lúc worker idle, cứ mỗi `vt_sync_every_idle` chu kỳ, mapper gọi:

```python
mapper.sync_poses_from_shared(keyframes)
```

Hàm này đọc pose mới nhất từ `SharedKeyframes` và cập nhật:

```text
kf_pose_data[kf_idx]
viewmats trong section
```

Vì Gaussian được view-tied, pose mới sẽ ảnh hưởng trực tiếp đến vị trí world khi
render/materialize.

### 4.11 Kết thúc SLAM và finalize mapping

Khi dataset kết thúc:

1. Main process đặt mode `TERMINATED`.
2. Backend join xong, tức pose global optimization đã hoàn tất.
3. Main gửi:

   ```python
   gs_queue.put({"type": "terminate"})
   ```

4. VTGS worker:
   - sync final poses;
   - rebuild từ final `SharedKeyframes` nếu bật;
   - chạy fine refinement;
   - save `.pt`;
   - export `.ply`.

Rebuild được bật mặc định:

```yaml
vt_rebuild_from_final_keyframes: true
```

Lý do: snapshot online ban đầu có thể được gửi trước khi backend tối ưu pose và
pointmap ổn định. Rebuild giúp map cuối dùng cùng nguồn keyframe đã tối ưu.

---

## 5. Materialization và xuất map

VTGS native state không phải PLY trực tiếp. Trước khi render hoặc export, mapper
materialize các section:

```text
for each Gaussian:
  T = kf_pose_data[source_kf]
  mean = Sim3(T).act(anchor)
  scale_xyz = exp(log_radius).repeat(3)
  quat = [1, 0, 0, 0]
  opacity = opacity_logit
  color = sh_dc
```

Sau đó gọi helper export PLY chuẩn:

```text
_export_ply(...)
```

File xuất:

```text
<seq_name>_vtgs_online_gs.ply
```

Đây là file dùng để:

- mở bằng SuperSplat;
- chạy `scripts/eval_gs_psnr.py`;
- so sánh trực quan với standard online GS.

File native:

```text
<seq_name>_vtgs.pt
```

dùng để debug section/anchor/params nội bộ, không mở trực tiếp bằng SuperSplat.

---

## 6. So sánh với mapper online GS chuẩn

| Thành phần | Online GS chuẩn | VTGS mapper |
|---|---|---|
| Vị trí Gaussian | Learnable `means` trong world | Suy ra từ `anchor + source_kf pose` |
| Scale | Learnable xyz dớng | Learnable radius đẳng hướng |
| Rotation | Learnable quaternion | Identity quaternion |
| Densification | `gsplat.DefaultStrategy` clone/split/prune | Insert theo coverage trong section |
| Optimize scope | Full map params trong sliding camera window | Current section params |
| Pose update | Cập nhật viewmat render | Cập nhật cả viewmat và materialized means |
| Output | PLY chuẩn | `.pt` native + PLY materialized |

VTGS hiện tại ưu tiên ổn định hình học và memory hơn expressiveness. Mapper chuẩn
có khả năng biểu diễn mạnh hơn nhưng cũng dễ bị lỗi khi Gaussian trôi, scale sai
hoặc densification/pruning không phù hợp.

---

## 7. Các tham số quan trọng

Chọn mapper:

```yaml
gaussian_splat:
  online_enabled: true
  mapper_type: view_tied
```

Section:

```yaml
vt_section_size: 8
vt_head_insert_cap: 120000
vt_regular_insert_cap: 60000
vt_max_section_gaussians: 500000
```

Coverage:

```yaml
vt_silhouette_thresh: 0.2
vt_depth_cover_tol: 0.05
```

Training:

```yaml
vt_steps_head: 300
vt_steps_regular: 80
vt_steps_idle: 20
vt_n_iters_fine: 10000
```

Loss:

```yaml
vt_lambda_rgb: 1.0
vt_lambda_depth: 0.1
vt_lambda_ssim: 0.2
```

Radius:

```yaml
vt_min_radius: 1.0e-5
vt_max_radius: 0.20
```

---

## 8. Lệnh chạy và đánh giá

Chạy VTGS mapping:

```bash
python main.py \
  --dataset datasets/tum/rgbd_dataset_freiburg1_room/ \
  --config config/eval_vtgs.yaml
```

Đánh giá PSNR/SSIM:

```bash
python scripts/eval_gs_psnr.py logs/<run_dir> \
  --device cuda \
  --ply logs/<run_dir>/<seq_name>_vtgs_online_gs.ply
```

Mở viewer:

```text
Mở <seq_name>_vtgs_online_gs.ply trong SuperSplat
```

Không mở `.pt` trong SuperSplat. `.pt` là native state của mapper.

---

## 9. Kết luận

Sau khi tích hợp VTGS mapping, hệ thống trở thành một pipeline gồm:

```text
RGB frames
  -> MASt3R-SLAM tracking
  -> keyframe + pointmap + confidence + pose
  -> backend global optimization
  -> VTGS online mapping worker
  -> section-based view-tied optimization
  -> final pose sync/rebuild/fine refinement
  -> native VT state + materialized PLY
```

Điểm cốt lõi của phương pháp mới là chuyển từ Gaussian tự do sang Gaussian gắn
với view/keyframe nguồn. Điều này tận dụng trực tiếp thế mạnh của MASt3R-SLAM:
pointmap dày và pose được backend tối ưu. Mapper không cố học lại hình học từ
đầu mà chỉ tối ưu các thuộc tính nhẹ hơn của Gaussian. Đây là hướng phù hợp để
giảm lỗi map rời rạc/blob/floater trong online 3DGS mapping, đồng thời vẫn giữ
được khả năng xuất PLY chuẩn cho viewer và đánh giá định lượng.
