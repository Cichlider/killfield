# Web 运行时取舍：Killfield UI + Rust/WASM 引擎

## 结论

网页展示以 `viewer/` 为入口，保留 Killfield 的视觉与手机交互，但计算只使用
`engine/` 编译出的 Rust/WASM。没有把 Rust 引擎复制回独立 Killfield JS 仓库，也没有把
两套物理同时保留。

原因：训练 Python、原生性能测试和浏览器 viewer 现在共享同一个 Rust crate。WASM 通过
C ABI 输出扁平 `f32` 渲染缓冲区，避免逐帧 JSON 和第二套物理移植。512-ray 实测密度场
约 `0.285ms`，而历史 JS 基线约 `12.2ms`；2048-ray MPC 原生 p95 约 `2.135ms`、最大
约 `4.889ms`。WASM 即使略慢于原生，仍明显低于 60Hz 的 `16.7ms` 帧预算。

## 线程与帧率

- Rust/WASM 物理与 MPC 固定 `25Hz / 40ms`，保持训练语义不变；
- `requestAnimationFrame` 按屏幕刷新率绘制；
- viewer 在相邻引擎状态之间插值坦克、角度与子弹位置，使 60/120Hz 屏幕不再显示为
  25Hz 位置跳变；
- Canvas DPR 上限为 2，避免高 DPR 手机把简单图形放大成数百万像素重绘；
- PPO HTTP 推理继续异步，迟到动作沿用上一个动作，不冻结或补跳游戏帧。

当前 MPC 仍在浏览器主线程同步调用 WASM。基准最大规划耗时低于一帧预算，因此暂不增加
Worker、SharedArrayBuffer 和跨线程状态协议；只有实机遥测重新出现超过 16ms 的长任务时
才值得引入 Worker。

## 人工控制与 PPO Action 的边界

PPO 的 Movement 17 类语义保持原设计：16 个世界方向 + STOP，底层只前进。人工网页轮盘
使用独立的人工控制路径：默认是 270° 前进区与正后方 90° 后退区。Page 的「前向对齐」
滑杆可以在 0°–360° 间调整；设为 360° / 360° 时完全禁用后退。设置保存在浏览器本地，
不会无声修改 checkpoint 的 Action contract。

## 手机展示

- 左侧三十二方向轮盘，右侧独立开火；
- 触控层可隐藏和恢复；
- 支持元素全屏及固定定位回退；
- 支持时调用 Screen Orientation API 锁定横屏；失败时显示关闭竖屏锁并旋转手机的提示；
- `manifest.webmanifest` 为添加到主屏幕的 Web App 声明横屏偏好。
