# Fish RL 开发 SOP

## 终极目标
- 在最具挑战的配置（捕食者速度快、数量多的小鱼、动作受限）下，训练出的策略仍能让绝大多数小鱼完成整局存活。
- 每次迭代都可复现：包含代码版本、训练参数、日志、曲线和检查点。
- 任意时刻可以回溯至某个 `dev_vX.md` 了解背景、假设和下一步计划。

## 工作流（循环迭代）
0. **启动新一轮**
   - 阅读历史 `dev_v*.md`（尤其上一轮的 “Learning / 下一步计划”）。
   - 把 `dev_v{X+1}.md` 当作工作文档：先写“本轮计划/假设/待验证列表”，后续每一步实时更新。

1. **复现上一轮基线**
   - 参考上一轮记录的命令，在开始前确认对应 `experiments/vX/train.py` 和检查点可跑通；必要时快速 sanity run，确保问题来自新改动而非旧环境漂移。

2. **观察 & 记录现状**
   - 对比最新基线数据与上一轮结论；若缺乏信心，先跑小规模实验。
   - 记录平均/终局存活率、奖励、损失、熵、日志路径；可以用 TensorBoard、纯文本或 JSON，但需在 `dev_v{X+1}.md` 内明确引用。

3. **设计与维护计划 (`dev_v{X+1}.md`)**
   - 将观察->假设->验证步骤写成可执行 checklist。
   - 对任何不确定项，先列“需要新增的 logging/metrics”。

4. **增加可观测性（如需）**
   - 在代码中添加轻量日志/TensorBoard scalar/JSON dump，注明开启方式。
   - 在 `dev_v{X+1}.md` 中记录新增信号及其用途，方便下一轮直接复用。

5. **实施改动与验证**
   - 小规模测试：例如降低迭代数、减小并行度；把结论写进工作文档。
   - 若结果符合预期，升级到目标规模；所有完整实验输出存于 `experiments/v{X+1}/artifacts/` 专属目录（`checkpoints/、logs/、tb_logs/、plots/`），并生成曲线图/参数摘要。
   - 若发现 Bug，优先在 `dev_v{X+1}.md` 标注，修复后重复小规模->大规模流程。

6. **收尾与交接**
   - 在 `dev_v{X+1}.md` 补全 “本轮 Learning / 下一步计划”，供下一轮开场阅读。
   - 将关键产物写入 `experiments/v{X+1}/artifacts/plots/`（曲线）与 `.../media/`（mp4/gif，即便 500 帧），这些目录是被版本控制的，需随代码一并提交并在文档中引用。
   - 提交相关代码（`experiments/v{X+1}/train.py` 等）、文档和可追溯的元数据，并推送到 `origin/master`，保证远端随时反映最新状态。
   - 下一轮启动时回到步骤 0，形成闭环。

## 目录结构
```
fish_rl/
├── src/                       # （预留）通用库/环境，当前 `fish_env.py` 等仍在根目录
├── experiments/
│   └── vX/
│        ├── train.py         # 当前版本的训练脚本
│        ├── dev_vX.md        # 工作文档
│        └── artifacts/
│             ├── checkpoints/
│             ├── logs/
│             ├── tb_logs/
│             ├── plots/      # 训练曲线、PNG、SVG
│             └── media/      # 视频/GIF（建议 mp4 以减小体积）
├── docs/（可选）
├── requirements.txt
└── SOP.md
```
- 历史版本只追加 `experiments/v{X}/` 文件夹，旧 artifacts 保持只读。
- 通用读取路径需以 `Path(__file__).resolve().parent` 为基准，避免硬编码根目录。
- 感知拓展选项：若需要研究“群体协作”行为，可在 `FishEscapeEnv._get_observations` 中加入其他小鱼的相对位置/速度等特征（例如邻近前 K 条鱼）；默认配置只观测捕食者，如果实验需要可在 `dev_vX.md` 说明后再开启。该拓展目前视为 stretch goal，可随时纳入后续迭代。

## 约定
- 任何 Python 操作前检查/创建/激活 `uv` 虚拟环境，并用 `uv pip install` 管理依赖。
- 日志渠道可混用（TensorBoard、文本、JSON）；但每轮至少保存一种易于 diff 的格式。
- `experiments/vX/train.py` 与 `experiments/vX/dev_vX.md` 的版本号保持同步：一次 major 迭代 +1。
- `experiments/vX/artifacts/` 下的历史模型视为不可变，只追加新目录。
- 本机资源：32 物理核 / 512 GB RAM，可稳定支撑 64~128 个并行环境；在 v2+ 迭代中默认把 `--num_envs` 设为 ≥64（建议 128）。
- 每轮结束必须 `git status` 检查、`git add` 相关文件、`git commit -m "..."` 并 `git push origin master`，形成远端可见的追溯链。
