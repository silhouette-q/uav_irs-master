# 交接文档（给接手的 agent / 协作者）

> 自包含，读完即可接手。工作目录 `D:\UAV`　｜　最后更新：2026-09-11
> 全局导航另见 `项目总览.md`。

---

## 0. 一句话现状

学生（研究生）在导师要求下研究**多服务器 LLM 推理卸载**：单用户产生异构任务，路由到 M 台距离不同的边缘/云服务器，各服务器自主决定层卸载。**系统建模已完成并全部验证通过**（代码在 `edge-unload-repro\`），英文 Formulation 已产出；架构图按导师要求迭代到 **v7**（TJCCT 风格双时标，见第 12 节），一并等导师审阅。下一步是接毕设的扩散策略 DRL 求解。

**⚠ 最容易踩的两个坑，先看第 2、3 节；画图与符号的坑看第 12 节。**

---

## 1. 人物与背景

- **用户**：研究生，长期方向 GDM-DRL（生成式扩散 + DRL）用于无线通信资源管理。本科毕设做过 **GD-MAPPO**（扩散策略网络 + MAPPO + CTDE），用于多无人机 ISCC 场景。
- **导师**：李家辉 / Jiahui LI。
- **任务链**：导师指定一篇可 follow 的论文 → 读懂公式 → 复现系统模型 → 扩展场景 → 迁移毕设方法求解。
- **目标论文**：*Efficient Layer-Granularity Unloading for LLMs in Edge Computing*（IEEE TMC 2026）
  PDF: `C:\Users\znhy\Downloads\Efficient_Layer-Granularity_Unloading_for_LLMs_in_Edge_Computing.pdf`
  方法是 **ski-rental + 拉格朗日对偶，不是 DRL** —— 它是我们要用 DRL 替代/增强的对象。

---

## 2. ⚠ 坑一：场景变更过一次（layer→GPU 已作废）

组会最初说"多 GPU"，一度被理解为**把一个模型的层拆到多张卡**（流水线并行）。**2026-07-31 组会导师画图澄清**：实际是**多台不同距离的边缘/云服务器，各存完整模型，用户的异构任务被路由到某一台**。

**后果**：
- `system_model_multi_gpu.py`、`env_multi_gpu.py`、`validate_multi_gpu.py` **已废弃**（文件头有 DEPRECATED 横幅，仅历史参考）。
- `system_model.py`（单 GPU 单模型）**仍然有效**，而且升级成了"每台服务器的内核"。
- 部分文档的多 GPU 章节作废 —— 见第 5 节文档地图。

**接手时不要再基于 layer→GPU 做任何扩展。**

---

## 3. ⚠ 坑二：导师的阶段性边界

1. **本阶段只做系统建模，不实现求解算法**。LUA、ski-rental 对偶解、DRL agent **一律不要写**。
2. 传统在线算法（竞争比/近似比、online paging、K-server）只需**基础了解**，别深挖；重心放在系统建模。
3. **7-31 组会待办**：绘制系统架构简图 + **用英文撰写 System Model 与 Formulation**，完成后与导师审阅，**再继续后续工作**。→ 已产出，等审阅。
4. **AI 工具可以用**来提升效率（早期"必须纯手写"的要求后来放宽），但学生要能讲清每个公式和每段代码。

**不确定是否越界（尤其"写不写算法"）时，先问用户。**

---

## 4. 当前场景

```
                 ┌─ 无线 (R1, δ1) ─→ Server 1  近边缘   算力弱   显存 C̄
   用户 ─────────┼─ 无线 (R2, δ2) ─→ Server 2  中边缘   算力中   显存 C̄
  (K 类异构任务)  └─ 无线 (R3, δ3) ─→ Server 3  云       算力强   显存 C̄
```

### 逐条设定（7-31 组会定场景；★ 为 8-21 组会定的决策归属与时标）

| 维度 | 设定 |
|---|---|
| 用户 | **单用户**，按概率随机产生任务 |
| 任务异构性 | ① 需要**不同的模型**（K 个）② **到达率不同** λ_k |
| 模型放置 | 每台服务器存**完整**模型，不拆层；任务在一台服务器上跑完 |
| 显存 | 各服务器**容量相同** C̄，但 **K 个模型共享**（新增耦合） |
| 算力 | 各服务器**不同**（云强边缘弱） |
| 通信 | 无线速率 R_m + 回程时延 δ_m，越远越大 |
| 智能体 | ★**用户 1 个 + 服务器 M 个**（异质）：用户决路由（每时隙）、服务器决层保留（每 Δ 时隙） |
| 目标 | min 推理时延 + 内存成本 − Jain 公平（负载均衡） |
| "搬运成本" | = 任务数据上传/回传，**就是传输时延本身**，不是独立第 4 项 |

### 三重权衡 + 双向耦合

- 近但弱（传输小/计算慢）vs 远但强 vs 该模型是否已常驻（加载延迟 ~2385 ms，压倒性）
- 路由决定每个「服务器×模型」的到达率 λ_{m,k} → 决定最优卸载时机；反过来卸载状态决定路由是否划算

### ★ 决策归属与双时标（8-21 组会拍板，最关键的一条）

| 决策 | 谁决定 | 频率 | 拟用方法 |
|---|---|---|---|
| 任务路由 x_{j,m} | **用户**（用户本身是一个智能体） | 每时隙 τ=100 ms | DRL |
| 层保留 s_{m,k} | **每台服务器**各自决定 | 每 Δ 时隙（默认 10 = 1 s） | MADRL |

- **服务器不竞价、不拒单**（原先的竞价机制被明确驳回）
- **server 的动作必须写进用户的环境** —— 落在用户观测里的 `D_{m,k,s}/T^max_k`（预估重载/deadline）
- Δ=1 时精确退化为原来的单时标模型（有测试断言）
- 奖励按"谁控制哪一项"切：用户背 等待+传输+计算+公平+罚，服务器背 加载+显存；
  **两边相加仍精确等于 −J**

---

## 5. 文档地图

### ✅ 有效

| 文档 | 内容 |
|---|---|
| `项目总览.md` | **全局导航，先读这个** |
| `系统建模整理-多服务器.md` | 中文数学版系统建模（可渲染 LaTeX） |
| `SystemModel-Formulation-多服务器.md` | **英文 System Model + Formulation（给导师的交付物）** |
| `系统建模-代码说明.md` | 代码片段 + 注释，逐要素说明 |
| `代码与论文差异说明.md` | 新场景 vs 论文的差异（汇报口径） |
| `edge-unload-repro/README.md` | 代码怎么组织、怎么跑、验证结果 |
| `edge-unload-repro/figures/draw_architecture_v7.py` | **当前架构图脚本（v7）**，产出 fig_architecture_v7.pdf/png；v4/v5/v6 脚本留作参考 |
| `_tjcct_full.txt` | TJCCT 参考论文全文提取（**Grep 对此文件无效**，用 Read 分段读） |
| `LLM边缘卸载论文-复现汇报.md` | 论文第四节 A→H 全公式详解（讲论文本身，不受场景变化影响） |
| `毕设-MDP要素提取.md` | 毕设 GD-MAPPO 的 MDP + 迁移映射草图 |
| `汇报讲稿.md` | 公式精读阶段的汇报口播稿（历史） |

### ⚠ 已转为指针 / 过时

`系统建模整理-单GPU与多GPU.md`、`多GPU-1a-2a-公式.md` —— 已改为指向新文档的存根（内容讲的是 layer→GPU）。

---

## 6. 代码

`edge-unload-repro\`，纯 Python + numpy（画图需 matplotlib）。

### 主线（新场景）

```
llm_models.py             LLM 逐层内存规格；TABLE III 实测值
channel.py                无线信道：路损、香农速率、传输时延（含回程）
tasks.py                  单用户 K 类异构任务、到达率 λ_k、QoS/deadline
server_model.py           单服务器：K 模型共享显存、i_check/B 钩子、加载与计算时延、显存驱逐
system_multi_server.py    全系统：三项时延、内存成本、Jain、加权目标 J
env_multi_server.py       异质双时标 MARL 环境：JointAction(user (K,M), servers (M,K))
validate_multi_server.py  19 项验证
demo_multi_server.py      5 个参考策略对比
figures/                  架构图 png/pdf/mmd/tex + 生成脚本
```

### 论文基线（有效，每台服务器的内核参考）

`system_model.py`、`env_single_gpu.py`、`baselines.py`、`validate_table1.py`、`demo.py`、`arrivals.py`

### ⚠ 已废弃

`system_model_multi_gpu.py`、`env_multi_gpu.py`、`validate_multi_gpu.py`（layer→GPU）

### 运行

```bash
pip install numpy matplotlib
python run_tests.py                  # 论文基线 + 多服务器（主线）
python run_tests.py --server         # 只跑多服务器 19 项
python validate_multi_server.py --list
python demo_multi_server.py
python figures/draw_architecture.py
```

### 单位约定（务必遵守）

内存 MB ｜ 时延 ms ｜ 数据 Mbit ｜ 速率 Mbit/s ｜ 加载带宽 β MB/ms ｜ 计算量 GFLOP ｜ 算力 f GFLOP/ms

---

## 7. 关键技术结论（省得重新踩）

1. **论文正文笔误**：$D_i=\sum_{i'\ge i}d_{i'}$ 应为 $\sum_{i'>i}$。只有 `>i` 能复现 TABLE I（100/140/185/225），也才与状态定义、算法伪代码自洽。
2. **TABLE I 参数**：5 层 × 1GB/1s、η=30、QoS≤2s ⇒ i_check=3、可卸层{4,5}、B=30、四成本 100/140/185/225。**用它做 ground-truth。**
3. **TABLE III 实测**：embedding 1188MB/380ms、transformer 386MB/125ms ×32、fc 1188MB/380ms ⇒ 34 层 / 14728 MB / 4760 ms；d/c≈0.32 ⇒ β≈3.088 MB/ms。
4. **B = η/β，与模型无关** —— 自动复现论文 C 节"d=kc ⇒ 所有直线交于同一点 B=ηk"。
5. **加载项压倒性**：传输 2\~80 ms、计算 5\~23 ms、加载 0 或 ~2385 ms。
6. **可行性缺口**：K 个模型的最小常驻之和可能超显存（8 GB 下就会）。已用 `hosts` 放置变量处理，有测试覆盖。
7. **编码坑**：Windows 控制台默认 GBK。脚本已加 `sys.stdout.reconfigure(encoding="utf-8")`；跑时也可设 `PYTHONIOENCODING=utf-8`。**别在输出里用 emoji**（GBK 会报错）。
8. **shell 是 git-bash**：用 `cd "D:/UAV/..."`，不要用 cmd 的 `cd /d`。
9. **旧文档的 $\bar\tau$ 应为 $\check i$**（最小保留层数；$\tau$ 是到达时间，别混）。

---

## 8. 验证状态

| 组 | 项数 | 状态 |
|---|---|---|
| 论文基线（单 GPU） | 8 | ✅ PASS |
| 多服务器（主线） | 17 | ✅ PASS |
| 已废弃（layer→GPU） | 7 | ✅ PASS（仅参考） |

**两个最有力的证据**：
1. 新模型在退化配置下精确复现论文 TABLE I 四成本；
2. MARL 环境**团队累计奖励 == −目标函数 J**（误差 <1e-6），三种策略下均成立。

---

## 9. 待导师确认的建模假设

| # | 假设 | 备注 |
|---|---|---|
| **1** | **Δ 取多大**（长时标 epoch，默认 10 时隙 = 1 s） | 跑 `demo_multi_server.py` 看第 ⑤ 张敏感性表。**最需确认。** |
| **1b** | **奖励分账**：用户背 等待+传输+计算+公平+罚；服务器背 加载+显存 | 团队和恒等式仍精确成立。含预取等待的外部性（有开关可重分配）。 |
| **2** | **可行性缺口**：需 `hosts` 放置变量 y_{m,k} | 若 Σ_k C_{k,i_check} > C̄ 该服务器无解。已实现+测试。 |
| 3 | decode 用 max(算力, 显存带宽) | 更精细可加 KV-cache 随上下文增长。 |
| 4 | 排队为单服务器 FIFO | 传输与计算流水；GPU 若支持批处理需改多服务台。 |
| 5 | 权重默认 ω=1 | 三项已归一化故可解释；建议做敏感性分析。 |

> ✅ **v4 已修复**（原为待确认项）：计算时延已拆 prefill/decode、已允许预取、
> QoS 统一为"deadline 唯一 + D^QoS 推导"、目标三项已归一化、**已加排队建模**。
> 详见 `对抗性审查-SystemModel-Formulation.md` 修复状态总表。

---

## 10. 下一步

1. **交付审阅**：架构图 **v7**（第 12 节；caption/(a)(b) 子图标签等用户提了再加）+ 英文 System Model / Formulation **v5.0** 给导师过。
2. **确认第 9 节假设**，尤其 Δ 取值与奖励分账。
3. **接 GD-MAPPO**：算法层（扩散策略 + MAPPO + CTDE）**照搬毕设不动**，只把环境换成 `env_multi_server`；离散选择沿用毕设的**连续松弛**技巧；用 `i_check` / `B_ms` **约束动作空间**（缩小无效解空间 + 保住在线算法的性能下界 —— 导师明确提过这点有论文价值）。

---

## 11. 协作约定

- 先读第 2、3 节的两个坑；不确定是否越界时**先问用户**，别擅自实现 LUA / DRL 求解。
- 代码：纯 Python + numpy，Gym 风格但不硬依赖 gymnasium；忠实对齐公式，关键处注明"公式↔代码"。
- 正确性以 **TABLE I** 为基准回归；改动后跑 `python run_tests.py`。
- 中文输出注意编码；**不要删用户的文件**（过时文档已改为存根指针而非删除）。

---

## 12. 双时标架构图与 TJCCT 参考（2026-08 底新增）

### 12.1 背景

导师要求模仿 TJCCT（*A Two-timescale Approach for UAV-assisted Mobile Edge Computing*，PDF 在 `d:\UAV\A Two-timescale Approach for UAV-assisted Mobile Edge Computing.pdf`）的图风格给架构图加"小时间轴"，作为第三章系统模型图。全文提取在 `d:\UAV\_tjcct_full.txt`（19 页）。

### 12.2 TJCCT 与我们的区别（结论：借骨架、不抄内容）

| | TJCCT | 我们 |
|---|---|---|
| 长时标（epoch） | UAV 轨迹 | 服务器层保留 $s_{m,k}$ |
| 短时标（slot） | 卸载 + 资源分配 + 定价 | 用户任务路由 $x_{j,m}$ |
| 解法 | Rubinstein 议价 + 多对一匹配 + 凸优化，**完全不用 RL**，SDN 集中式 | 异质 MARL（1 user + M server agents）、CTDE，计划 GD-MAPPO |
| 排队/显存 | 无 | FIFO 排队耦合 + K 模型共享显存 |

双时标骨架（中央时间轴 + 左右两列 + Decision making / State update 两行）同构 → 借风格站得住；场景与解法完全不同，内容不能抄。

### 12.3 符号对齐（画图/写文最重要的一条）

- **Slot = 短时标 = $\mathcal{T}$（下标 $t$），时长 τ=100 ms** → 用户路由。
- **Epoch = 长时标 = $\mathcal{T}_0$（下标 $t_0$），时长 Δτ=1 s（Δ=10）** → 服务器保留。
- 与 MD `SystemModel-Formulation-多服务器.md` TABLE I 首行、公式 (28) 一致。花体 𝒯/𝒯₀ 是下标**集合**，τ/Δτ 是**时长**，图上两者都写。
- **TJCCT 原图的坑**：轴上刻度侧别与列含义交叉（左绿刻度是 slot 却挨着 Long timescale 列，右蓝刻度是 epoch 却挨着 Short 列），纯排版问题、正文无依据，**不要继承**。
- 我们的对齐：左绿 = epoch（𝒯₀, Δτ）↔ 长时标列；右蓝 = slot（𝒯, τ）↔ 短时标列。位置=颜色=含义三者一致。

### 12.4 当前图交付物：v7

- 脚本 `edge-unload-repro/figures/draw_architecture_v7.py`，产出 `fig_architecture_v7.pdf/png`；`py -3 draw_architecture_v7.py` 运行（matplotlib Agg，脚本内不 show）。
- **v7 重构动机（师兄 2026-09 反馈）**：v6 的 Decision making / State update 横排矩阵太杂。v7 改为两张"时标卡片"，直接对应师兄三点：**(1)** 左蓝卡 = 短期（DO every slot: 路由 $x_{j,m}$；STATE every slot: $\lambda_k$/$R_m(t)$/$\mathfrak{b}_m$）；**(2)** 右绿卡 = 长期（DO every epoch: 层保留 $s_{m,k}$；STATE every epoch: 保留状态/显存占用）；**(3)** 卡间只放双向耦合箭头（长→短：$D_{m,k,s}$ 进路由成本；短→长：$\lambda_{m,k}$ 决定 idle time）+ 竖排 "(3) coupling" 标签。
- 下方两条：slot 排队甘特（含 epoch 嵌套括号、epoch 起点橙刻度 = 长时标动作点）；CTDE 条（user actor per slot + 3 server actors per epoch → centralized critic）。
- 谱系：v3（场景+HBM 比例条）→ v4（左场景右双时标两面板）→ v5（紧凑水平时间轴，用户不满意）→ v6（TJCCT 骨架+横排矩阵）→ **v7（当前，双卡片+耦合箭头，按师兄意见去掉横排矩阵）**。
- 配色：绿 #1E7A3C=长时标、蓝 #1F5FA8=短时标、橙 #CC4400=epoch 标记、红 #B22222=排队等待；serif 字体仿 IEEE。
- 挂起的可选项（用户提了再做）：caption、(a)(b) 子图标签、调色。

### 12.5 "K 个模型抢显存"标准答案（用户问过，可能再被问）

- 是**约束**不是策略：$\sum_k y_{m,k}C_{k,s_{m,k}(t)}\le\bar C$，MD 公式 (5)(6)(7)、问题约束 (27d)。
- 代码：`check_mem_cap` / `clip_states` / `make_room_for`（超容量自动压低保留状态腾空间）；`validate_multi_server.py` 有专项验证。
- 主算例 C̄=24 GB **有意不绑紧**（3 模型全常驻 = 18138 MB）；8-22 决策：不人为收紧，竞争效应留给第五章敏感性轴 `mem_cap_mb` 扫 16298~18138 MB（README 有记录）。

### 12.6 本线程环境坑

- **Grep 对 `_tjcct_full.txt` 返回 0 匹配**（连已存在字符串都搜不到，疑似编码）→ 用 Read 分段读。
- Windows 下 Python 用 `py -3`；画图脚本必须 Agg backend + `plt.close()`，不弹窗口。
