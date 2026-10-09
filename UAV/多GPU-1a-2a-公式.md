# ⚠ 本文档已作废 —— 请看新版

**原因**：本文给出的 (1a-MG) / (2a-MG) 是 **layer→GPU** 场景（一个模型的层拆到多卡、
跨卡聚合重载延迟 $D_i^{\mathrm{agg}}$）的公式。**2026-07-31 组会**澄清后确认，
实际场景是 **task→server**（多服务器任务路由），这套公式形态不再适用。

---

## 请改看

| 想了解 | 看这份 |
|---|---|
| 新场景的 (1a)/(2a) 及完整公式（可渲染 LaTeX） | **`系统建模整理-多服务器.md`** 第 1、2 章 |
| 英文正式版（给导师） | `SystemModel-Formulation-多服务器.md` |
| 全局导航 | `项目总览.md` |

---

## 新场景下这两个式子长什么样（速览）

**最少常驻层数**（论文式 1a 的 per-(服务器, 模型) 版）：

$$
\check i_{m,k}=\min\bigl\{\,s\ \big|\ D_{m,k,s}\le D^{\mathrm{QoS}}_k\,\bigr\},
\qquad D_{m,k,s}=\frac{C_{k,N_k}-C_{k,s}}{\beta_m}
$$

**新增**共享显存可行性（论文里不存在，因为它只有 1 个模型）：

$$
\sum_{k} C_{k,\check i_{m,k}}\le\bar C,
\qquad
C_{k,N_k}+\sum_{k'\ne k} C_{k',\check i_{m,k'}}\le\bar C\ \ (\text{服务器 }m\text{ 能服务模型 }k)
$$

**目标**（论文式 2a 的多服务器推广）：

$$
\min_{\{x_{j,m}\},\,\{t_{m,k,i}\}}\ \ 
\omega_1\sum_{j}\underbrace{\Bigl(T^{\mathrm{tx}}_{j,m}+D_{m,k_j,s_{m,k_j}(t_j)}+\tfrac{W_j}{f_m}\Bigr)}_{T_j:\ \text{传输}+\text{加载}+\text{计算}}
\;+\;\omega_2\underbrace{\sum_m\sum_t\sum_k C_{k,s_{m,k}(t)}\tau}_{\text{内存成本 }\Phi}
\;-\;\omega_3\underbrace{\sum_t F(t)}_{\text{Jain 负载均衡}}
$$

相对论文 (2a) 的变化：**新增任务路由变量 $x_{j,m}$、传输项、计算项、Jain 项**，内存成本对 $m,k$ 双重求和。

**退化一致性**：单服务器 + 单模型 + 无传输无计算时，精确退回论文 (1a)/(2a)，
TABLE I 四成本 100/140/185/225 全部复现（代码已断言）。

> 保留本存根是为了不破坏可能已分享出去的链接。
