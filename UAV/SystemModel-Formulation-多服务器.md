# III. System Model and Problem Formulation

| Item | Content |
|---|---|
| **Version** | v5.0 — **user agent + server agents on two timescales** (2026-08-21 meeting); structure follows *TJCCT: A Two-timescale Approach for UAV-assisted Mobile Edge Computing* (INFOCOM 2024) |
| **Edited** | 2026-08-21; TABLE I trimmed to 15 non-obvious symbols on 2026-08-26 |
| **Scenario fixed at** | 2026-07-31 meeting (multi-server routing); **decision ownership and timescales fixed at the 2026-08-21 meeting** (advisor: Jiahui LI) |
| **Base paper** | *Efficient Layer-Granularity Unloading for LLMs in Edge Computing*, IEEE TMC 2026 |
| **Code** | `edge-unload-repro\` — 19 validations passed; degenerate case reproduces TABLE I exactly; team reward $=-J$ verified for $\Delta\in\{1,10\}$ |

> **What changed in v5.0** (the previous version had servers bidding for tasks):
> routing is decided by the **user**, which is itself an agent; servers decide
> **only** layer retention; the two act on **two timescales** ($\tau$ vs $\Delta\tau$).
> The bidding rule of the former (31) is deleted. Sections 3.1.1, 3.6 and IV are
> rewritten accordingly; the cost model (3.2–3.4) and the problem $\mathbf{P}$ (3.5)
> are unchanged, since $x_{j,m}$ remains the same variable — only its owner changed.

> 中文备注（本版修复清单、建模假设、待确认项）见文末附录，正式投稿时删除。

---

In this section, a multi-server LLM inference offloading architecture is first
introduced, followed by the models of communication, layer loading, inference
computing, server occupancy, and cost. The joint task-routing and layer-retention
problem is then formulated.

## TABLE I: Notations

| Symbol | Description | Typical value |
|---|---|---|
| $\mathcal{T},\ \mathcal{T}_0$ | Set of slots (**short timescale**, user routing, duration $\tau$) / set of epochs of $\Delta$ slots (**long timescale**, server retention), see (28) | $\tau=100$ ms; $\Delta=10$, i.e. $1$ s |
| $\omega_m$ | GPU memory bandwidth of server $m$ (bounds the decode stage, see (16); **not** the objective weights below) | $900/900/2000$ MB/ms, i.e. $0.9$–$2.0$ TB/s (RTX 3090 to A100-80GB class) |
| $\beta_m$ | Storage-to-GPU bandwidth of server $m$ (governs layer loading) | $3.088$ MB/ms (measured, base paper TABLE III) |
| $\psi_k$ | Per-token forward computation of model $k$, derived via (2) | $14.73/3.14/0.27$ GFLOP/token |
| $C_{k,s},\ D_{m,k,s}$ | Cumulative memory / reload delay at retention state $s$ | $C_{k,N_k}=14728/3140/270$ MB; $D_{m,k,0}=4769$ ms (7B) |
| $\check{i}_{m,k}$ | Minimum number of resident layers imposed by QoS, see (13) | $19/14/3$ across the three servers (7B) |
| $D_{m,k}^{\mathrm{QoS}}$ | Loading-delay budget, **derived** from the deadline via (12) | $2146$ ms (7B, $m=1$) |
| $B_{m,k}$ | Ski-rental break-even instant (analytical hook), see (37) | $424$ ms (7B) |
| $T_k^{\max}$ | End-to-end deadline of type-$k$ tasks (**the only exogenous QoS parameter**) | $5500/900/120$ ms |
| $\mathfrak{b}_m$ | *Busy-until* instant of server $m$ (server occupancy), see (19) | — |
| $x_{j,m}$ | **Decision of the user** (every slot): task $j$ is served by server $m$ | binary |
| $s_{m,k}(t)$ | **Decision of server $m$** (every epoch): retention state of model $k$ in slot $t$; may decrease (unloading) or increase (prefetching) | $\in\{\check{i}_{m,k},\dots,N_k\}$ |
| $y_{m,k}$ | Placement indicator: server $m$ hosts model $k$ (**given**, not a decision) | binary |
| $\tilde\Psi,\ \tilde\Phi,\ F$ | Normalized latency / memory cost / Jain fairness index | dimensionless |
| $\omega_1,\omega_2,\omega_3$ | Weights of the (dimensionless) objective terms | $1/1/1$ (default) |

Only non-obvious symbols are tabulated; standard quantities ($\mathcal{M}$, $\mathcal{K}$,
$\mathcal{I}_k$, $\mathcal{J}$, $\bar C$, $f_m$, $\ell_m$, $\lambda_k$, $c_{k,i}$,
$d_{m,k,i}$, $R_m(t)$, $L_j^{\mathrm{in/out}}$, $\rho_m$) are defined at first use.
The third column lists the values adopted in the evaluation (Section V) and is given
only to indicate the physical order of magnitude of each quantity; the model itself
imposes no such choice. Multiple entries are ordered from the nearest edge server to
the cloud, or from the largest to the smallest task type.

---

## 3.1 System Model

### 3.1.1 System Overview

We consider a multi-server LLM inference offloading system as shown in Fig. 1.

**In the spatial dimension**, the system comprises a *user layer*, an *edge layer*,
and a *cloud layer*. *At the user layer*, a single mobile user stochastically
generates heterogeneous LLM inference tasks. *At the edge layer*, a set of edge
servers is deployed at different distances from the user, each equipped with one
GPU. *At the cloud layer*, a remote cloud server provides abundant computing
capability at the price of a larger transmission delay. The edge servers and the
cloud server are collectively referred to as *servers*, indexed by
$m\in\mathcal{M}=\{1,\dots,M\}$, ordered such that $\ell_1<\dots<\ell_M$ and
$f_1<\dots<f_M$, i.e. **a server closer to the user offers a lower transmission
delay but a weaker computing capability, and vice versa**. Every server stores the
*complete* parameter set of the models it hosts, so that a task is executed
entirely on one server and no inter-server activation transfer is required. The
user accesses the servers over a wireless link.

**In the decision dimension**, the two groups of decisions are made by two
*different* entities. The **user** decides the *task routing*, i.e. which server
serves each arriving task, whereas **each server** decides its own *layer
retention*, i.e. which layers of which hosted models are unloaded from or prefetched
into its GPU memory. A server neither refuses nor bids for a task, and no central
controller dictates the retention of any server.

**In the temporal dimension**, the time horizon is discretized into $T$ slots of
equal duration $\tau$, denoted by $\mathcal{T}=\{1,\dots,t,\dots,T\}$, which is
consistent with the coherence block of the wireless channel. The two decision groups
operate on **two different timescales**: routing must react to every arrival and is
therefore decided in every slot, whereas layer retention is a rent-or-buy decision
whose natural granularity is an idle period and is therefore decided once every
$\Delta$ slots, as formalized in Section 3.6. Within each slot, the task arrivals,
the channel state, and the layer residency of every server are captured and updated.
Since the service time of a task may span multiple slots, each server maintains an
occupancy state and pending tasks are queued, as detailed in Section 3.3.3.

### 3.1.2 Basic Models

The basic models of the system are given as follows.

**(1) Task Arrival Model.** The user generates $K$ types of heterogeneous tasks.
A type-$k$ task requires model $k$ and arrives according to an independent
Bernoulli process with rate $\lambda_k$, i.e. the indicator of a type-$k$ arrival
in slot $t$ satisfies

$$
\zeta_k^t\in\{0,1\},\qquad \Pr\{\zeta_k^t=1\}=\lambda_k,\qquad \forall k\in\mathcal{K},\ t\in\mathcal{T}. \tag{1}
$$

The heterogeneity is therefore twofold: different types **require different
models**, and different types have **different arrival rates**. The task set is
accordingly $\mathcal{J}=\{(k,t)\mid \zeta_k^t=1\}$, i.e. each realized arrival
constitutes one task, and each task $j\in\mathcal{J}$ is characterized by the tuple

$$
j \triangleq \bigl(t_j,\ k_j,\ S_j^{\mathrm{in}},\ S_j^{\mathrm{out}},\
L_j^{\mathrm{in}},\ L_j^{\mathrm{out}},\ T_{k_j}^{\max}\bigr),
$$

where $t_j$ is the arrival slot, $k_j$ the required model, $S_j^{\mathrm{in}}$ /
$S_j^{\mathrm{out}}$ the input / output data size, and $L_j^{\mathrm{in}}$ /
$L_j^{\mathrm{out}}$ the number of input / output **tokens**, which determine the
prefill and decode workloads respectively (Section 3.3.2). The deadline
$T_{k_j}^{\max}$ is the **only** QoS parameter specified exogenously; the
loading-delay budget is derived from it in (12).

**(2) LLM Model.** Model $k\in\mathcal{K}$ is characterized by the tuple
$\langle N_k,\ \{c_{k,i}\}_{i\in\mathcal{I}_k},\ \psi_k\rangle$, where $N_k$ is the
number of layers, $c_{k,i}$ the memory footprint of layer $i$, and $\psi_k$ the
per-token forward computation. Layers are indexed in the order in which they are
used during inference (embedding, transformer blocks, output head), so that layers
used earlier carry smaller indices. Following the standard estimate that a
transformer forward pass costs about two floating-point operations per parameter
per token, and that the resident footprint of a model equals its parameter count
times the numerical precision $b$ (bytes per parameter), $\psi_k$ follows directly
from the model footprint:

$$
\psi_k=\frac{2\,C_{k,N_k}}{b}. \tag{2}
$$

For instance, a footprint of $14728$ MB at $b=2$ (FP16) corresponds to $7.36$ G
parameters and $\psi_k=14.73$ GFLOP/token, consistent with a 7B-class model.

**(3) Server Model.** Server $m\in\mathcal{M}$ is characterized by the tuple
$\langle\bar{C},\ \beta_m,\ f_m,\ \omega_m,\ \ell_m,\ \delta_m\rangle$, where
$\bar{C}$ is the GPU memory capacity, identical across servers, $\beta_m$ the
storage-to-GPU bandwidth, $f_m$ the computing capability, $\omega_m$ the **GPU
memory bandwidth** (which bounds the decode stage, see (16)), $\ell_m$ the distance
to the user, and $\delta_m$ the one-way backhaul delay. A binary indicator
$y_{m,k}\in\{0,1\}$ denotes whether server $m$ hosts model $k$. Each server
processes at most one task at a time; its occupancy is tracked by the *busy-until*
instant $\mathfrak{b}_m$ defined in Section 3.3.3.

**(4) Layer Retention Model.** Following the layer-granularity unloading
principle, the residency of model $k$ on server $m$ is described by a **retention
state** $s_{m,k}(t)\in\{0,\dots,N_k\}$: layers $1,\dots,s_{m,k}(t)$ are resident in
GPU memory while the remaining layers have been unloaded. Accordingly, the occupied
memory and the delay required to restore full residency are respectively

$$
C_{k,s}=\sum_{i\le s} c_{k,i}, \tag{3}
$$

$$
d_{m,k,i}=\frac{c_{k,i}}{\beta_m},\qquad
D_{m,k,s}=\sum_{i>s} d_{m,k,i}=\frac{C_{k,N_k}-C_{k,s}}{\beta_m}. \tag{4}
$$

Eq. (4) adopts the widely validated observation that the loading delay of a layer
is dominated by its parameter volume and the storage-to-GPU bandwidth, hence
proportional to its memory footprint. Although $\beta_m$ is written as
server-dependent for generality, the servers considered here are equipped with
identical storage interfaces, so $\beta_m=\beta$ is adopted in the evaluation.
Since $C_{k,s}$ is non-decreasing and $D_{m,k,s}$ is non-increasing in $s$,
**a state occupying more memory always incurs a smaller reload delay, and vice
versa**, which is the origin of the memory–latency conflict.

**(5) Shared-Memory Model.** Unlike the single-model case, the $K$ co-hosted
models compete for the same GPU memory budget of a server:

$$
\sum_{k\in\mathcal{K}} y_{m,k}\,C_{k,\,s_{m,k}(t)}\;\le\;\bar{C},
\qquad\forall m\in\mathcal{M},\ t\in\mathcal{T}. \tag{5}
$$

Two feasibility conditions follow from (5). First, the QoS-mandated minimum
residency of all hosted models must fit into the memory, i.e.

$$
\sum_{k\in\mathcal{K}} y_{m,k}\,C_{k,\check{i}_{m,k}}\;\le\;\bar{C}. \tag{6}
$$

Second, since serving a type-$k$ task requires model $k$ to be *fully* resident
while the remaining models can at best be compressed to their own minimum
residency, server $m$ is able to serve model $k$ only if

$$
C_{k,N_k}+\sum_{k'\ne k} y_{m,k'}\,C_{k',\check{i}_{m,k'}}\;\le\;\bar{C}. \tag{7}
$$

Let $\mathcal{M}_k$ denote the set of servers satisfying $y_{m,k}=1$, (7), and
$D_{m,k}^{\mathrm{QoS}}\ge0$, i.e. those eligible to serve type-$k$ tasks.

---

## 3.2 Communication Model

### 3.2.1 Channel Power Gain

The user accesses the servers over a wireless link. Adopting a distance-dependent
path-loss model, the channel power gain of link $m$ in slot $t$ is given by

$$
|h_m(t)|^{2}=g_0\,\ell_m^{-\alpha}\,|\tilde h_m(t)|^{2}, \tag{8}
$$

where $g_0$ is the reference channel gain at unit distance, $\alpha$ the path-loss
exponent, and $\tilde h_m(t)$ the small-scale fading coefficient.

### 3.2.2 Transmission Delay

With transmit power $P$, bandwidth $B$, and noise power spectral density $N_0$,
the achievable uplink rate of link $m$ in slot $t$ is

$$
R_m(t)=B\log_2\!\Bigl(1+\frac{P\,|h_m(t)|^{2}}{N_0 B}\Bigr), \tag{9}
$$

and the downlink rate $R_m^{\mathrm{dl}}(t)$ is obtained analogously. A server
farther from the user additionally incurs a one-way backhaul delay $\delta_m$, with
$\delta_1<\dots<\delta_M$. The transmission delay of task $j$ if served by server
$m$ therefore consists of the uplink upload of the input, the downlink return of
the result, and the round-trip backhaul:

$$
T_{j,m}^{\mathrm{tx}}
=\underbrace{\frac{S_j^{\mathrm{in}}}{R_m(t_j)}}_{\text{input upload}}
+\underbrace{\frac{S_j^{\mathrm{out}}}{R_m^{\mathrm{dl}}(t_j)}}_{\text{result return}}
+\underbrace{2\delta_m}_{\text{backhaul}} . \tag{10}
$$

Eq. (10) is *task-level*: it depends on the realized sizes of task $j$ and on the
instantaneous rate in its arrival slot, and is therefore only available once the task
has arrived. The QoS budget (12), by contrast, must be dimensioned **before** any
arrival, since it determines the pinned layers (14) that have to be resident in
advance. We therefore also define the *type-level nominal* transmission delay
$T_{k,m}^{\mathrm{tx}}$, obtained by evaluating (10) with the nominal payload
$\bigl(\bar S_k^{\mathrm{in}},\bar S_k^{\mathrm{out}}\bigr)$ of type $k$ and the mean
rates $\bar R_m=\mathbb{E}[R_m(t)]$, $\bar R_m^{\mathrm{dl}}$:

$$
T_{k,m}^{\mathrm{tx}}
=\frac{\bar S_k^{\mathrm{in}}}{\bar R_m}
+\frac{\bar S_k^{\mathrm{out}}}{\bar R_m^{\mathrm{dl}}}
+2\delta_m . \tag{10$'$}
$$

The same convention applies to the nominal computation delay $T_{k,m}^{\mathrm{cp}}$,
obtained by evaluating (17) with the nominal token counts
$\bigl(\bar L_k^{\mathrm{in}},\bar L_k^{\mathrm{out}}\bigr)$ of type $k$. Both
quantities enter **only** the QoS budget (12); the latency accounting (21) and the
deadline constraint (27h) always use the task-level values (10) and (17). Because for
text-oriented LLM tasks the payload is small and $\delta_m$ dominates
$T_{k,m}^{\mathrm{tx}}$ (measured: $2.2/16.3/80.4$ ms), replacing the instantaneous
rate by its mean perturbs the budget by well under one percent and does not affect
$\check{i}_{m,k}$.

Throughout we assume $R_m(t)\ge R^{\min}$ for all $m,t$, so that every server remains
reachable over the horizon; the transmit power and bandwidth are exogenous in this
work, and extending the formulation to joint power and bandwidth allocation is left
as future work.

---

## 3.3 Computation Model

Serving a task consists of reloading the missing layers of the required model and
then performing the inference, both of which occupy the GPU. This subsection
models the two stages and the resulting server occupancy.

### 3.3.1 Layer Loading Model

**(1) Loading Delay.** If task $j$ starts service when model $k_j$ on server $m$ is
in retention state $s_{m,k_j}$, the unloaded layers must be reloaded first, taking

$$
T_{j,m}^{\mathrm{ld}}=D_{m,\,k_j,\,s_{m,k_j}} . \tag{11}
$$

**(2) QoS-Induced Minimum Residency.** The end-to-end deadline $T_k^{\max}$ is the
only exogenous QoS parameter. The admissible loading delay is therefore whatever
remains after the transmission and the inference computation have been accounted
for:

$$
D_{m,k}^{\mathrm{QoS}}=T_k^{\max}-T_{k,m}^{\mathrm{tx}}-T_{k,m}^{\mathrm{cp}}, \tag{12}
$$

where $T_{k,m}^{\mathrm{tx}}$ and $T_{k,m}^{\mathrm{cp}}$ are the *type-level nominal*
transmission and computation delays defined in (10$'$) and below it. Since
$D_{m,k,s}$ is non-increasing
in $s$, the minimum number of layers of model $k$ that server $m$ must keep
resident is

$$
\check{i}_{m,k}=\min\bigl\{\,s\in\{0,\dots,N_k\}\ \big|\ D_{m,k,s}\le D_{m,k}^{\mathrm{QoS}}\,\bigr\}, \tag{13}
$$

and consequently the first $\check{i}_{m,k}$ layers are pinned in memory, i.e.

$$
s_{m,k}(t)\ge\check{i}_{m,k},\qquad\forall t\in\mathcal{T}. \tag{14}
$$

**Remark 1 (self-consistency of the QoS specification).** Deriving
$D_{m,k}^{\mathrm{QoS}}$ from $T_k^{\max}$ rather than specifying both
independently removes the redundancy between the two, and makes the specification
self-consistent by construction: keeping exactly $\check{i}_{m,k}$ layers resident
is the *least* residency for which a task meets its deadline **in the absence of
queueing**, so any deadline violation is attributable solely to the waiting time in
(18). Numerically, compressing to $\check{i}_{m,k}$ consumes $98$–$100\%$ of the
deadline on all three servers, confirming the tightness of (13). Furthermore, since
$D_{m,k}^{\mathrm{QoS}}$ increases with the computing capability, **a faster server
enjoys a larger loading budget and hence a smaller $\check{i}_{m,k}$** — an
emergent coupling between computation and residency that is absent in the
single-server model. A negative $D_{m,k}^{\mathrm{QoS}}$ indicates that server $m$
cannot serve type $k$ at all, and is captured by $\mathcal{M}_k$.

### 3.3.2 Inference Computing Model

LLM inference proceeds in two stages whose bottlenecks differ, so a single lumped
workload cannot describe both.

**(1) Prefill Stage.** All $L_j^{\mathrm{in}}$ input tokens are processed in one
parallel forward pass, which is **compute-bound**:

$$
T_{j,m}^{\mathrm{pre}}=\frac{\psi_{k_j}\,L_j^{\mathrm{in}}}{f_m}. \tag{15}
$$

**(2) Decode Stage.** The $L_j^{\mathrm{out}}$ output tokens are generated one at a
time, and every step must read the entire weight matrix from GPU memory. The
per-token time is therefore the larger of the compute time and the
memory-bandwidth time, making the stage typically **memory-bandwidth-bound**:

$$
T_{j,m}^{\mathrm{dec}}=L_j^{\mathrm{out}}\cdot
\max\Bigl\{\underbrace{\frac{\psi_{k_j}}{f_m}}_{\text{compute}},\
\underbrace{\frac{C_{k_j,N_{k_j}}}{\omega_m}}_{\text{memory bandwidth}}\Bigr\}. \tag{16}
$$

**(3) Total Computation Delay.**

$$
T_{j,m}^{\mathrm{cp}}=T_{j,m}^{\mathrm{pre}}+T_{j,m}^{\mathrm{dec}}. \tag{17}
$$

**Remark 2 (which term dominates).** With the parameters of Section V, the second
argument of (16) attains the maximum on every server, i.e. **decode is
memory-bandwidth-bound**, in agreement with the well-known behaviour of LLM serving
systems. For a 7B-class model with $L^{\mathrm{in}}=512$ and
$L^{\mathrm{out}}=128$, the nearest edge server spends $1257$ ms on prefill and
$2095$ ms on decode, whereas the cloud spends $251$ ms and $943$ ms respectively.
Consequently the cloud, despite a $78$ ms larger transmission delay, attains the
lowest end-to-end latency for such compute-heavy tasks, while the nearest server
remains preferable for light tasks. **The "near-but-weak versus far-but-strong"
trade-off is therefore genuine and the routing decision is non-trivial**; this
would not hold under a lumped-workload model with an under-estimated workload.

### 3.3.3 Server Occupancy and Queueing

Because $T^{\mathrm{ld}}+T^{\mathrm{cp}}$ can far exceed the slot duration $\tau$,
several tasks may be routed to one server within a short interval. Treating them as
being served instantaneously and in parallel would understate the latency and admit
physically impossible schedules. We therefore model each server as a single-server
queue and track its **busy-until** instant $\mathfrak{b}_m$, initialized to $0$.
Denoting the absolute arrival time of task $j$ by $a_j=t_j\tau$, the waiting time
of task $j$ on server $m$ is

$$
T_{j,m}^{\mathrm{wait}}=\max\{0,\ \mathfrak{b}_m-a_j\}, \tag{18}
$$

and the occupancy is advanced by the GPU-occupying part of the service, i.e. the
loading and the computation (the transmission is carried by the network and
overlaps with the GPU pipeline):

$$
\mathfrak{b}_m\ \leftarrow\ \max\{\mathfrak{b}_m,\ a_j\}
+T_{j,m}^{\mathrm{ld}}+T_{j,m}^{\mathrm{cp}}. \tag{19}
$$

Prefetching, i.e. proactively restoring layers from $s$ to $s'>s$ during idle
periods, likewise occupies the GPU and advances $\mathfrak{b}_m$ by

$$
T_{m,k}^{\mathrm{pf}}(s\!\to\!s')=\frac{C_{k,s'}-C_{k,s}}{\beta_m}. \tag{20}
$$

Combining (10), (11), (17) and (18), the end-to-end latency of task $j$ is

$$
T_j=\sum_{m\in\mathcal{M}} x_{j,m}\Bigl(
\underbrace{T_{j,m}^{\mathrm{wait}}}_{\text{queueing}}
+\underbrace{T_{j,m}^{\mathrm{tx}}}_{\text{transmission}}
+\underbrace{T_{j,m}^{\mathrm{ld}}}_{\text{layer loading}}
+\underbrace{T_{j,m}^{\mathrm{cp}}}_{\text{computation}}\Bigr). \tag{21}
$$

Finally, the utilization of server $m$ over the horizon is

$$
\rho_m=\frac{1}{T\tau}\Bigl(\sum_{j\in\mathcal{J}} x_{j,m}\bigl(T_{j,m}^{\mathrm{ld}}+T_{j,m}^{\mathrm{cp}}\bigr)
+\sum_{t}T_{m}^{\mathrm{pf}}(t)\Bigr)\ \le\ 1, \tag{22}
$$

which is a **necessary** condition for server $m$ to remain stable, in the sense of a
work-conserving single-server queue whose long-term backlog stays bounded. It is not
sufficient: under bursty arrivals the backlog may still grow over long transients even
when $\rho_m<1$, and strict stochastic stability would additionally require $\rho_m$ to
be bounded away from one. We therefore impose (22) as a necessary stability condition
and report the realized utilization alongside the realized latency in Section V, rather
than claiming boundedness from (22) alone.

**Remark 3 (why queueing changes the problem qualitatively).** Modelling occupancy
has two consequences that are invisible otherwise. First, **load balancing becomes
a stability requirement rather than a fairness preference**: routing every task to
the nearest server drives $\rho_1$ to $2.74$ in the evaluation, so the queue
diverges and the waiting time accounts for $98\%$ of the latency, degrading the
objective by two orders of magnitude relative to a queue-aware policy. Second,
**unloading is no longer free in capacity terms**: the reload work competes with
inference for the same GPU, so an aggressive unloading policy that halves the
memory cost may push $\rho_m$ above one and destabilize the system. Both effects
create genuine tension that the joint decision must resolve.

---

## 3.4 Cost Model

The three cost terms have heterogeneous physical units (time, memory$\times$time,
and dimensionless). Adding them directly would be meaningless, so each term is
**normalized** before being combined, following common practice in
multi-objective resource-management formulations.

### 3.4.1 Normalized Latency Cost

Each task is normalized by its own deadline, so that a value of one corresponds to
exactly exhausting the QoS budget:

$$
\tilde\Psi=\sum_{j\in\mathcal{J}}\frac{T_j}{T_{k_j}^{\max}} . \tag{23}
$$

### 3.4.2 Normalized Memory Cost

The memory cost is the occupied memory accumulated over time, normalized by the
total capacity of the system so that each slot contributes the mean memory
utilization in $[0,1]$:

$$
\tilde\Phi=\sum_{t\in\mathcal{T}}\frac{1}{M\bar{C}}
\sum_{m\in\mathcal{M}}\sum_{k\in\mathcal{K}} y_{m,k}\,C_{k,\,s_{m,k}(t)} . \tag{24}
$$

### 3.4.3 Load Balancing Metric

The accumulated load of server $m$ up to slot $t$ is its cumulative GPU occupancy,
which comprises the loading, the computation **and the prefetching**, since all
three occupy the device:

$$
L_m(t)=\sum_{j:\,t_j\le t} x_{j,m}\bigl(T_{j,m}^{\mathrm{ld}}+T_{j,m}^{\mathrm{cp}}\bigr)
+\sum_{t'\le t}T_m^{\mathrm{pf}}(t') . \tag{25}
$$

Excluding the loading term would severely distort the metric, since the loading
delay can exceed the computation delay by two orders of magnitude. The load
balancing degree is then quantified by the Jain fairness index

$$
F(t)=\begin{cases}
\dfrac{\bigl(\sum_{m} L_m(t)\bigr)^{2}}{M\sum_{m} L_m^{2}(t)}, & \text{if } \sum_{m} L_m(t)>0,\\[2ex]
1, & \text{otherwise},
\end{cases}
\qquad F(t)\in\Bigl[\tfrac{1}{M},1\Bigr], \tag{26}
$$

which attains $1$ under a perfectly balanced load and $1/M$ when all tasks are
served by a single server. The second branch resolves the indeterminate form at the
beginning of the horizon; in particular $F(0)=1$.

---

## 3.5 Problem Formulation

The optimization problem is formulated to minimize the weighted cost of the system
over $T$ slots by jointly determining the **task routing strategy**
$\mathbf{X}=\{x_{j,m}\}$ and the **layer retention strategy**
$\mathbf{S}=\{s_{m,k}(t)\}$.

$$
\mathbf{P}:\quad
\min_{\mathbf{X},\,\mathbf{S}}\quad
\omega_1\underbrace{\tilde\Psi}_{\text{latency}}
+\omega_2\underbrace{\tilde\Phi}_{\text{memory}}
-\omega_3\underbrace{\textstyle\sum_{t}F(t)}_{\text{load balancing}} \tag{27a}
$$

$$
\begin{aligned}
\text{s.t.}\quad
&\sum_{m\in\mathcal{M}} x_{j,m}=1, && \forall j\in\mathcal{J} &&\text{(27b)}\\
&x_{j,m}\in\{0,1\},\quad x_{j,m}=0\ \text{ if } m\notin\mathcal{M}_{k_j}, && \forall j,m &&\text{(27c)}\\
&\sum_{k\in\mathcal{K}} y_{m,k}\,C_{k,s_{m,k}(t)}\le\bar{C}, && \forall m,t &&\text{(27d)}\\
&s_{m,k}(t)\ge\check{i}_{m,k}, && \forall m,k,t &&\text{(27e)}\\
&s_{m,k}(t)=\chi_{m,k}^{t}N_k+\bigl(1-\chi_{m,k}^{t}\bigr)\hat{s}_{m,k}(t), && \forall m,k,t &&\text{(27f)}\\
&\hat{s}_{m,k}(t)\in\{\check{i}_{m,k},\dots,N_k\}, && \forall m,k,t &&\text{(27g)}\\
&T_j\le T_{k_j}^{\max}, && \forall j\in\mathcal{J} &&\text{(27h)}\\
&\rho_m\le 1, && \forall m\in\mathcal{M} &&\text{(27i)}\\
&(1)\sim(26). && &&\text{(27j)}
\end{aligned}
$$

Here $\mathbf{X}$ is decided by the **user** and $\mathbf{S}$ by the **servers**, on
the two timescales of Section 3.6; $\hat{s}_{m,k}(t)$ denotes the retention level
chosen by the decision in slot $t$ before any service, and
$\chi_{m,k}^{t}=\mathbb{1}\{\exists j:\,t_j=t,\ k_j=k,\ x_{j,m}=1\}$ indicates
whether a type-$k$ task is served by server $m$ in slot $t$.

Constraints (27b)–(27c) route every task to exactly one eligible server; (27d) is the
shared GPU memory budget; (27e) enforces the minimum residency of (14); (27h) is the
end-to-end deadline; (27i) is the stability condition (22); and (27j) collects the
models established above. Constraint (27f) restores the required model to full
residency whenever a task is served, and otherwise adopts the level selected by the
decision, which (27g) permits to be either **lower** than the current level
(unloading) or **higher** (prefetching); a prefetch of duration (20) advances the
occupancy (19), so its opportunity cost is endogenous and needs no penalty term.

**Theorem 1.** *Problem $\mathbf{P}$ is a non-convex NP-hard mixed-integer
nonlinear program (MINLP).*

*Proof.* $\mathbf{P}$ involves binary variables $\mathbf{X}$ and integer variables
$\mathbf{S}$, hence it is a mixed-integer program. The objective (27a) is non-convex,
since the Jain index (26) is a ratio of quadratic forms of $\{L_m(t)\}$, the loading
term (11) is a state-dependent piecewise function of the retention trajectory, and the
waiting term (18) is a maximum of affine functions coupled across tasks through (19).
NP-hardness follows from two independent reductions: the routing subproblem under the
load-balancing objective contains PARTITION, and the single-server retention
subproblem under (5) is a multiple-choice knapsack problem. Both are detailed in
Appendix A. $\blacksquare$

---

## 3.6 Two-Timescale Decision Structure

The two decision groups of $\mathbf{P}$ differ by an order of magnitude in the rate
at which they must be revised, and the formulation reflects this explicitly.

- **Short timescale (task routing, every slot).** A routing decision is triggered by
  an arrival, so $x_{j,m}$ is decided in every slot $t\in\mathcal{T}$ on the basis of
  the instantaneous channel state, server occupancy and layer residency.
- **Long timescale (layer retention, every $\Delta$ slots).** Retention is a
  rent-or-buy decision whose relevant horizon is an idle period, i.e. $O(1/\lambda_k)$,
  which is one to two orders of magnitude longer than a slot. The horizon is
  therefore partitioned into $T_0=\lceil T/\Delta\rceil$ **epochs** of $\Delta$
  consecutive slots, $\mathcal{T}_0=\{1,\dots,t_0,\dots,T_0\}$ with
  $\mathcal{T}(t_0)=\{(t_0-1)\Delta+1,\dots,t_0\Delta\}$, and the retention target is
  revised once per epoch:

$$
\hat{s}_{m,k}(t)=\hat{s}_{m,k}\bigl((t_0-1)\Delta+1\bigr),
\qquad \forall t\in\mathcal{T}(t_0). \tag{28}
$$

Within an epoch the *realized* state $s_{m,k}(t)$ may still change, but only through
(27f): a served model is restored to full residency. A *deliberate* unloading or
prefetching therefore occurs only at an epoch boundary, whereas the residency can
rise spontaneously in between. The separation also matches the rates at which the two
decisions are actionable — revising retention every slot would merely thrash the
PCIe link — and lets the two policies be updated at their own frequencies
(Section IV). The placement indicators $\{y_{m,k}\}$ remain a **given configuration**;
promoting them to a third, still slower timescale is left as future work.

---

# IV. Multi-Agent Reinforcement Learning Formulation

The two decision groups of $\mathbf{P}$ are **bidirectionally coupled**, which is what
makes a learning-based joint policy necessary. On the one hand, the routing strategy
$\mathbf{X}$ determines the effective arrival rate observed by each (server, model)
pair,

$$
\lambda_{m,k}=\lim_{T\to\infty}\frac{1}{T}\sum_{j:\,k_j=k}x_{j,m}, \tag{29}
$$

which dictates the optimal retention trajectory of model $k$ on server $m$. On the
other hand, $s_{m,k}(t)$ determines the loading delay in (11) and, through (19), the
occupancy that subsequent tasks must queue behind — hence whether routing a task to
server $m$ is profitable at all. Furthermore, the arrival instants of future tasks are
unknown at decision time, which makes $\mathbf{P}$ an **online** decision problem, so
that a worst-case online algorithm cannot exploit the predictability of the arrival
process while a learned policy can.

Since $\mathbf{P}$ is a non-convex NP-hard MINLP whose future arrivals are unknown,
we therefore model it as a partially observable Markov game solved under the
centralized-training–decentralized-execution (CTDE) paradigm.

Following the decision structure of Section 3.1.1 and the timescale separation of
Section 3.6, the game involves $1+M$ **heterogeneous agents** operating on **two
timescales**:

- one **user agent**, which observes the state of all servers and routes every
  arriving task, acting once per slot;
- $M$ **server agents**, each of which decides the layer retention of the models it
  hosts, acting once per epoch.

The two groups have different observation spaces, different action spaces and
different decision frequencies, hence they cannot be merged into a homogeneous
Markov game. A single centralized critic is nevertheless shared during training,
so that the coupling described in Section 3.5 is learned rather than assumed.

## 4.1 Observation

**User agent.** In slot $t$ the user observes

$$
o^{\mathrm{u}}(t)=\Bigl\{\tfrac{t}{T},\,F(t{-}1),\,\tfrac{t\bmod\Delta}{\Delta},\
\bigl\{R_m(t),\delta_m,f_m,\beta\!\ell_m(t),u_m(t)\bigr\}_{m},\
\bigl\{\tfrac{s_{m,k}(t)}{N_k},\tfrac{D_{m,k,s_{m,k}}}{T_k^{\max}},\mathbb{1}_{m\in\mathcal{M}_k}\bigr\}_{m,k},\
\bigl\{\hat\lambda_k\bigr\}_{k}\Bigr\}, \tag{30}
$$

where $u_m(t)$ is the memory utilization of server $m$,
$\beta\!\ell_m(t)=\max\{0,\mathfrak{b}_m-t\tau\}$ its queue backlog, and
$\hat\lambda_k$ the sliding-window estimate of the arrival rate of type $k$.

Two components deserve attention. First, the term $D_{m,k,s_{m,k}}/T_k^{\max}$
**writes the server actions into the user observation**: it is the reload delay that
the current residency of $(m,k)$ would incur, normalized by the deadline, and it is
precisely what the user needs in order to anticipate whether routing a task to
server $m$ costs an extra loading. Without it the user cannot distinguish a server
that holds the model resident from one that has unloaded it. Second,
$(t\bmod\Delta)/\Delta$ is the **epoch phase**, which tells the user how soon the
retention states may change.

**Server agent.** At the beginning of epoch $t_0$, server $m$ observes

$$
o_m(t_0)=\Bigl\{\underbrace{\tfrac{t}{T},\,u_m,\,\varrho_m,\,F(t{-}1),\,R_m,\,\delta_m,\,f_m,\,\beta\!\ell_m}_{\text{global part}},\
\underbrace{\bigl\{\tfrac{s_{m,k}}{N_k},\Delta_{m,k},\tfrac{\check i_{m,k}}{N_k},B_{m,k},\hat\lambda_k,\hat\lambda_{m,k},\mathbb{1}_{m\in\mathcal{M}_k}\bigr\}_{k}}_{\text{per-model part}}\Bigr\}, \tag{31}
$$

evaluated at $t=(t_0-1)\Delta+1$, where $\varrho_m$ is the relative load,
$\Delta_{m,k}$ the elapsed idle time of model $k$, and $\hat\lambda_{m,k}$ the
estimate of the *local* arrival rate, i.e. of the arrivals that the user actually
routes to $m$. The latter is the channel through which the user's policy enters the
server's observation, so the coupling of Section 3.5 is visible from both sides.
The global state used by the critic is
$s(t)=\{o^{\mathrm{u}}(t)\}\cup\{o_m\}_{m\in\mathcal{M}}$.

## 4.2 Action

**User agent.** The user outputs a routing preference
$a^{\mathrm{u}}(t)=\{\varsigma_{k,m}(t)\}\in[-1,1]^{K\times M}$ and each arriving
task is assigned to its most preferred *eligible* server,

$$
x_{j,m^{\star}}=1,\qquad
m^{\star}=\arg\max_{m\in\mathcal{M}_{k_j}}\varsigma_{k_j,m}(t_j). \tag{32}
$$

**Server agent.** Server $m$ outputs a retention action
$a_m(t_0)=\{\phi_{m,k}(t_0)\}_{k}\in[-1,1]^{K}$, mapped onto the admissible interval by

$$
\hat{s}_{m,k}=\check{i}_{m,k}+\Bigl\lceil\tfrac{\phi_{m,k}+1}{2}\bigl(N_k-\check{i}_{m,k}\bigr)\Bigr\rceil, \tag{33}
$$

which holds for the whole epoch by (28). Eq. (33) enforces (27e) and (27g) **by
construction**; a target below the current state corresponds to unloading, and one
above it to prefetching, the latter paying (20) and advancing the occupancy (19).

Both discrete decisions are thus handled by a **continuous relaxation** — (32) for
the routing and (33) for the retention level — so that a single class of
continuous-action policy networks suffices for both agent types.

## 4.3 Reward

The reward is split between the two agent types **along the quantities each of them
controls**. The user receives

$$
r^{\mathrm{u}}(t)=-\omega_1\!\!\sum_{j:\,t_j=t}\!\!
\frac{T_j^{\mathrm{wait}}+T_{j}^{\mathrm{tx}}+T_{j}^{\mathrm{cp}}}{T_{k_j}^{\max}}
\;+\;\omega_3F(t)\;-\;\omega_4\,\Xi(t), \tag{34}
$$

and server $m$ receives, at the end of epoch $t_0$,

$$
r_m(t_0)=-\omega_1\!\!\!\sum_{j:\,t_j\in\mathcal{T}(t_0),\,x_{j,m}=1}\!\!\!
\frac{T_{j}^{\mathrm{ld}}}{T_{k_j}^{\max}}
\;-\;\frac{\omega_2}{M\bar{C}}\sum_{t\in\mathcal{T}(t_0)}\sum_{k} y_{m,k}C_{k,s_{m,k}(t)} . \tag{35}
$$

The rationale is that the loading delay $T^{\mathrm{ld}}$ is the *only* latency
component that the retention decision controls, and the memory occupancy is its
direct consequence; conversely, waiting, transmission and computation are fully
determined by *which server was chosen*, and so is the fairness index $F(t)$. The
penalty $\Xi(t)$, which covers violations of (27h) and the absence of an eligible
server, is charged to the user because $\check{i}_{m,k}$ already guarantees through
(13) that the reload of a *pinned* model fits within the deadline budget, so the
residual violations are queueing-induced and therefore routing-induced. Constraints
(27d)–(27g) need no penalty at all, being satisfied by construction through (33).

Summing (34) and (35) over the horizon telescopes the two timescales into the
objective,

$$
\sum_{t\in\mathcal{T}} r^{\mathrm{u}}(t)+\sum_{m\in\mathcal{M}}\sum_{t_0\in\mathcal{T}_0} r_m(t_0)
=-\Bigl(\omega_1\tilde\Psi+\omega_2\tilde\Phi-\omega_3\textstyle\sum_t F(t)
+\omega_4\textstyle\sum_t\Xi(t)\Bigr), \tag{36}
$$

i.e. **maximizing the cumulative team reward is equivalent to minimizing the
objective (27a) augmented with the penalty of the relaxed constraints**; when no
violation occurs, the bracket reduces exactly to (27a). Note that (36) holds for
*any* $\Delta$, because the epoch-wise reward (35) accumulates the per-slot memory
term rather than sampling it. The identity has been verified numerically for
$\Delta\in\{1,10\}$ under three reference policies, with a relative error below
$10^{-6}$.

> **A known externality, stated explicitly.** A prefetch occupies the GPU and
> advances $\mathfrak{b}_m$ by (20), so it lengthens the *waiting* of subsequent
> tasks — a term charged to the user by (34), although it was caused by a server.
> The server still pays for the prefetch indirectly, since the reload it avoids
> appears in its own term (35); but part of the opportunity cost is externalized.
> The shared critic is expected to absorb this, and the implementation additionally
> provides the option of charging the prefetch-induced waiting to the server, which
> is a redistribution *within* (36) and therefore leaves the identity intact.

## 4.4 Analytical Restriction of the Action Space

For a single (server, model) pair with a stationary arrival process, the retention
subproblem reduces to a two-slope ski-rental problem. Under the normalized
objective, staying fully resident costs $\omega_2C_{k,N_k}/(\tau M\bar{C})$ per unit
time, whereas compressing to the minimum residency costs
$\omega_2C_{k,\check{i}_{m,k}}/(\tau M\bar{C})$ per unit time plus a one-off
$\omega_1 D_{m,k,\check{i}_{m,k}}/T_k^{\max}$. Equating the two yields the
break-even instant

$$
B_{m,k}=\frac{\eta_k\,D_{m,k,\check{i}_{m,k}}}{C_{k,N_k}-C_{k,\check{i}_{m,k}}}
=\frac{\eta_k}{\beta_m},
\qquad \eta_k\triangleq\frac{\omega_1}{\omega_2}\cdot\frac{\tau M\bar{C}}{T_k^{\max}} . \tag{37}
$$

Note that $\eta_k$ is **not** a free parameter: it is fully determined by the
objective weights and the normalization constants, since a break-even instant
inconsistent with (27a) would clip away exactly the actions the agent should take.
Its dependence on $1/T_k^{\max}$ has a clear interpretation: **the tighter the
deadline, the more expensive latency becomes relative to memory, and the more the
model should stay resident**. Accordingly, $B_{m,k}<D_{m,k}^{\mathrm{QoS}}$
identifies the (server, model) pairs for which unloading can pay off, whereas the
converse indicates that the model should remain resident — as is the case for small
models with tight deadlines, whose reload is cheap to avoid and whose deadline
tolerates no loading at all.

Both $\check{i}_{m,k}$, acting as a hard lower bound on residency, and $B_{m,k}$,
acting as the break-even instant, are provided to the **server** agents and used to
clip the retention action. This shrinks the effective action space and preserves a
worst-case guarantee inherited from the underlying online algorithm.

---

# Appendix A: Proof of Theorem 1

We detail the two reductions establishing the NP-hardness of $\mathbf{P}$.

**Reduction 1 (routing under load balancing contains PARTITION).**
Consider an instance with $\omega_1=\omega_2=0$, $\omega_3=1$, $M=2$, $f_1=f_2$,
$\omega_1^{\mathrm{bw}}=\omega_2^{\mathrm{bw}}$, in which all models are fully
resident on both servers (so the loading term vanishes) and a single decision epoch
is considered. Then $L_m=\sum_j x_{j,m}T_{j}^{\mathrm{cp}}$ and, by (27b), the total
load $\sum_m L_m=\sum_j T_j^{\mathrm{cp}}\triangleq S$ is a constant independent of
the routing. By (26), $F=1$ holds if and only if $L_1=L_2=S/2$. Consequently,
deciding whether a routing strategy attaining the maximum objective $F=1$ exists is
equivalent to deciding whether the multiset
$\{T_j^{\mathrm{cp}}\}_{j\in\mathcal{J}}$ can be partitioned into two subsets of
equal sum, i.e. the PARTITION problem, which is NP-complete.

**Reduction 2 (retention under shared memory is a multiple-choice knapsack).**
An independent source of hardness arises from the shared-memory coupling (5): for a
single server with a given arrival distribution, choosing a retention level
$s_k\in\{\check{i}_k,\dots,N_k\}$ for every model so as to minimize
$\omega_2\tilde\Phi+\omega_1\mathbb{E}[\text{loading}]$ subject to
$\sum_k C_{k,s_k}\le\bar{C}$ is a multiple-choice knapsack problem, which is also
NP-hard.

Combining the above, $\mathbf{P}$ is a non-convex NP-hard MINLP. $\blacksquare$

---

# 附录（中文，投稿前删除）

## A. 与 TJCCT 结构的对照

| TJCCT | 本文 | 说明 |
|---|---|---|
| 3.1.1 System Overview（空间+时间维度） | 3.1.1 同 | 三层架构 + 时隙 $T/\tau$ + 跨时隙服务需排队 |
| 3.1.2 Basic Models (1)~(5) | 3.1.2 (1)~(5) | 任务/LLM/服务器/层保留/共享显存，实体均用 tuple |
| 3.2 Communication | 3.2 同 | 按你的选择保持简化（不建 LoS/NLoS 概率） |
| 3.3 Computation（本地/边缘，各分 Delay+Energy） | 3.3（加载/推理/占用，各分小点） | **新增 3.3.3 占用与排队**；推理拆 prefill/decode |
| 3.4 Utility Model（QoE/Revenue/System） | 3.4 Cost Model（三项已归一化） | 保持 min 成本，但按 TJCCT 做法**先归一化再加权** |
| 3.5 Problem Formulation + Theorem 1 | 3.5 同 | 决策变量集合化、约束择要解释、Theorem 1 正文只留思路，PARTITION 与多选背包两个归约挪 **Appendix A**（对齐 TJCCT 正文只放 8 行证明的做法） |
| **其核心卖点：双时间尺度** | **3.6 双时标（已转正）** | 用户路由每时隙 / 服务器保留每 $\Delta$ 时隙。与 TJCCT 的差别：他们的长时标是**无人机轨迹**，我们的是**层保留**；他们的短时标是资源分配+卸载，我们的是**路由** |
| TABLE 符号表 | TABLE I: Notations | 已砍到 15 行（2026-08-26）：只留读者猜不到的（双时标 $\mathcal{T}/\mathcal{T}_0,\Delta$、$\omega_m,\beta_m$、$\psi_k$、$C/D$ 累计量、$\check i$、$D^{\mathrm{QoS}}$、$B$、$T^{\max}$、$\mathfrak b_m$、两个决策变量、$y_{m,k}$、目标量）；标准记号一律正文首次出现处定义 |

## B. v4 修复清单（对抗性审查全部 7 项）

| # | 问题 | 严重度 | 修复方式 |
|---|---|---|---|
| **①** | **服务器占用/排队完全未建模** | 🔴 致命 | 新增 **3.3.3 节**：忙至时刻 $\mathfrak{b}_m$、等待时延式(18)、占用更新式(19)、利用率式(22) 与稳定性约束 (27i)。**副产品**：负载均衡从"公平偏好"升级为"稳定性必要条件"（Remark 3）；且卸载不再"容量免费"——重载会与推理抢 GPU |
| **②** | **"近弱远强"权衡不成立**（云需 $W>587$ GFLOP，实际 140） | 🟠 高 | 由 ③ 自动解决：按 token 建模后 7B 任务计算量达 $\sim9000$ GFLOP，云总时延 $1274$ ms < 近端 $3354$ ms。新增回归测试 `tradeoff`：**重任务云最优、轻任务近端最优**，路由非平凡（Remark 2） |
| **③** | **计算时延未区分 prefill/decode** | 🟡 中 | 新增 **3.3.2 节**：prefill 式(15) 算力受限、decode 式(16) 取 $\max$(算力, 显存带宽)。$\psi_k$ 由内存经式(2) 反推（7B 得 $14.73$ GFLOP/token，参数量 $7.36$G，与"7B"相符）。实测三台服务器的 decode 均为**带宽受限**，与 LLM 服务已知结论一致 |
| **④** | **只允许卸载、不预取** | 🟡 中 | 式(27g) 放开为可升可降；预取代价式(20) 并**占用服务器**（(19)），故预取期间到达的请求要排队——机会成本内生，无需额外惩罚项 |
| **⑤** | **QoS 双重形式**（$D^{\mathrm{QoS}}$ 与 $T^{\max}$ 并存） | 🟡 中 | 只保留 $T^{\max}$，$D^{\mathrm{QoS}}$ 由式(12) 推导。**自洽性**：压到 $\check i$ 恰好用满 $98\%\sim100\%$ 的 deadline（Remark 1），超时只可能来自排队。副产品：算力强的服务器加载预算更大 ⇒ $\check i$ 更小（$[19,14,3]$，云最小） |
| **⑥** | **目标三项量纲混加** | 🟡 中 | 3.4 节三项全部无量纲化：$\tilde\Psi$ 按各任务 deadline 归一、$\tilde\Phi$ 按 $M\bar C$ 归一、$F$ 本就无量纲。$\omega$ 变成**可解释的纯权衡系数**（默认全取 1） |
| **⑦** | **$B$ 与 QoS 不一致**（$B=3238>D^{\mathrm{QoS}}_{\max}=2500$） | 🟡 中 | 由 ⑥ 解决：式(37) 重新推导得 $\eta_k=(\omega_1/\omega_2)\tau M\bar C/T^{\max}_k$。现 7B 的 $B=424$ ms $<D^{\mathrm{QoS}}=2146$ ms，卸载确实可能划算。并给出 $B$ vs $D^{\mathrm{QoS}}$ 的**结构性解释**（§4.4 末段）与系统级诊断 |

**前一批（P0/P1/P2 级）已在 v3 修复**：Theorem 1 的 GAP 归约错误 → 改 PARTITION；
$B$ 与 $\omega$ 差 333 倍 → 统一 $\eta$；决策变量改保留状态轨迹；负载 $L_m$ 计入加载时间；
$L_m(t)$ 时间索引；奖励式的惩罚项；$F$ 的 0/0 约定；(19i) 降级为链路可达前提（现并入 §3.2.2 末句）；(27f) 形式化。

## B2. v5.0 变更（2026-08-21 组会，师兄拍板）

| # | 原设定 | 现设定 | 落在哪 |
|---|---|---|---|
| 1 | 服务器**竞价**接任务，$\arg\max v_{m,k}$ | **路由由用户决定**；服务器不竞价、不拒单 | 3.1.1「决策维度」、式(32)；旧式(31) 删除 |
| 2 | 每台服务器一个 agent，兼管路由 + 保留 | **用户与服务器都是独立智能体**（$1+M$ 个，异质） | 第四章开头、4.1、4.2 |
| 3 | 双时标是 3.6 的**可选扩展** | **双时标转正**：用户每时隙、服务器每 $\Delta$ 时隙 | 3.6 重写、式(28)、TABLE I 新增 $\mathcal{T}_0,\Delta$ |
| 4 | 奖励按服务器切，每台一份 | **按"谁控制哪一项"切**：用户背等待+传输+计算+公平+罚，服务器背加载+显存 | 式(34)(35)，恒等式(36) |
| 5 | 硬件同构/异构是待确认项 | **不影响建模**，不再是卡点 | C 表已删除该条 |

> 未变的部分（重要）：成本模型 3.2~3.4 与问题 $\mathbf{P}$（3.5）**一字未改** ——
> $x_{j,m}$ 还是同一个变量，只是**归属**从服务器换成了用户。这说明系统建模层
> 与"谁来决策"是解耦的，也是这次改动能这么快落地的原因。

## C. 仍待师兄确认

| # | 假设 | 位置 | 影响 |
|---|---|---|---|
| **1** | **$\Delta$ 取多大**（现默认 10 时隙 = 1 s） | 式(28) | 实测：对"全保留"策略完全无影响；对"立刻全卸"策略从 $\Delta{=}1$ 到 $50$ 使 $J$ 从 $11943$ 降到 $7650$（显存 $0.297\!\to\!0.544$、加载 $119564\!\to\!84004$ ms）。**但那是坏策略被放慢的假象**，真实影响要等学出来的策略 |
| **2** | **奖励分摊是否认可**（式(34)(35)） | 4.3 | 已知外部性：预取抬高的**等待**记在用户账上（代码留了 `charge_prefetch_wait_to_server` 开关做重分配，恒等式不破） |
| **3** | **"简单规则辅助"用在哪一侧** | 4.4 | 师兄提到可引入简单规则；若用在服务器侧，$\check i/B$ 的解析约束正好是现成的规则 |
| ~~4~~ | ~~共享显存约束要不要人为收紧~~ → **已自行决定：维持 $\bar C=24000$ MB** | 式(5)(6)、§V | 师兄说硬件同构异构不影响建模 ⇒ 这条归我们自己定。**决定（2026-08-22）：主算例保持不绑定**（全常驻 18138 MB $<$ 24000 MB）—— 主结果跑在"显存不是瓶颈"的配置上，才能把时延/公平的效应与显存挤压分开；转而把 $\bar C$ 列为 **§V 的一条敏感性轴**，扫绑定窗口 $16298\sim18138$ MB。约束本身建模完整（`infeasible`/`memcap` 两项测试覆盖）|
| 5 | decode 的显存带宽模型取 $\max$ 形式 | 式(16) | 更精细可引入 KV-cache 随上下文增长的影响 |
| 6 | 排队为单服务器 FIFO、传输与计算流水 | 3.3.3 | 若 GPU 支持批处理/多流，需改为多服务台或批调度 |
| 7 | 权重默认 $\omega_1=\omega_2=\omega_3=1$ | 式(27a) | 归一化后已可解释；正式实验建议做敏感性分析 |

> **已由 8-21 组会解决、不再是待确认项**：竞价机制（→ 用户路由）、多智能体用在哪
> （→ 用户 + 服务器双方都是）、$M{=}3$ 要不要 MARL（→ 用户端 DRL + 服务器端 MADRL）、
> 放置变量 $y$ 是否升级为长时标决策（→ 保持给定，长时标给了层保留）、硬件同构性。

## D. 关键数据（Remark 1~3 的实测来源）

**7B 任务（512→128 token）时延分解（ms，无排队）**

| 服务器 | 传输 | 加载（全常驻 / 压到 $\check i$） | prefill | decode | 总计（全常驻） | /deadline |
|---|---|---|---|---|---|---|
| edge-near | 2.2 | 0 / 2135 | 1257 | 2095 | **3354** | 0.610 |
| edge-mid | 16.3 | 0 / 2760 | 628 | 2095 | 2739 | 0.498 |
| cloud | 80.4 | 0 / 4135 | 251 | 943 | **1274** | 0.232 |

**参考策略对比（400 时隙，$\Delta=10$；路由策略现在写在用户动作里）**

| 策略 | 超时 | 平均时延 (ms) | 最大 $\rho_m$ | Jain | 显存 | 目标 $J$ |
|---|---|---|---|---|---|---|
| keep+nearest | 181 | 35222 | **2.738** ⚠发散 | 0.345 | 0.756 | 33042 |
| keep+roundrobin | 107 | 2451 | 0.861 | 0.767 | 0.756 | 1902 |
| **keep+leastbacklog** | **45** | **647** | 0.769 | **0.905** | 0.756 | **305** |
| unload+roundrobin | 175 | 14354 | 1.712 ⚠ | 0.891 | **0.347** | 12886 |
| unload+leastbacklog | 180 | 12447 | 1.594 ⚠ | 0.962 | **0.350** | 10956 |

> 三个 keep 行与 v4 **完全一致**（全保留策略不择时，$\Delta$ 对它无影响）；
> 两个 unload 行因 $\Delta=10$ 而略有改善（少压回几次 ⇒ 少重载几次）。
> 令 $\Delta=1$ 可**精确复现** v4 的 $13540$ ms / $1.713$ / $0.297$ / $11943$ ——
> 这正是"$\Delta=1$ 退化为单时标"的数值证据。

**奖励分账（同上 400 时隙）**

| 策略 | 用户 | 服务器合计 | 团队和 | $-J$ | 误差 |
|---|---|---|---|---|---|
| keep+nearest | $-32740.07$ | $-302.30$ | $-33042.37$ | $-33042.37$ | $1.5\times10^{-11}$ |
| keep+leastbacklog | $-2.90$ | $-302.30$ | $-305.20$ | $-305.20$ | $6.3\times10^{-13}$ |
| unload+leastbacklog | $-10742.00$ | $-214.39$ | $-10956.39$ | $-10956.39$ | $2.9\times10^{-11}$ |

> keep 系的服务器奖励只剩显存项（全常驻 ⇒ 加载为 0）；unload 系把成本从"显存"
> 搬到"加载" —— 这正是 rent-or-buy 的两端，而分账让两端各自记在该负责的 agent 头上。

> 参数：Qwen-7B（34 层 / 14728 MB，来自基础论文 TABLE III 实测）、$\bar C=24000$ MB、
> $f=6/12/30$ GFLOP/ms、$\omega=900/900/2000$ MB/ms、$\ell=100/250/400$ m、
> $\delta=1/8/40$ ms、$\tau=100$ ms、$T^{\max}=5500/900/120$ ms。
> 系统总负载 $\approx256$ ms 工作 / $100$ ms 时隙 ⇒ 至少需 $2.56$ 台服务器（现有 3 台），
> 因此**负载均衡是稳定性的必要条件**。全部由 `edge-unload-repro\` 实测导出（19 项验证通过）。

## E. v4.1 收尾修订（5 项遗留小问题）

| # | 问题 | 位置 | 修改 |
|---|---|---|---|
| 1 | TABLE I 中 $s_{m,k}(t)$ 出现两次 | TABLE I | 删去纯描述行，只保留 **Decision** 行，并把"可降（卸载）/ 可升（预取）"的说明并入该行，信息不丢 |
| 2 | $T_{k,m}^{\mathrm{tx}}$ 在式(12) 中使用但未显式定义 | §3.2.2 | 式(10) 后新增式(10$'$)：明确 (10) 是**任务级**（用实际负载与瞬时速率、到达后才可得），而(12) 必须**到达前**定标，故另定义**类型级名义值**（名义负载 + 均值速率）。并说明 $T^{\mathrm{cp}}_{k,m}$ 同约定、二者**仅**用于 QoS 预算，(21)/(27h) 一律用任务级值。附误差量级：文本任务 $\delta_m$ 主导（$2.2/16.3/80.4$ ms），用均值速率代替瞬时速率对预算的扰动 $<0.01\%$，不改变 $\check i$ |
| 3 | $\rho_m\le1$ 是**必要**而非充分条件 | 式(22)、(27i) | 两处均改写：明确"work-conserving 单服务台队列、长期积压有界"的含义，并显式说明**不充分**（突发到达下 $\rho_m<1$ 仍可能长期瞬态增长；严格随机稳定还需 $\rho_m$ 远离 1），故只作为必要条件施加，实测 $\rho_m$ 在 §V 与时延一并汇报 |
| 4 | $\omega_m$ 等参数无参考量级 | TABLE I | TABLE I 加第三列**Typical value**，并把 $f_m,\omega_m$ 拆成两行独立给值。$\omega_m=900/900/2000$ MB/ms，即 $0.9\sim2.0$ TB/s（RTX 3090 ~ A100-80GB 量级）。表下加脚注：该列取自 §V，仅示物理量级，模型本身不限定取值 |
| 5 | 预取如何被触发未明写 | (27f) 说明段 | 补一句："当 $\hat s_{m,k}(t)>s_{m,k}(t-1)$ 时产生一次时长为式(20) 的预取，推进占用式(19)"，并接上"预取期间到达的请求需排队 ⇒ 机会成本内生、无需额外惩罚项" |

> 全部 Typical value 已用 `edge-unload-repro/` 逐项核对（$M{,}K{=}3$、$\tau{=}100$ ms、
> $\bar C{=}24000$ MB、$f{=}6/12/30$、$\omega{=}900/900/2000$、$\beta{=}3.088$、
> $\delta{=}1/8/40$ ms、$\ell{=}100/250/400$ m、$R{=}225.8/146.7/106.6$ Mbit/s、
> $\lambda{=}0.06/0.15/0.30$、$T^{\max}{=}5500/900/120$ ms、$N_k{=}34/26/8$、
> $C_{k,N_k}{=}14728/3140/270$ MB、$\psi_k{=}14.73/3.14/0.27$ GFLOP/token、
> $\check i_{7B}{=}19/14/3$、$D^{\mathrm{QoS}}_{7B,m=1}{=}2146.4$ ms、$B_{7B}{=}423.9$ ms）。
> 本次修订**未改动任何公式语义**，故代码无需改动（v4.1 时）；v5.0 的改动已同步到代码，19 项验证全部通过。
