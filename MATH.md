# RakshyaNet Mathematical Model

This is the authoritative mathematical description of the current RakshyaNet
implementation. It documents what the backend computes today, what Gemma is
allowed to contribute, how values shown in the UI should be interpreted, and
which claims the project must not make.

Code is the final source of runtime truth. The principal implementations are:

- `backend/services/gemma_service.py`
- `backend/services/optimization_service.py`
- `backend/algorithms/urgency_calculator.py`
- `backend/algorithms/vrp_solver.py`
- `backend/algorithms/nash_solver.py`
- `backend/algorithms/social_welfare_optimizer.py`
- `backend/algorithms/kkt_verifier.py`
- `backend/algorithms/state_manager.py`

## 1. Symbols and units

| Symbol | Meaning | Unit or range |
|---|---|---|
| \(v\) | village or incident location | identifier |
| \(r\) | resource type | identifier |
| \(a\) | fleet asset | identifier |
| \(t\) | elapsed incident time | hours |
| \(N_{vr}\) | current reported need at village \(v\) | resource-native unit |
| \(A^{existing}_{vr}\) | quantity already allocated or present | resource-native unit |
| \(U_{vr}\) | unmet need | resource-native unit |
| \(M_{vr}\) | minimum survival threshold | resource-native unit |
| \(m_r\) | configured urgency multiplier | dimensionless |
| \(C_a\) | asset mass capacity | kg |
| \(S_r\) | available depot stock | resource-native unit |
| \(w_r\) | resource mass per unit | kg per resource unit |
| \(d\) | route distance | km |
| \(s_a\) | asset speed | km/h |
| \(f_a\) | asset fuel endurance | hours |
| \(q\) | normalized Gemma-supported risk score | \([0,1]\) |
| \(c_{sys}\) | deterministic system confidence | \([0,1]\) |

Quantities from different resource types are never validly added as if they
share one physical unit. For example, liters of water, medicine kits, and
food packages remain separate constraints. A dimensionless score may combine
their normalized coverage ratios.

## 2. Evidence-to-math boundary

Gemma does not calculate routes, allocate stock, authorize a plan, or dispatch
vehicles. Gemma produces a schema-validated analysis whose normalized,
evidence-cited fields may influence deterministic urgency.

### 2.1 Supported input signal

The optimization service considers three structured Gemma outputs:

- severity expected value;
- medical urgency value;
- accessibility risk value.

Unknown values are ignored. The bounded signal is:

\[
q = \max(q_{severity}, q_{medical}, q_{accessibility})
\]

where only non-null supported values participate and the default is \(0\).

### 2.2 System confidence

System confidence is computed from evidence metadata and model-output
limitations, separately from any model-reported confidence:

\[
\begin{aligned}
c_{sys} = \operatorname{clamp}_{[0,1]}(&
0.55\bar{R} +
0.25D +
0.20F \\
&- P_{contradiction}
- P_{missing})
\end{aligned}
\]

with:

\[
\bar{R} = \frac{1}{n}\sum_i reliability_i
\]

\[
D = \min\left(1,\frac{\text{number of distinct source categories}}{3}\right)
\]

\[
F = \frac{1}{n}\sum_i
\max\left(0,1-\frac{\min(freshnessMinutes_i,1440)}{1440}\right)
\]

\[
P_{contradiction}=\min(0.30,0.08\,n_{contradictions})
\]

\[
P_{missing}=\min(0.25,0.04\,n_{missing})
\]

The implementation rounds \(c_{sys}\) to four decimal places.

### 2.3 External urgency boost

\[
B = \operatorname{round}(q \times c_{sys},4)
\]

The boost is applied only to villages whose configured ID or name is
explicitly mentioned in the submitted evidence text. A valid score without an
explicit location match does not silently alter every village.

Example:

\[
q=0.8800,\quad c_{sys}=0.8742
\]

\[
B=\operatorname{round}(0.8800\times0.8742,4)=0.7693
\]

This value is a bounded prioritization input, not a probability that a disaster
occurred and not an authorization score.

## 3. Urgency model

### 3.1 Unmet need

\[
U_{vr}=\max(0,N_{vr}-A^{existing}_{vr})
\]

The normalized unmet ratio is:

\[
\rho_{vr} =
\begin{cases}
\operatorname{clamp}_{[0,1]}(U_{vr}/N_{vr}), & N_{vr}>0\\
0, & N_{vr}=0
\end{cases}
\]

### 3.2 Time escalation

\[
T(t)=1+0.5\left(e^{0.3t}-1\right),\quad t\ge0
\]

Selected exact approximations are:

| Elapsed time | \(T(t)\) |
|---:|---:|
| 0 h | 1.000 |
| 2 h | 1.411 |
| 4 h | 2.160 |
| 8 h | 6.012 |

The function is intentionally escalating and currently has no upper cap.
Therefore, very large elapsed-time values can dominate the score. Inputs must
represent the intended operational horizon, and the UI must show the elapsed
time beside the result.

### 3.3 Per-resource contribution

\[
u_{vr}=\rho_{vr}\,m_r\,T(t)
\]

### 3.4 Critical-shortage penalty

A village has a critical shortage when any resource allocation is below its
configured minimum survival threshold. The fixed penalty is:

\[
P_v =
\begin{cases}
10, & \exists r:A^{existing}_{vr}<M_{vr}\\
0, & \text{otherwise}
\end{cases}
\]

### 3.5 Total village urgency

\[
Urgency_v=\sum_r u_{vr}+P_v+B_v
\]

Villages are sorted in descending order; rank 1 is most urgent. The API exposes
the component values, including current need, existing allocation, unmet need,
ratio, multiplier, time factor, critical status, and Gemma boost. A UI label
must not show only the final number without this explanation.

The urgency score is an ordering heuristic. It is not a percentage, mortality
forecast, or calibrated probability.

## 4. Deterministic fleet and route solver

The current solver is a deterministic greedy, capacity-constrained routing
heuristic. It is not an exact mixed-integer vehicle-routing optimizer.

### 4.0 Complexity class and the heuristic choice

The dispatch problem is a **capacitated vehicle-routing problem** with a
heterogeneous fleet, capability constraints, payload limits, fuel endurance and
time-escalating priority. This problem is **NP-hard** — it contains the
travelling salesman problem, whose decision form is NP-complete, and the
assignment of cargo across non-interchangeable assets sits above it. No
polynomial-time algorithm is known that returns a provably optimal plan, and
unless \(P = NP\) none exists.

Consequently an exact solve is **not the appropriate engineering choice** for a
real-time dispatch tool at this instance size. The implemented method is a
greedy urgency-ordered assignment (§4.4–§4.5) followed by a nearest-neighbour
tour over a capability- and closure-filtered graph (§4.2). It is a **heuristic**,
and describing it as one is a statement of the design, not an apology for it.

The mathematical claim attached to a produced plan is therefore:

\[
\text{plan is feasible} \;\land\; \text{plan is deterministic}
\;\land\; \text{plan is traceable}
\]

and explicitly **not**

\[
Objective_{route}(\text{plan}) = \max_{\text{plans}} Objective_{route}
\]

Two precise qualifications:

- **Where the hardness lies.** With at most two village stops per asset (§4.4),
  intra-route stop ordering is trivially enumerable. The combinatorial
  difficulty is the partition of locations and cargo across assets. The
  two-stop cap is itself a heuristic restriction of the feasible search space.
- **What is unproved.** The greedy result carries no optimality certificate, no
  approximation ratio, and no measured optimality gap. §5–§7 concern a separate
  *continuous* allocation problem and say nothing about these discrete
  decisions.

**Roadmap.** The bundled instance is small enough to encode exactly as a MILP or
CP-SAT model under the same constraints and the same coverage objective of
§4.7. Solving that model would permit reporting
\((Objective^{greedy} - Objective^{exact})/Objective^{exact}\) as a genuine
optimality gap. That measurement has not been performed, so the correct
statement today is **unmeasured**, not *small*.

### 4.1 Air distance

Aircraft use the Haversine great-circle distance:

\[
a=\sin^2\left(\frac{\Delta\phi}{2}\right)
+\cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)
\]

\[
d=2R\operatorname{atan2}(\sqrt{a},\sqrt{1-a}),\quad R=6371\text{ km}
\]

An aircraft leg is represented as a direct geodesic corridor. Road closures do
not constrain it.

### 4.2 Road graph

Ground assets use Dijkstra's shortest-path algorithm over the deterministic
road graph.

An edge is removed before search when:

- its ID is in the active blocked-edge set;
- it has no road;
- its road quality is incompatible with the asset's terrain capability.

Search cost for edge \(e\) is:

\[
cost_e=d_e\left(1+0.06\max(0,\tau_e-1)\right)
\]

where \(d_e\) is edge distance and \(\tau_e\) is terrain difficulty.

The weighted cost chooses the path. The route's reported physical distance is
the sum of raw edge distances, not the weighted search cost.

When a closure arrives, a new child optimization run is created. The original
run remains immutable history. A ground route is accepted only if no selected
leg contains an active blocked edge.

### 4.3 Travel time and ETA

For a leg:

\[
timeMinutes=\frac{d}{s_a}\times60
\]

Stop ETA is the cumulative time of all preceding outbound legs:

\[
ETA_k=\sum_{j=1}^{k}timeMinutes_j
\]

Total route time includes the return leg to the depot. A route is fuel-feasible
when:

\[
totalTimeMinutes\le60f_a
\]

The frontend map advances an asset along each returned route geometry using
the same cumulative stop ETAs. Ground assets follow road-leg geometry;
aircraft follow air legs. Decorative movement must not replace solver geometry.

### 4.4 Capacity and compatibility

For each asset:

\[
\sum_r x_{ar}w_r\le C_a
\]

Assignments also enforce:

- terrain capability;
- road reachability for ground assets;
- depot stock availability;
- village unmet need;
- at most two village stops per asset.

The current allocation pass processes villages by descending urgency. Within a
village it processes critical resources first, then higher configured urgency
multipliers. Configured `preferred_resources` are a soft specialty, not a hard
prohibition.

### 4.5 Asset-selection score

Every viable asset receives an auditable score before cargo is committed.
Let \(t_a\) be its direct one-way travel time to the village, \(C_a^{rem}\) its
remaining payload mass, and \(q_rw_r\) the requested payload mass:

\[
E_a=\frac{1}{1+t_a/120}
\]

\[
P_a=\min\left(1,\frac{C_a^{rem}}{\max(q_rw_r,1)}\right)
\]

The time-pressure weight is:

\[
T=\min\left(0.90,\,
0.12+0.20S+0.18I+0.15U+0.20M+0.15G\right)
\]

where:

- \(S\) is 1 for a critical survival shortage, otherwise 0;
- \(I\) is normalized incident impact;
- \(U=\min(1,Urgency_v/15)\);
- \(M\) is 1 for medical, safety, or communication resources;
- \(G=\min(1,External_v)\), the bounded Gemma contribution.

The final candidate score is:

\[
Score_a=\min\left(1,\,
TE_a+(1-T)P_a+B_{specialty}+B_{consolidation}+B_{mode}\right)
\]

with:

\[
B_{specialty}=
\begin{cases}
0.08,&\text{asset prefers resource}\\
0,&\text{otherwise}
\end{cases}
\]

\[
B_{consolidation}=
\begin{cases}
0.08,&\text{village is already on the asset tour}\\
0,&\text{otherwise}
\end{cases}
\]

\[
B_{mode}=
\begin{cases}
0.07T,&\text{aircraft}\\
0.07(1-T),&\text{ground asset}
\end{cases}
\]

This makes high-impact, time-sensitive medical shortages favor faster aircraft
when feasible, while large lower-time-pressure loads favor ground payload
capacity. Before scoring, the solver constructs the projected complete
depot-to-stops-to-depot tour and rejects the asset if any leg is unreachable or
the full tour exceeds fuel endurance. The API exposes every score component,
direct ETA, projected tour time, selected quantity, and payload mass.

### 4.6 Route-level feasibility

A proposed run is considered route-feasible only when:

\[
routeFeasible=(|Routes|>0)\land\bigwedge_{route}route.feasible
\]

Backend approval rejects a run unless `route_feasible` is exactly true.

### 4.7 Routing objective shown by the API

For each village-resource pair, coverage counts existing field allocation plus
new route allocation:

\[
coverage_{vr}=
\begin{cases}
\min\left(1,\frac{A^{existing}_{vr}+A^{new}_{vr}}{N_{vr}}\right), & N_{vr}>0\\
1, & N_{vr}=0
\end{cases}
\]

The VRP objective is the mean coverage ratio:

\[
Objective_{route}=\frac{1}{K}\sum_{v,r}coverage_{vr}
\]

where \(K\) is the number of evaluated village-resource pairs. It is
dimensionless and normally lies in \([0,1]\).

This is the objective value **achieved** by the greedy plan. It is not a
maximum, not a bound, and not evidence that no better plan exists (§4.0).

## 5. Capped proportional allocation

Historical class and response names include `NashSolver` and
`NashEquilibrium`. They are retained for API compatibility. The implemented
method is deterministic capped proportional allocation, not a strategic game
and not a proof of Nash equilibrium.

For one resource \(r\):

\[
weight_{vr}=N_{vr}m_r
\]

\[
share_{vr}=
\frac{weight_{vr}}{\sum_j weight_{jr}}S_r
\]

\[
x_{vr}=\min(share_{vr},N_{vr})
\]

Surplus from a village that reaches its need cap is redistributed
proportionally among still-active villages. This repeats for at most
\(|V|+2\) cap-redistribution passes per resource.

The reported utility is:

\[
Utility_v=\sum_r
\frac{\min(x_{vr},N_{vr})}{N_{vr}}m_r
\]

and:

\[
TotalUtility=\sum_v Utility_v
\]

### 5.1 Convergence metric

The proportional rule is initialized from the greedy routing allocation and
reapplied until the deterministic fixed point stabilizes. For every
village-resource pair:

\[
\delta_{vr}^{norm}=
\frac{|x_{vr}^{new}-x_{vr}^{old}|}
{\max(N_{vr}^{new},N_{vr}^{old},S_r,1)}
\]

\[
\epsilon_{norm}=\max_{v,r}\delta_{vr}^{norm}
\]

Convergence is:

\[
\epsilon_{norm}<0.01
\]

The raw maximum change is separately returned as
`max_strategy_change`; it can carry resource-native units and must not be
compared across heterogeneous resource types. The chart's authoritative
series is `max_normalized_change`. If a logarithmic chart replaces exact zero
with a small visual floor, that floor is presentation-only and must be
identified as such.

## 6. Weighted Nash social-welfare comparison

This is a continuous fairness-aware allocation comparison, also called Nash
bargaining or Nash social welfare. It is not a strategic Nash equilibrium and
does not include vehicles or route feasibility.

Decision variable:

\[
x_{vr}\in[0,U_{vr}]
\]

For each village, resource urgency multipliers are normalized over its positive
unmet needs:

\[
\hat{m}_{vr}=\frac{m_r}{\sum_{j\in R_v}m_j}
\]

Village coverage is:

\[
c_v=\operatorname{clamp}_{[0,1]}
\left(\sum_{r\in R_v}\hat{m}_{vr}\frac{x_{vr}}{U_{vr}}\right)
\]

Urgency weights are normalized to mean one across positive village urgency
scores:

\[
\alpha_v=\frac{Urgency_v}{mean(Urgency_{positive})}
\]

The optimizer maximizes:

\[
\max_x\sum_v\alpha_v\log(10^{-6}+c_v)
\]

subject to:

\[
\sum_vx_{vr}\le S_r\quad\forall r
\]

\[
0\le x_{vr}\le U_{vr}\quad\forall v,r
\]

SciPy SLSQP solves the continuous problem, warm-started by the proportional
candidate. The response includes solver status, iterations, objective,
coverage, stock use, and maximum constraint violation.

The method comparison evaluates the proportional and optimized candidates
against the same continuous social-welfare objective. An objective improvement
does not imply that an equivalent route or fleet schedule exists.

## 7. KKT diagnostics

The KKT panel evaluates feasibility and partial consistency of the submitted
capped-proportional continuous allocation. It does not independently prove
global optimality and does not apply to discrete routes.

Scope, stated before the algebra: the allocation problem examined here is
**continuous and convex**, and is a different problem from the **discrete,
NP-hard** routing problem of §4.0. Nothing on this panel constrains, validates,
or certifies a route.

The diagnostic utility is:

\[
f(x)=\sum_v\sum_r\frac{x_{vr}}{N_{vr}}m_r
\]

with constraints:

\[
\sum_vx_{vr}\le S_r,\quad x_{vr}\le N_{vr},\quad x_{vr}\ge0
\]

The verifier reports:

1. stationarity arithmetic consistency;
2. primal feasibility;
3. non-negative estimated dual multipliers;
4. complementary slackness for aggregate resource capacity.

For a tight resource, its estimated aggregate multiplier is:

\[
\lambda_r=
\frac{\sum_v(\partial f/\partial x_{vr})x_{vr}}
{\sum_vx_{vr}}
\]

For a slack resource, the implementation sets \(\lambda_r=0\).

**Three of the four conditions are satisfied by construction, not by
optimality.** This is stated plainly because it is the first thing a reader
should test:

- **Stationarity.** Slack resources are skipped entirely. For a tight resource
  the residual is \(\bigl|\sum_v(\partial f/\partial x_{vr})x_{vr}-\lambda_r\sum_v x_{vr}\bigr|\),
  and \(\lambda_r\) was *defined* above as exactly that ratio. The residual is
  algebraically zero for every input.
- **Dual feasibility.** \(\lambda_r\) is a non-negatively-weighted mean of
  non-negative terms, so \(\lambda_r\ge 0\) always.
- **Complementary slackness.** \(\lambda_r\) is set to zero precisely when the
  resource is slack, so the product \(\lambda_r s_r\) is always zero.

Only **primal feasibility** — allocations non-negative, within need, and within
depot stock — carries information. A zero allocation that delivers nothing to
anyone therefore passes all four conditions.

The panel is a **feasibility and consistency check**, not a certificate of
optimality, and it is a diagnostic on the *continuous* allocation only: it says
nothing about the discrete routing decisions. `KKTVerificationResult` ships
`independently_proves_optimality = false` and
`applies_to_discrete_route_decisions = false` in the API payload so that this
cannot be misread downstream. The UI must never call it a proof.

## 8. Human-review invariants

An approval or rejection request must match:

- the run ID;
- the immutable Gemma `analysis_id`;
- the operator's `expected_updated_at` snapshot;
- the latest optimization run.

Approval additionally requires route feasibility. Any new evidence, road
closure, or reopened road produces a new analysis or optimization snapshot and
therefore invalidates stale approval context.

These are state-integrity constraints, not scoring formulas, but they are part
of the decision system's correctness.

## 9. Display contract for the Math Engine UI

The UI should expose, on demand rather than all at once:

- Gemma-supported input values and their evidence IDs;
- which supported input became the maximum signal;
- system-confidence terms and resulting boost;
- urgency formula and per-resource substitutions;
- depot stock, reported demand, existing allocation, survival threshold, new
  assignment, unmet remainder, and units;
- chosen route, leg mode, road edge IDs, closure avoidance, ETA, and fuel limit;
- proportional-allocation iterations and normalized convergence series;
- social-welfare solver status and comparison scope;
- all four KKT conditions and their explicit limitation;
- approval blockers.

The interface must distinguish `unknown`, zero, unavailable, and not
applicable. It must never invent a missing numeric value to keep a chart full.

## 10. Known mathematical limitations

- Time escalation is uncapped and can become very large for long incident
  horizons.
- The routing problem is **NP-hard** (§4.0). The solver is a greedy,
  deterministic heuristic chosen for that reason; it does not establish global
  optimality, carries no approximation ratio, and its optimality gap against an
  exact solve is **unmeasured**.
- Ground travel times use configured average asset speed, not live traffic.
- The road graph and fleet data are bundled hackathon fixtures unless replaced
  with an explicitly labelled external source.
- The proportional allocation fixed-point is not a Nash equilibrium.
- Weighted Nash social welfare is continuous and ignores vehicle integrality,
  payload packing order, and route feasibility.
- KKT results are scoped consistency checks, not an independent optimality
  proof.
- Gemma scores depend on submitted evidence quality and schema-grounding
  checks; they are not calibrated disaster probabilities.

Any presentation, paper, or demo narration should use these exact limitations
instead of claiming autonomous dispatch, perfect routing, live intelligence, or
mathematically proven global optimality.
