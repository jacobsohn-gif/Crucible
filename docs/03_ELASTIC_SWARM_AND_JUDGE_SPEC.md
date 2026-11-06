# `03` — Elastic Stigmergic Swarm & Calibrated Judge Specification

> Companion to [MASTER_IMPLEMENTATION_PLAN.md](../MASTER_IMPLEMENTATION_PLAN.md). Grounded in empirical harness teardowns (`10_ARCHITECT_IMPLEMENTER_JUDGE_TRIAD_AND_FUZZY_EVALUATORS.md`, `11_SWARM_OF_SWARMS_TRIADIC_POD_ARCHITECTURE.md`, and `12_ADAPTIVE_FLUID_SWARM_ARCHITECTURE.md`).

---

## 1. Core Swarm Philosophy: "Fluid in the Worktree, Strict at the Merge Gate"

`USH` avoids both **chaotic group-chat swarms** (anchoring groupthink + shared-repo clobbering) and **rigid airgapped waterfalls** (ivory-tower specs + `'out-of-scope'` file handcuffs + 66% idle pods) by enforcing **two non-negotiable invariants strictly at promotion/merge time** while keeping exploration and implementation fluid:

1. **`INV-SWARM-1` (Zero Self-Certification)**: Any agent may dynamically rotate across `Scout`, `Spike Architect`, `Worktree Builder`, `Integrator`, and `Judge` roles across waves—**never `Builder` and `Judge` on the same candidate artifact** (`judge.worker_id not in candidate.contributor_ids`), and whenever $\ge 2$ model families are configured, **`judge.model_family != candidate.primary_model_family`**.
2. **`INV-SWARM-2` (Uncertainty-Driven Escalation)**: Default lean (`Gear 1` or `Gear 2`), and **upshift automatically** into parallel prototype spikes, sharded/competitive Git worktrees, and multi-judge tournaments when objective runtime triggers fire.

---

## 2. The 4-Gear Automatic Gearbox & Live Upshift Triggers

| Gear | Default Routing Signal | Active Swarm Topology | Runtime Auto-Upshift Trigger |
| :--- | :--- | :--- | :--- |
| **Gear 1: `Solo Direct`** | Single-file / deterministic command / clear local fix (`complexity < 0.25`) | `1` Builder + **Stage A** `$0` deterministic check | `2` consecutive failed verify runs or touch of auth/crypto/schema/prod config $\to$ **Gear 2** |
| **Gear 2: `Strike Pair`** *(Default Workhorse)* | Standard 2–6 file feature, refactor, or bug (`0.25 <= complexity < 0.55`) | `1` Optimistic Worktree Builder (`Model A`) + **Stage B** `1` Blind Cross-Family Judge (`Model B`) | Popperian `thrash_streak >= 2`, Builder emits `HIGH_UNCERTAINTY`, or Stage B Judge rejects structural design $\to$ **Gear 3** |
| **Gear 3: `Spike + Sharded Swarm`** | Cross-module build (`>5` files) or uncertain API/perf constraints (`0.55 <= complexity < 0.80`) | `2–3` **Executable Spikes** (`<=8` turns in Shadow Git) $\to$ Spec Compiler promotes best spike + spawns $M$ soft-scoped parallel worktrees sharing the Stigmergic Blackboard $\to$ **Stage B/C** Judge | Spikes hit fundamental algorithmic tradeoff, `<30%` pass-rate bug stalls shards, or fuzzy non-deterministic goal $\to$ **Gear 4** |
| **Gear 4: `Evolutionary Crucible`** | Greenfield architecture, kernel/perf discovery, security-critical, or open-ended fuzzy evaluation (`>= 0.80`) | **Pod 1**: `3` Blind Orthogonal Lenses (`Systems`, `Occam/Minimalist`, `Adversarial`) + `1` Spec Compiler (emits Public Spec + **Sealed Holdout Rubric**) $\to$ **Pod 2**: $P \in \{3..5\}$ heterogeneous model Islands + Dedicated Repro/Test Worker + **Round-2 Genetic Crossover** $\to$ **Pod 3**: `4`-Judge Blind Jury + Swapped Bradley-Terry Tournament | Terminal tie ($\hat{\beta}_1 \approx \hat{\beta}_2$ within $1\sigma$) escalates a crisp 2-option `/grill-me` trade-off card to the Human. |

---

## 3. Stigmergic Blackboard & Dual Deployment Backends

Workers never sit in a noisy synchronous chat loop. They coordinate asynchronously via typed, append-only records on the **Stigmergic Blackboard**:

### Canonical Blackboard Event Types
- `TASK_CLAIMED { task_id, worker_id, model_id, worktree_path, lease_until_ms, primary_scope[] }`
- `SPIKE_COMPLETED { spike_id, hypothesis, compilable: bool, benchmark_metrics, promoted_seed_commit? }`
- `CONTRACT_DELTA { worker_id, touched_shared_file, symbol, three_line_summary }` *(injected into sibling workers' next `<dynamic_instructions>` tail so cross-module changes adapt without blocking!)*
- `HYPOTHESIS_FALSIFIED { scope, attempted_approach, exact_failure_evidence }` *(prevents sibling islands from repeating a proven dead end)*
- `CANDIDATE_SUBMITTED { candidate_id, base_sha, head_sha, behavior_fingerprint, contributor_ids[], primary_model_family }` *(author `<thinking>` excluded)*
- `JUDGE_VERDICT { candidate_id, judge_id, judge_family, stage, hard_vetoes{}, bars_scores{}, citations[], confidence, pass: bool }`
- `CONSENSUS_RATIFIED { winning_candidate_id, merged_sha, prev_ledger_sha256, entry_sha256 }`

### Zero-Friction Pluggable Transport Drivers
1. **`LocalWorktreeSwarmDriver` (1 Laptop / Single Workstation)**:
   - Backed by a single SQLite WAL database (`.ush/swarm.db` for atomic task leases + SHA-256 consensus chain) + `.ush/blackboard.jsonl` + `.ush/worktrees/worker-{1..N}`. Zero servers or Docker required to run a 6-agent multi-model swarm locally!
2. **`ClusterSwarmDriver` (Distributed Containers — Remote Swarm Cluster Compatible)**:
   - Maps directly onto containerized worker fleets (shared blackboard mount, NDJSON/HTTP host bridge, and Linux container isolation), permitting 12–50+ headless workers to share tasks, worktree bundles, and cross-family reviews seamlessly.

---

## 4. Heterogeneous Model Synergy Matrix

Because `USH` runs Tier-S, Tier-A, and Tier-B models behind one normalized interface, a mixed-model swarm beats any homogeneous swarm on **both accuracy (uncorrelated errors) and cost (`50–70%` lower frontier token burn)**:

| Swarm Function | Optimal Model Assignment | Why |
| :--- | :--- | :--- |
| **Parallel Repo/Log Recon Scouts** | `Tier A/B` (`Gemini Flash`, `Flash Lite`, local `Qwen-32B`) | Fast, cheap/free parallel reads; root frontier models keep clean context windows. |
| **Blind Architect Lenses (`Gear 4`)** | Mix `Model Family A` (`Gemini 4 Argon`) + `Model Family B` (`Claude Opus`) + `Model Family C` (`GPT-5.2` / `DeepSeek-R1`) | Maximizes Scott Page's Prediction Diversity ($\text{Ensemble Error} = \bar{E} - \text{Diversity}$). |
| **Competitive Islands (`Pod 2`)** | Distinct model families *plus* distinct architectural priors per island | Eliminates identical patches produced by sampling one model 3 times. |
| **Blind Stage B/C Judges (`Pod 3`)** | Cross-family `Tier S` models (`author_family != judge_family`) | Eliminates 10–25% self-preference bias ("narcissistic LLM judging"). |

---

## 5. The 3-Stage Progressive Judge & Fuzzy Evaluator Engine

### Stage A — Zero-Token Deterministic & Executable-Proxy Sieve (`<5s`)
1. **`clean_patch` Test-Tampering Guard**: Strip candidate modifications to *existing* baseline test files before running the regression suite (new tests run separately; existing assertions cannot be weakened to fake a green build).
2. **Static & Security Gates**: Typecheck/compile, architectural layer check, `.ushignore`/secret diff scan, and AST dangerous-sink scan.
3. **Executable Proxy Runner**: For UI/CLI/perf tasks lacking unit tests, run synthesized deterministic probes (headless CDP console-error/bounding-box checks, CLI exit/latency benchmarks, citation line-bound checkers).
*(Any Stage A failure returns structured stderr directly to the Builder—spending `0` Judge LLM tokens.)*

### Stage B — Single Provenance-Blind Cross-Family Sentinel (`Fast Gate`)
- Receives **only**: verbatim user request, Public + Sealed Acceptance Rubric, `git diff`, and Stage A raw test outputs (`ContextMode::DropAll` — zero Builder `<thinking>` or self-praise).
- Scores each criterion on an **Atomic `BARS` (`0.0 / 0.5 / 1.0`)** scale with mandatory `file:line` evidence + binary Hard Vetoes $h_m \in \{0, 1\}$:
  $$S(c) = \left(\prod_{m} h_m(c)\right) \cdot \sum_{k} w_k b_k(c)$$
- **Early-Stop Fast Pass**: If single-candidate, $\prod h_m = 1$, $S(c) \ge 0.90$, `0` `[P0/P1]` findings, and `uncertainty <= 0.15` $\implies$ **ratify & merge immediately** (~80% of valid tasks stop here!).

### Stage C — Blind Specialist Jury, Pairwise Tournament & Genetic Crossover
Triggered automatically on Stage B ambiguity, multi-island candidate sets (after behavioral deduplication), security-critical paths, or **non-deterministic / fuzzy tasks**:
1. **Parallel Blind Votes (Zero Cross-Talk)**:
   - `J1 (Sandbox Red-Team Prosecutor)`: Live read-only container with `exec_pty`/`grep_files`; attempts actively to break the candidate with adversarial inputs or find broken untouched callers.
   - `J2 (Architecture & Occam Critic)`: Penalizes unnecessary LoC growth, leaky abstractions, and unidiomatic hacks.
   - `J3 (Security & Verbatim User-Intent Auditor)`: Audits trust boundaries + checks every literal clause of the user's raw prompt against host-verified execution provenance.
2. **Split-Only Bounded Rebuttal**: Unanimous rubric rows lock immediately; high-entropy rows run 1 round of evidence-cited cross-examination.
3. **Position-Swapped Pairwise Bradley-Terry Ranking**: Candidate pairs $(A, B)$ and $(B, A)$ are compared in both slot orders to cancel position bias, fitting $\hat{\beta}_i \pm \text{SE}_i$ weighted by Dawid-Skene judge calibration matrices.
4. **4-Way Closed-Loop Routing**:
   - *Clear Winner* $\to$ `3-Tier Merge Reconcile` (`git merge-tree` $\to$ Integrator $\to$ SHA-256 Ledger Ratify).
   - *Complementary Rubric Winners* (Island $\alpha$ wins speed, Island $\beta$ wins safety) $\to$ spawn **1 Round-2 Genetic Crossover Builder** given both diffs + rubric matrix.
   - *All Islands Hit Same Wall* $\to$ escalate `SPEC_DEFECT` to Architect Compiler to amend the flawed contract.
   - *True Value Tie ($\hat{\beta}_1 \approx \hat{\beta}_2$)* $\to$ present 2-option `/grill-me` choice to Human.
