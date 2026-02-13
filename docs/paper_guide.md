# Probabilistic Computers for Neural Quantum States — a nuts-and-bolts reproduction guide

This document is a “from first principles” guide to the paper’s *theory → estimators → algorithms → hardware sampler abstraction* so that you (or a future LLM) can rebuild the full pipeline **in simulation** (including the probabilistic computer) and reproduce the experiments **without** needing the authors’ private code.

It is intentionally detailed. It includes:

* exact mathematical objects the code must represent,
* the estimators used during VMC,
* how FRBM sampling is delegated to a p-bit “probabilistic computer,”
* how DBM training is made tractable via **dual sampling** (with the key bias correction),
* and what a faithful software simulation of their hardware should do.

---

## 0) One-paragraph statement of what the paper actually does

The paper shows that **probabilistic computers** built from networks of stochastic binary units (“p-bits”) can efficiently sample the probability distributions underlying **energy-based neural quantum states** and can therefore accelerate the sampling bottleneck in **variational Monte Carlo (VMC)** for large stoquastic quantum systems. Concretely, they train a **sparse Restricted Boltzmann Machine** (“Further RBM”, FRBM) to approximate the **ground state of the 2D transverse-field Ising model (TFIM)** on lattices up to **80×80** by using FPGA hardware to sample the FRBM, while a CPU performs the optimization. For deeper models (DBMs), where wavefunction ratios become intractable, they introduce a **dual sampling** estimator that computes required single-spin-flip ratios by **conditional sampling of auxiliary layers**, preserving unbiasedness in the infinite-inner-sample limit and adding a **second-order Taylor correction** for finite inner sampling. 

---

## 1) Problem setup: the quantum system and why the wavefunction can be taken nonnegative

### 1.1 Hamiltonian (2D TFIM)

They study the 2D TFIM on an (L\times L) square lattice with periodic boundaries, in the (\sigma^z) computational basis:
[
H = -J \sum_{\langle i,j\rangle} \sigma_i^z \sigma_j^z ;-; \Gamma_x \sum_i \sigma_i^x.
]
This appears as Eq. (4) (paper’s numbering). 

**Basis and configurations.** A basis configuration (v) (they also use (S)) is a vector of (N=L^2) spins (v_i\in{-1,+1}).

### 1.2 Stoquasticity and Perron–Frobenius ⇒ nonnegative ground-state amplitudes

For stoquastic Hamiltonians like TFIM in the (\sigma^z) basis (off-diagonal elements (\le 0)), the ground state can be chosen with **nonnegative real amplitudes**. The paper uses this to model the wavefunction as:
[
\Psi_\theta(v) = \sqrt{p_\theta(v)}\quad\text{with }p_\theta(v)\ge 0,
]
so the NQS becomes an explicit probability model. The dual-sampling derivation in the supplement relies on exactly this construction. 

---

## 2) VMC fundamentals: what you must compute every iteration

### 2.1 Variational energy as an expectation over (|\Psi|^2)

For any parameterized state (\Psi_\theta), the variational energy is:
[
E(\theta) = \frac{\langle \Psi_\theta|H|\Psi_\theta\rangle}{\langle\Psi_\theta|\Psi_\theta\rangle}.
]
VMC rewrites it as an expectation over samples (v\sim|\Psi_\theta(v)|^2):
[
E(\theta) = \mathbb{E}*{v\sim|\Psi*\theta|^2}\left[ E_{\text{loc}}(v)\right],
]
where (E_{\text{loc}}(v)=\frac{(H\Psi_\theta)(v)}{\Psi_\theta(v)}). 

### 2.2 Local energy for TFIM: diagonal + off-diagonal ratios

In the (\sigma^z) basis, TFIM local energy becomes:
[
E_{\text{loc}}(v)= -J\sum_{\langle i,j\rangle} v_i v_j ;-; \Gamma_x \sum_{i=1}^{N}\frac{\Psi_\theta(v^{(i)})}{\Psi_\theta(v)}.
]
Here (v^{(i)}) is configuration (v) with spin (i) flipped. 

**This is the computational heart of everything.** Every training iteration needs:

1. samples (v^{(s)}\sim |\Psi_\theta|^2),
2. for each sample, the full set of (N) ratios (\Psi(v^{(i)})/\Psi(v)),
3. gradients / natural gradients of the energy wrt (\theta).

---

## 3) The neural quantum states used here: FRBM (sparse RBM) and DBM

### 3.1 FRBM / sparse RBM definition (the hardware-friendly one)

They use a **Further Restricted Boltzmann Machine (FRBM)** with **strictly local connectivity** to make hardware routing feasible. The RBM “energy” (i.e., negative log unnormalized probability) is:
[
E_\theta(S,h) = -\sum_i a_i S_i ;-;\sum_j b_j h_j ;-;\sum_{\langle i,j\rangle_k} W_{ij} S_i h_j,
]
where (S\in{-1,+1}^N) is the visible layer, (h\in{-1,+1}^M) the hidden layer, and (\langle i,j\rangle_k) restricts connections to nodes within Euclidean distance (k). This is Eq. (5). 

Key hardware parameter:

* They set **(k=2)**, which corresponds to **13 neighbors per spin**. 

The associated probability model is:
[
p_\theta(S)=\frac{1}{Z_\theta}\sum_h e^{-E_\theta(S,h)}.
]
The wavefunction is then (\Psi_\theta(S)=\sqrt{p_\theta(S)}). (This is the same construction used explicitly in the supplement for DBM.) 

### 3.2 DBM (deep Boltzmann machine): why it breaks the easy parts

A DBM adds another auxiliary layer (d), producing a 3-layer undirected model with couplings (v!-!h) and (h!-!d). The wavefunction remains (\Psi=\sqrt{p(v)}), but now:

* (p(v)) involves an intractable sum over ((h,d)),
* and the ratio (p(v^{(i)})/p(v)) is **not** analytically reducible the way RBMs are.

That “ratio intractability” is what their **dual sampling** fixes. 

---

## 4) The probabilistic computer abstraction: p-bits as a hardware sampler for Boltzmann models

### 4.1 What a p-bit is, mathematically

A p-bit is a binary stochastic unit (\sigma_i\in{-1,+1}) updated according to a local field (I_i). They define the update as:

**Local field**
[
I_i = \sum_j W_{ij}\sigma_j + b_i.
]
**Stochastic update rule**
[
\sigma_i = \mathrm{sgn}\big(\tanh(\beta I_i)-r_{[-1,1]}\big),
]
where (r_{[-1,1]}) is uniform on ([-1,1]). (Eqs. (8)–(9).) 

Equivalently, this is a Bernoulli draw:
[
\mathbb{P}(\sigma_i=+1)=\frac{1}{2}\big(1+\tanh(\beta I_i)\big),\quad
\mathbb{P}(\sigma_i=-1)=\frac{1}{2}\big(1-\tanh(\beta I_i)\big),
]
which are Eqs. (10)–(11). 

### 4.2 Why this samples the intended distribution

This is the standard **Glauber / heat-bath** update for an Ising-type energy:
[
E(\sigma) = -\frac{1}{2}\sum_{i\neq j}W_{ij}\sigma_i\sigma_j - \sum_i b_i\sigma_i,
]
when (W) is symmetric. For RBM/DBM graphs, couplings are bipartite (no within-layer edges), so you can also write (E(\sigma)=-\sum_{(i,j)\in\text{edges}} W_{ij}\sigma_i\sigma_j-\sum_i b_i\sigma_i) without the (\tfrac12) ambiguity.

Under sequential (or properly colored) updates, the stationary distribution is the Boltzmann distribution (p(\sigma)\propto e^{\beta \sum_{ij} W_{ij}\sigma_i\sigma_j + \beta\sum_i b_i\sigma_i}). **In software reproduction, you should treat the p-computer as an MCMC engine implementing this update kernel.**

### 4.3 Hardware-specific implementation details you must mimic in simulation (if you want fidelity)

They describe a hardware implementation with:

* **fixed-point** weights/biases in format (s{6}{3}) (1 sign bit, 6 integer bits, 3 fractional bits; “10-bit precision”), and
* a **xoshiro** PRNG for the random bits. 

They also use **graph coloring** to update many p-bits in parallel on sparse graphs (so that within a color class there are no edges and updates are conditionally independent given the rest). 

---

## 5) Scaling the sampler beyond one FPGA: multi-FPGA cluster and asynchronous boundary exchange

To reach (L=50) and (L=80), they partition the FRBM graph across 6 FPGAs using **METIS min-cut**. Cross-partition edges become “shadow weights,” and only boundary p-bit **states** are communicated. 

Key engineering choices:

* **Cut fractions** are reported as ~8.6% (for (L=50)) and ~5.6% (for (L=80)). 
* Boundary p-bits are exchanged **asynchronously**, and the receiving FPGA holds boundary states constant between communication events. 
* This allows local p-bit update clocks to be overclocked to **15 MHz**, whereas strict global synchronization would reduce the local clock dramatically (they cite 2.4 MHz / 1.2 MHz bounds for (L=50/80)). 

### What this means for software simulation fidelity

If you want to simulate their *multi-FPGA effect* (not just “some MCMC”), you should simulate:

1. A graph partition into subgraphs.
2. Each subgraph runs local p-bit updates with *stale* boundary values.
3. At some communication interval, boundary states are exchanged (synchronously in simulation, asynchronously in concept).

This creates a controlled approximation to global Gibbs sampling; empirically they show it works well enough for the VMC outer loop.

---

## 6) FRBM-specific “analytic tractability”: what you should exploit in code

The paper states that with the FRBM sparsity they keep “analytical tractability for wavefunction amplitudes, local energies, and gradients.” 

Here is the explicit math you should implement (even if the paper doesn’t spell every algebra step).

### 6.1 RBM marginal over hidden spins

Given energy
[
E(S,h)= -a^\top S - b^\top h - S^\top W h
]
(with (S_i,h_j\in{\pm1})), the unnormalized visible probability is:
[
\tilde p(S)=\sum_h e^{-E(S,h)}
= e^{a^\top S}\prod_{j=1}^M \sum_{h_j=\pm1}\exp\left(h_j\left(b_j + (W^\top S)*j\right)\right)
= e^{a^\top S}\prod*{j=1}^M 2\cosh!\left(b_j + (W^\top S)_j\right).
]

Therefore the wavefunction (unnormalized) can be taken as:
[
\Psi(S)\propto \sqrt{\tilde p(S)}
= \exp!\left(\tfrac12 a^\top S\right)\prod_{j=1}^M \left[2\cosh!\left(b_j + (W^\top S)_j\right)\right]^{1/2}.
]

### 6.2 Log-derivatives needed for SR / natural gradient

Define the “log-derivative operators”
[
O_k(S)=\frac{\partial}{\partial \theta_k}\log \Psi_\theta(S).
]

From the log above:

* For visible bias (a_i): (O_{a_i}(S)=\tfrac12 S_i.)
* For hidden bias (b_j): (O_{b_j}(S)=\tfrac12\tanh!\left(b_j + (W^\top S)_j\right).)
* For weight (W_{ij}) (when present): (O_{W_{ij}}(S)=\tfrac12 S_i \tanh!\left(b_j + (W^\top S)_j\right).)

Because FRBM is sparse, computing all these scales like (O(#\text{edges})), not (O(NM)).

### 6.3 Single-spin-flip ratio for TFIM local energy

Let (S^{(i)}) be (S) with (S_i\to -S_i). Then:
[
\frac{\Psi(S^{(i)})}{\Psi(S)}
= \exp(-a_i S_i)\prod_{j\in \mathcal N(i)} \left[\frac{\cosh!\left(\theta_j(S) - 2W_{ij}S_i\right)}{\cosh!\left(\theta_j(S)\right)}\right]^{1/2}
]
where (\theta_j(S)=b_j + \sum_{i'} W_{i'j} S_{i'}) and (\mathcal N(i)) are hidden units connected to visible (i). Sparsity makes (|\mathcal N(i)|) small and local.

**This is the fast path** for FRBM: local energy ratios are analytic and cheap.

---

## 7) Optimization: Stochastic Reconfiguration (SR) with conjugate gradient

The paper uses SR and explicitly mentions a **conjugate gradient** (CG) implementation “to avoid direct inversion.” 
The supplement provides concrete CG settings: relative tolerance (10^{-4}), max 500 iterations, and matrix-free evaluation of (S\cdot v). 

### 7.1 SR equations (what you implement)

Given samples (S^{(s)}\sim |\Psi|^2), define:

* Energy mean: (\bar E = \langle E_{\text{loc}}\rangle.)
* Log-derivatives: (O_k^{(s)} = O_k(S^{(s)}).)

Then define:

* “Force” / gradient-like vector:
  [
  f_k = \left\langle O_k E_{\text{loc}}\right\rangle - \left\langle O_k\right\rangle\left\langle E_{\text{loc}}\right\rangle.
  ]
* SR metric (covariance / Fisher):
  [
  S_{kl} = \left\langle O_k O_l\right\rangle - \left\langle O_k\right\rangle\left\langle O_l\right\rangle.
  ]

Regularize with diagonal shift (\lambda):
[
(S+\lambda I),\delta\theta = f.
]

Update parameters (sign convention varies; a common one):
[
\theta \leftarrow \theta - \eta,\delta\theta.
]

### 7.2 Matrix-free CG for SR

You **never** form (S) explicitly for large parameter counts. You implement a function that computes ((S+\lambda I)v) for any vector (v) using sample averages:

1. Compute (u^{(s)} = \sum_l O_l^{(s)} v_l).
2. Then (((Sv)_k) = \langle O_k u\rangle - \langle O_k\rangle\langle u\rangle).
3. Add (\lambda v_k).

This is exactly what “matrix-free implicit products” means. 

### 7.3 Learning rate / diagonal shift schedules (what the paper actually sets)

The supplement gives a training hyperparameter summary (Table S1), including:

* (N_\text{iter}=1000),
* (N_s=10{,}000) outer samples,
* (N_c=1{,}000) clamped (inner) samples for dual sampling,
* cosine-decayed learning rate from (\eta_{\max}=0.1) to (\eta_{\min}=10^{-5}),
* SR diagonal shift (\lambda_0=0.1) with geometric decay factor (b_0=0.9),
* CG tolerance (10^{-4}), max CG iters 500,
* final evaluation with (N_\text{eval}=10^6) samples. 

> Practical note: Table S1 is written in the DBM/dual-sampling context; FRBM experiments also report specific iteration counts and sample sizes in the main figure caption (see below).

---

## 8) The DBM bottleneck and the “dual sampling” solution

### 8.1 Where DBM becomes expensive: the ratio in the TFIM local energy

TFIM local energy needs:
[
\frac{\Psi(v^{(i)})}{\Psi(v)} = \sqrt{\frac{p(v^{(i)})}{p(v)}}.
]
But for DBM:
[
p(v)=\sum_{h,d} p(v,h,d)\propto \sum_{h,d}e^{-E(v,h,d)},
]
and that sum is not closed-form.

### 8.2 The key identity: ratio as a conditional expectation (this is the paper’s core math trick)

Define the *joint* model weight:
[
p_\theta(v,h,d)=\frac{1}{Z_\theta}e^{-E_\theta(v,h,d)}.
]
For a flip at (i), define the energy difference:
[
\Delta E_i(v,h,d) = E_\theta(v^{(i)},h,d)-E_\theta(v,h,d) = 2 I_i(v,h,d), v_i.
]
This is Eq. (S.9). 

Then:
[
\frac{p(v^{(i)})}{p(v)}
= \mathbb{E}_{(h,d)\sim p(h,d|v)}\left[e^{-\Delta E_i(v,h,d)}\right]
\equiv r_i(v),
]
which is Eq. (S.13) establishing Eq. (S.7). 

Finally, because (\Psi=\sqrt{p}),
[
\frac{\Psi(v^{(i)})}{\Psi(v)} = \sqrt{r_i(v)}.
]
That’s Eq. (S.5). 

### 8.3 Why it’s called “dual sampling”

There are two nested Monte Carlo loops:

* **Outer loop**: sample visible configurations (v\sim|\Psi(v)|^2\propto p(v)).
* **Inner loop**: for each fixed (v), sample ((h,d)\sim p(h,d|v)) and estimate (r_i(v)) for all (i).

Because the inner samples are conditional on the same (v), you can reuse them to estimate all (N) ratios (r_i(v)) with only (N_c) conditional samples, rather than doing something like (N\times N_c) independent inner runs.

---

## 9) Unbiasedness and the finite-(N_c) bias: what must be corrected

### 9.1 Unbiasedness in the infinite-inner-sample limit

Let (\hat r_i(v)) be the sample average estimator of (r_i(v)) using (N_c) clamped samples. Under (N_c\to\infty):
[
\hat r_i(v)\to \frac{p(v^{(i)})}{p(v)} \quad\Rightarrow\quad \sqrt{\hat r_i(v)}\to \frac{\Psi(v^{(i)})}{\Psi(v)}.
]
This is Eq. (S.6). 

Plugging into the local energy expression yields the correct (E_{\text{loc}}(v)) in the limit (Eq. (S.7)). 

### 9.2 The finite-(N_c) issue: square root is nonlinear ⇒ residual bias

Even if (\hat r) is unbiased for (r), (\sqrt{\hat r}) is not unbiased for (\sqrt{r}) at finite (N_c). The paper addresses this with a **second-order Taylor correction** in the estimator used inside the TFIM local energy sum (and they codify it directly in Algorithm S1). 

---

## 10) Algorithm S1 (DBM training) — interpret it as executable pseudocode

Algorithm S1 is the paper’s full “machine-learning quantum Hamiltonians using deep NQS” procedure, written in hardware-friendly terms.

### 10.1 What Algorithm S1 is actually doing at each level

* It maintains p-bit states for all layers ({v,h,d}).
* For each iteration (t):

  1. generate (N_s) visible samples,
  2. for each visible sample, clamp it and run (N_c) conditional updates on hidden+deep,
  3. accumulate:

     * diagonal energy,
     * off-diagonal terms via (p_\text{flip}) estimates,
     * log-derivatives (O) for SR,
  4. solve SR update and apply (\eta(t)).

The algorithm explicitly uses the p-bit update rule repeatedly in parallel loops. 

### 10.2 The crucial lines: how they compute the ratio estimator and Taylor correction

Inside Algorithm S1, after accumulating:

* (pflip_i \leftarrow \frac{1}{N_c}\sum_{k=1}^{N_c} e^{-2 I_i v_i}),
* (pflip_sq_i \leftarrow \frac{1}{N_c}\sum_{k=1}^{N_c} e^{-4 I_i v_i}),

they compute a population variance proxy:
[
\mathrm{Varpop}_i = \frac{pflip_sq_i - (pflip_i)^2}{N_c},
]
and then:
[
\Delta_i = \frac{\mathrm{Varpop}_i}{8,pflip_i \sqrt{pflip_i}}.
]
These are explicit in the algorithm. 

Then the local energy off-diagonal update is (algorithm line 37):
[
E_{\text{loc}}(v)\leftarrow E_{\text{loc}}(v) + H_{v,v^{(i)}}\big(\sqrt{pflip_i}+\Delta_i\big).
]


For TFIM, (H_{v,v^{(i)}}=-\Gamma_x) (spin-flip matrix element), so this implements the corrected (-\Gamma_x \sum_i (\sqrt{pflip_i}+\Delta_i)) term.

> Interpretation tip: a second-order Taylor approximation of (\mathbb{E}[\sqrt{\hat r}]) produces a correction proportional to (\mathrm{Var}(\hat r)). The algorithm “bakes” that correction into the energy estimator rather than trying to debias the ratio itself.

---

## 11) Parameter counting and sparse connectivity (needed to reproduce Figure 4-type sweeps)

The supplement defines a geometric distance on a periodic lattice by assigning each layer a 2D coordinate system isomorphic to the visible lattice. The distance between neuron (i) at (r_i) and neuron (j) at (r_j) is:
[
d(i,j) = \min_{\delta\in\mathbb{Z}^2}|r_i-r_j + L\cdot \delta|_2,
]
and a connection exists iff (d(i,j)\le k). (Eq. (S.15).) 

They then enumerate parameter counts for sparse RBM vs sparse DBM under this mask (Tables S2 and S3 are referenced right after). 

**What you should implement:** a deterministic connectivity constructor:

* assign each layer the same (L\times L) coordinate grid,
* for every node in layer (L), connect it to nodes in layer (L+1) within radius (k) (wrapping periodically),
* store adjacency in CSR/COO form (you’ll need it for fast local-field computation).

---

## 12) What experiments you must reproduce (and the settings the paper actually reports)

### 12.1 Single-FPGA FRBM experiment (Figure 2 caption gives the essential protocol)

They train on (35\times 35) at the TFIM critical field (\Gamma_c/J=3.044), and report that energy per spin reaches “chemical accuracy” (relative error (\le 1.6\times 10^{-3})) within ~100 iterations. 

They validate energy vs field against CT-PIMC and compute final points as an average over (10^6) samples, with standard error via blocking (50 bins). 

### 12.2 Multi-FPGA FRBM scaling (what to reproduce, qualitatively and architecturally)

Reproduce:

* same FRBM definition (k=2),
* same sampler kernel,
* partitioned graph with asynchronous boundary exchange,
* show you can train stably at larger (L) and that the sampler remains efficient.

The key architectural description is in Fig. S2. 

### 12.3 DBM dual sampling experiments

To reproduce dual sampling:

* implement Algorithm S1 exactly as written (outer (N_s), inner (N_c), compute (pflip_i, pflip_sq_i, \Delta_i)),
* implement SR with CG tolerance (10^{-4}) and max 500,
* use Table S1 schedule for (\eta(t)) and (\lambda(t)). 

---

## 13) How to simulate the probabilistic computer in software (faithful reproduction recipe)

This section is the “if you were to write the code” blueprint, but **not code**.

### 13.1 Data model you must represent

You need a graph (G=(V,E)) of p-bits where:

* nodes (V) are all spins across layers (FRBM: (V=v\cup h); DBM: (V=v\cup h\cup d)),
* edges (E) correspond to nonzero couplings (W_{ij}),
* biases (b_i) are per node.

Store:

* `bias[i]` (fixed-point if you want fidelity),
* adjacency lists with `(j, W_ij)` for each node i,
* current state `sigma[i] ∈ {−1,+1}`.

### 13.2 One p-bit update (the core kernel)

For a chosen node (i):

1. compute (I_i=\sum_{j\in \mathcal N(i)} W_{ij}\sigma_j + b_i),
2. draw uniform (r\in[-1,1]),
3. set (\sigma_i = \mathrm{sgn}(\tanh(\beta I_i)-r)).

That is exactly Algorithm S1’s update lines and the Methods equations. 

### 13.3 A “sweep” / update schedule: match hardware parallelism with graph coloring

Because they use graph coloring to parallelize sparse updates, implement:

* Precompute a coloring of the p-bit graph (or of a conflict graph) so that within each color class, nodes have no edges.
* For each sweep:

  * iterate colors (c=1..C),
  * update all nodes in color (c) (in any order; can be parallel in simulation).

This matches the paper’s hardware approach (“graph coloring to maximize parallelism”) for sparse instances. 

### 13.4 Sampling (v\sim p(v)) for VMC outer loop

For FRBM:

* run the sampler on the full joint network ((v,h)),
* after burn-in, periodically record the visible layer states (v),
* those recorded (v) are samples from the marginal (p(v)) (up to MCMC mixing).

For DBM dual sampling:

* you still need outer samples (v). The algorithm indicates visible-layer updates happen before clamping (lines 7–10), meaning you can run visible updates as part of a full network evolution and then “freeze” the visible configuration for inner sampling. 

### 13.5 Inner loop: conditional sampling (p(h,d|v)) by clamping

For each recorded visible configuration (v):

* set the visible p-bits to those fixed values (“clamp or fix”),
* run (N_c) conditional sampling steps on the auxiliary layers only (hidden then deep, as the algorithm loops). 

During these inner samples, compute:

* (I_i(v,h,d)) for each visible unit (i) (yes, visible units are clamped, but their local field still depends on current auxiliary spins),
* accumulate (e^{-2 I_i v_i}) and (e^{-4 I_i v_i}) to form (pflip_i) and (pflip_sq_i). 

### 13.6 Multi-FPGA behavior (optional fidelity layer)

To mimic multi-FPGA sampling, implement:

* partition nodes into subgraphs,
* maintain boundary states for cross-partition neighbors,
* run local sweeps for some number of steps with boundary held fixed,
* then “communicate” boundary bits (swap latest boundary states),
* repeat.

This matches the described asynchronous boundary exchange / stale boundaries. 

---

## 14) Reproduction blueprint: end-to-end pipeline (FRBM and DBM)

Below is *pseudocode-level* detail (not an implementation), designed to be directly translatable to code.

### 14.1 FRBM VMC training loop

1. **Initialize FRBM parameters** (a,b,W) (small random values; paper uses Gaussian initializations in Algorithm S1 for DBM; FRBM likely similar).

2. For each iteration (t):

   * Sample (N_s) visible configs (v^{(s)}\sim p_\theta(v)) using the p-bit sampler on ((v,h)).
   * For each sample (v^{(s)}):

     * Compute diagonal term (-J\sum_{\langle i,j\rangle} v_i v_j).
     * Compute all ratios (\Psi(v^{(i)})/\Psi(v)) using the **analytic FRBM ratio** (Section 6.3).
     * Form (E_{\text{loc}}(v^{(s)})).
     * Compute log-derivatives (O_k(v^{(s)})).
   * Compute sample estimates of (f) and (S).
   * Solve ((S+\lambda I)\delta\theta=f) with CG.
   * Update (\theta\leftarrow \theta - \eta(t)\delta\theta).

3. After training, freeze (\theta) and compute energy with (N_\text{eval}=10^6) samples and blocking error bars (50 bins). 

### 14.2 DBM dual-sampling VMC training loop (Algorithm S1 faithful)

Use Algorithm S1 as the blueprint:

* outer samples (N_s),
* inner clamped samples (N_c),
* compute (pflip_i), (\Delta_i),
* compute (E_{\text{loc}}(v)) including corrected off-diagonal term,
* compute SR gradients and update parameters. 

Use Table S1 training settings unless you have experiment-specific overrides:

* (N_s=10{,}000), (N_c=1{,}000), (N_\text{iter}=1000), (\eta:0.1\to10^{-5}), (\lambda_0=0.1), (b_0=0.9), CG tol (10^{-4}), CG max 500, final (10^6) eval samples. 

---

## 15) “Gotchas” and validation tests (high value for a clean reimplementation)

### 15.1 Sign / basis conventions

* Spins are (\pm 1).
* TFIM off-diagonal matrix elements in (\sigma^z) basis are (-\Gamma_x) for single-spin flips.
* Make sure your (v^{(i)}) is exactly one bit flipped.

### 15.2 Connectivity construction must be periodic and Euclidean

Use the periodic Euclidean distance definition in Eq. (S.15) for sparsity masks. 

### 15.3 Debug the sampler independently

Before coupling to VMC:

* On a tiny RBM/DBM, compare marginal statistics against brute force enumeration.
* Verify detailed balance numerically (transition probabilities).

### 15.4 Verify the dual sampling identity numerically

On a tiny DBM:

* brute force compute (p(v^{(i)})/p(v)),
* compare to the inner-loop conditional expectation estimator (r_i(v)). 

### 15.5 Check the Taylor correction effect

On a fixed (v):

* run repeated inner loops with the same (N_c),
* estimate bias of (\sqrt{\hat r}) and see whether (\Delta_i) reduces it in the energy estimate. 

---

## 16) What you need to reproduce Figure 2 numerically (minimum viable checklist)

From the Figure 2 caption you must match:

* FRBM with **k=2 (13 neighbors)**, 
* (35\times 35) at (\Gamma_c/J=3.044), 
* convergence to **chemical accuracy** threshold (|\Delta E/E_\text{ref}|\le 1.6\times 10^{-3}) in ~100 iterations, 
* final energy points from **(10^6)** samples, blocking error bars (50 bins), 
* comparison to CT-PIMC (not required for “rebuild pipeline,” but needed for exact plot match). 

---

## 17) Minimal module breakdown for a clean reimplementation (guide-level, not code)

If you were building this cleanly, you’d likely create these modules:

1. `lattice.py`

   * build periodic (L\times L) index map
   * build nearest-neighbor bonds (\langle i,j\rangle)
   * build periodic Euclidean distance function (d(i,j))

2. `connectivity.py`

   * build sparse masks for FRBM (visible-hidden radius (k))
   * build sparse masks for DBM (radii (k_1,k_2)) using Eq. (S.15) 

3. `pbit_sampler.py`

   * fixed-point option (to match (s{6}{3})) 
   * xoshiro RNG option 
   * graph coloring scheduler 
   * optional partitioned “multi-FPGA” asynchronous boundary mode 
   * update kernel (Eqs. 8–11) 

4. `frbm_wavefunction.py`

   * compute (\log\Psi)
   * compute ratios (\Psi(v^{(i)})/\Psi(v))
   * compute log-derivatives (O_k)

5. `dbm_dual_sampling.py`

   * implement Algorithm S1 inner/outer loops 
   * compute (pflip, pflip_sq, \Delta), and corrected off-diagonal energy contribution 

6. `vmc.py`

   * local energy TFIM formula 
   * sample averaging, blocking (50 bins) 

7. `sr_solver.py`

   * matrix-free (S\cdot v) 
   * CG settings from supplement 
   * schedules (\eta(t), \lambda(t)) from Table S1 

---

## 18) Where to be “extra careful” if you want to match the *hardware* experiment, not just the physics

If you want your software simulation to match *their* sampler behavior closely, these are the fidelity knobs:

1. **Update schedule**: color-parallel vs fully sequential. (They explicitly use graph coloring.) 
2. **Numeric precision**: emulate (s{6}{3}) fixed-point and tanh LUT quantization (they use fixed-point and FPGA hardware). 
3. **PRNG**: xoshiro bitstream. 
4. **Multi-FPGA**: stale boundary updates & communication interval. 
5. **Sampling thinning / burn-in**: tune to reproduce effective sample size implied by their iteration budgets and final blocking statistics.

---

## 19) Closing: what “godly understanding” means operationally for rebuilding it

If you can do all of the following, you have the same functional understanding as the authors’ implementation:

* Derive TFIM local energy in (\sigma^z) basis and explain why it reduces to diagonal + flip ratios. 
* Derive RBM marginalization and implement fast analytic ratios and log-derivatives (Section 6).
* Explain p-bit dynamics as heat-bath updates and map FRBM/DBM parameters onto the p-bit network fields and couplings. 
* Re-derive the dual sampling identity (p(v^{(i)})/p(v)=\mathbb{E}_{p(h,d|v)}[e^{-\Delta E_i}]) and show why it yields unbiased local energies in the (N_c\to\infty) limit.  
* Implement Algorithm S1 logic exactly (outer/inner loops, (pflip), (\Delta), SR update).  
* Implement SR with matrix-free CG and match CG tolerances and schedules from Table S1. 
* Reproduce the Figure 2 protocol numerically (k=2, (35\times35), (\Gamma_c/J=3.044), chemical accuracy threshold, (10^6) eval samples, blocking). 