# Methodology

This document is the math walkthrough behind the implementation. It is intended to be read alongside the source — every formula here corresponds to a function in `src/`. Notation follows Glasserman (2003) and Bouzoubaa & Osseiran (2010) where convenient.

> Note: GitHub renders LaTeX inside `$...$` and `$$...$$` via KaTeX. The formulas below have been written for that renderer.

## 1. Risk-neutral dynamics

Each underlying $S_i$, $i=1,\dots,d$, follows correlated geometric Brownian motion under the pricing measure $\mathbb{Q}$:

$$\frac{dS_i(t)}{S_i(t)} = (r - q_i)\, dt + \sigma_i\, dW_i(t), \qquad d\langle W_i, W_j\rangle_t = \rho_{ij}\, dt.$$

Constants:
- $r$: risk-free rate (continuous compounding).
- $q_i$: continuous dividend yield of name $i$.
- $\sigma_i$: implied (or hardcoded textbook) volatility of name $i$.
- $\rho_{ij}$: pairwise correlation; the matrix $\Sigma = (\rho_{ij})$ is positive semi-definite.

Closed-form solution along a grid $0 = t_0 < t_1 < \dots < t_N = T$:

$$S_i(t_{k+1}) = S_i(t_k) \exp\!\left[\left(r - q_i - \tfrac{1}{2}\sigma_i^2\right)\Delta t + \sigma_i\sqrt{\Delta t}\, Z^{(i)}_k\right],$$

where $(Z^{(1)}_k,\dots,Z^{(d)}_k) \sim \mathcal{N}(\mathbf{0}, \Sigma)$ i.i.d. across $k$.

### Cholesky for correlated normals

If $\Sigma = L L^\top$ (Cholesky) and $\eta \sim \mathcal{N}(\mathbf{0}, I_d)$, then $L\eta \sim \mathcal{N}(\mathbf{0}, \Sigma)$. We sample $\eta$ as `numpy.random.standard_normal((n_paths, n_steps, d))` and contract with $L$ along the last axis — fully vectorised, no Python loops.

## 2. Variance reduction

### 2.1 Antithetic variates

For each draw $\eta$ we also use $-\eta$ and average the two payoffs. This kills the linear-in-noise component of the payoff variance and is essentially free.

### 2.2 Control variate: worst-of European put

Let $X$ be the FCN discounted payoff and $Y$ be the discounted payoff of a worst-of European put on the same basket. The control-variate estimator is

$$\widehat{X}_{cv} = \overline{X} - \beta\,(\overline{Y} - \mathbb{E}[Y]),$$

with $\beta^\star = \mathrm{Cov}(X,Y)/\mathrm{Var}(Y)$, estimated from the simulated sample. $\mathbb{E}[Y]$ is computed analytically when possible, or with a much larger MC run otherwise. The worst-of put is the right control because the FCN's downside risk is dominated by the worst-performer breaching the knock-in.

## 3. FCN payoff

Notation:
- Notional $N = 100$.
- Initial fixings $S_i(0)$.
- Worst performer at time $t$: $W(t) = \min_i S_i(t)/S_i(0)$.
- Observation dates $\mathcal{T}_{obs} = \{t_1, \dots, t_M\}$ with $t_M = T$.
- Coupon barrier $B_c$ (default 0.70), autocall barrier $B_{ac}$ (1.00), knock-in barrier $B_{ki}$ (0.65).
- Per-period coupon $c$ (e.g. 8% p.a. quarterly $\Rightarrow c = 0.02 \cdot N$).

### Cashflows along a path

Walk forward through observation dates. Let $j^\star$ be the first $j \ge 2$ (autocall typically excludes the first observation; configurable) such that $W(t_j) \ge B_{ac}$. Then:

- For $j = 1, \dots, j^\star$, pay $c \cdot \mathbb{1}\{W(t_j) \ge B_c\}$ at $t_j$.
- At $t_{j^\star}$, additionally redeem at $N$ and the note terminates.

If no such $j^\star$ exists, walk through all $t_j$ paying conditional coupons, and at $t_M = T$ apply maturity logic:

$$\text{Final redemption} = \begin{cases} N, & W(T) \ge B_{ac} \\ N, & W(T) \ge B_{ki}\ \text{(no knock-in)} \\ N \cdot W(T), & \text{knock-in breached and } W(T) < B_{ac} \end{cases}$$

The "knock-in breached" check is the European variant by default: $W(T) < B_{ki}$. Continuous monitoring is also implemented (knocked in if $\min_{t \in [0,T]} W(t) < B_{ki}$); see `fcn_payoff.py`.

### Discounted price

$$V_0 = \mathbb{E}^{\mathbb{Q}}\!\left[\sum_{j=1}^{j^\star} e^{-r t_j}\, CF_j\right],$$

estimated as the sample mean across MC paths.

## 4. Crank–Nicolson PDE (1D reduced case)

For the single-asset case the value $V(S, t)$ satisfies

$$\frac{\partial V}{\partial t} + \tfrac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - q) S \frac{\partial V}{\partial S} - r V = 0,$$

solved backward from $T$ to $0$.

### Discretisation

- Non-uniform space grid concentrated near the barriers $\{B_{ki}\,S_0,\, B_c\,S_0,\, B_{ac}\,S_0\}$ via a sinh-stretched mapping.
- Crank–Nicolson in time with $\theta = 1/2$, giving $O(\Delta t^2 + \Delta S^2)$ accuracy and unconditional A-stability.
- Tridiagonal system solved via Thomas algorithm each step.

### Boundary conditions

- $S \to 0$: $V \to 0$ (no recovery on zero spot, with knock-in already breached).
- $S \to S_{\max}$: linear extrapolation $V_{S\to S_{\max}} = N$ (autocall absorbs).

### Discrete observation dates

Between observations, evolve the PDE freely. At each observation date $t_j$ (walking backward we hit them after they have been "applied" forward in real time), we modify the in-grid values:

- If $S \ge B_{ac} S_0$: replace $V(S, t_j^-)$ with $N + c$ (autocall, terminate locally).
- Otherwise add the conditional coupon: $V(S, t_j^-) \mathrel{+}= c \cdot \mathbb{1}\{S \ge B_c S_0\}$.

For the European knock-in variant we apply the maturity payoff at $t_M$ and then evolve back.

## 5. Greeks

### 5.1 Bump-and-revalue with common random numbers (CRN)

For an underlying $i$ and a small bump $\epsilon$:

$$\Delta_i \approx \frac{V(S_i + \epsilon) - V(S_i - \epsilon)}{2\epsilon},$$

where both prices are computed using the *same* sequence of random draws (CRN). This reuses $\eta$ across bumps and is essential for variance reduction in MC Greeks.

Gamma is computed analogously with three points:

$$\Gamma_i \approx \frac{V(S_i + \epsilon) - 2 V(S_i) + V(S_i - \epsilon)}{\epsilon^2}.$$

### 5.2 Pathwise method (smoother MC delta)

For the smooth pieces of the payoff (the down-side participation region $V \cdot W(T)$), the pathwise estimator gives

$$\Delta_i^{\text{pw}} = \mathbb{E}\!\left[ e^{-rT}\,\frac{\partial \Phi}{\partial S_i(T)} \cdot \frac{S_i(T)}{S_i(0)} \right],$$

with $\Phi$ the maturity payoff. The pathwise estimator does not work at the discontinuities (the autocall and knock-in events), where we fall back to bump-and-revalue or smoothed-payoff techniques.

### 5.3 Vega and correlation

Vega and pairwise correlation sensitivities are computed by re-Cholesky-ing $\Sigma$ on each bump and re-driving the simulation with the same underlying $\eta$. This isolates the parameter perturbation cleanly.

## 6. References

- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*, Ch. 2 (variance reduction), Ch. 7 (sensitivity estimation).
- Bouzoubaa, M. & Osseiran, A. (2010). *Exotic Options and Hybrids*, Ch. 12 (FCNs and worst-of structures).
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*, Ch. on finite-difference methods for path-dependent options.
