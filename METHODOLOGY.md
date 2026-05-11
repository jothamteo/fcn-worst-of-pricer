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

solved backward from the final valuation date $t_M$ to the issue date.

### Log-spot transformation

We solve in $x = \log(S/S_0)$, which removes the $S$-dependence from the
diffusion and convection coefficients:

$$\frac{\partial V}{\partial t} + \tfrac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial x^2} + (r - q - \tfrac{1}{2}\sigma^2)\frac{\partial V}{\partial x} - r V = 0.$$

A uniform $x$-grid then translates into exponentially-stretched $S$
spacing — denser in the low-$S$ region where the strike and knock-in sit.
That's the cheap, clean alternative to a sinh-stretched grid in $S$.

### Discretisation

- Uniform $x$-grid on $[-x_{\max}, +x_{\max}]$ with $x_{\max} = k\,\sigma\sqrt{T}$, typically $k = 6$.
- Central differences in $x$ for both $\partial_x$ and $\partial_{xx}$.
- Crank–Nicolson in time with $\theta = 1/2$, giving $O(\Delta t^2 + \Delta x^2)$ accuracy and A-stability.
- Tridiagonal system per step, solved via scipy's banded LAPACK driver.

### Boundary conditions

Linear extrapolation at both ends, i.e. $V_{xx} = 0$ at $x = \pm x_{\max}$.
With $x_{\max} = 6\sigma\sqrt{T}$ both boundaries are well outside the
support of the payoff distribution, so the choice of boundary policy has
no measurable effect on the at-the-money price. The substitution
$V_{\text{new}}[0] = 2 V_{\text{new}}[1] - V_{\text{new}}[2]$ folds into
the row-$1$ equation of the tridiagonal system, and analogously at the
right end.

### Discrete observation events

The terminal condition is imposed at $t = t_M$ (final valuation), not at the
maturity payment date. The cashflow at the payment date $T_{\text{pay},M}$
is locked in by $S(t_M)$, so we set

$$V(S, t_M) = \big[R(S) + c_M(S)\big]\cdot e^{-r(T_{\text{pay},M} - t_M)},$$

where $R(S) = N$ if $S/S_0 \ge B_{ki}$ and $R(S) = N \cdot S/S_0$ otherwise
(non-geared), and $c_M(S)$ is the final coupon (flat or conditional on
$S/S_0 \ge B_c$). Walking backward, between consecutive autocall fixings
the PDE evolves freely. At each autocall observation $t_j$, $j < n_{ac}$:

- **Autocall region** ($S/S_0 \ge B_{ac}$): replace $V(S, t_j^-)$ with
  $(N + c_j(S))\cdot e^{-r(T_{\text{pay},j} - t_j)}$ — the note dies, the
  cashflow is deterministic.
- **Continuation region** ($S/S_0 < B_{ac}$): add the coupon,
  $V(S, t_j^-) \mathrel{+}= c \cdot e^{-r(T_{\text{pay},j} - t_j)} \cdot \mathbb{1}\{S/S_0 \ge B_c\}$
  (or unconditionally for flat coupons).

The price at issue is $V(0, t=0)$, read off the grid by interpolation at
$x = 0$. With the snap-to-zero grid construction this is just an exact
lookup at the central node.

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
