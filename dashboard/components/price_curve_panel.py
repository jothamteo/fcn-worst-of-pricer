"""Price-vs-spot curve panel — FCN value as a function of basket-wide spot.

Reads from the pre-computed scenario grid: ``axes.basket_spot_shift`` is the
x-axis, evaluated at the current ``(basket_vol_shift, corr_shift)`` projection.
Δ and Γ curves are central-difference / second-difference of the same
``ScenarioGrid.price`` slice along the spot axis.

The plot annotates the FCN's two key barriers (autocall at 100% of issue,
knock-in at 70%) and shades the autocall (green) / KI (red) regions so a
reader who hasn't seen an FCN before can read the payoff geometry off the
chart.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.engine import ensure_grid


def render() -> None:
    st.subheader("Price & Greek curves vs spot")
    st.caption(
        "FCN price, Δ, and Γ as a function of a basket-wide spot shift. Vertical "
        "lines mark the autocall barrier (100% of issue) and the knock-in / "
        "strike (70%); shaded regions show the autocall (green) and KI (red) zones."
    )

    initial = st.session_state["initial"]
    current = st.session_state["current"]
    product = st.session_state["product"]

    grid = ensure_grid()
    pt = grid.project(current)              # (s_shift, v_shift, c_shift)
    v_shift, c_shift = pt[1], pt[2]

    # Locate the nearest (v, c) grid indices for the 1D slice along the spot axis.
    vi = int(np.argmin(np.abs(grid.axes.basket_vol_shift - v_shift)))
    ci = int(np.argmin(np.abs(grid.axes.corr_shift - c_shift)))

    s_axis = grid.axes.basket_spot_shift            # relative shifts, e.g. -0.30 = -30%
    price_curve = grid.price[:, vi, ci]
    perf = 1.0 + s_axis                             # = S_basket / S_basket_initial

    # Δ and Γ along the basket-spot axis — central differences in s_axis units.
    # Δ is in price-per-relative-shift; convert to price-per-absolute-basket-dollar
    # by dividing by mean(initial.spots) so it's a comparable spot delta.
    dprice = np.gradient(price_curve, s_axis)
    d2price = np.gradient(dprice, s_axis)
    basket_mean_spot = float(np.mean(initial.spots))
    delta_curve = dprice / basket_mean_spot
    gamma_curve = d2price / (basket_mean_spot ** 2)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=("Price", "Δ (per $1 of basket spot)", "Γ (per $1² of basket spot)"),
        row_heights=[0.5, 0.25, 0.25],
    )
    fig.add_trace(go.Scatter(x=perf, y=price_curve, mode="lines", name="Price",
                             line=dict(width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=perf, y=delta_curve, mode="lines", name="Δ",
                             line=dict(width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=perf, y=gamma_curve, mode="lines", name="Γ",
                             line=dict(width=2)), row=3, col=1)

    # Vertical barrier markers + shaded zones (price subplot only, but x-axes are shared)
    autocall_x = product.autocall_barrier
    ki_x = product.strike
    x_min, x_max = float(perf.min()), float(perf.max())

    fig.add_vline(x=autocall_x, line=dict(color="green", dash="dash"),
                  annotation_text="Autocall 100%", annotation_position="top right")
    fig.add_vline(x=ki_x, line=dict(color="red", dash="dash"),
                  annotation_text="KI 70%", annotation_position="top right")

    # Shaded regions
    fig.add_vrect(x0=autocall_x, x1=x_max, fillcolor="green", opacity=0.07,
                  layer="below", line_width=0, row=1, col=1)
    fig.add_vrect(x0=x_min, x1=ki_x, fillcolor="red", opacity=0.08,
                  layer="below", line_width=0, row=1, col=1)

    # Current spot marker
    cur_perf = 1.0 + pt[0]
    fig.add_vline(x=cur_perf, line=dict(color="black", dash="dot"),
                  annotation_text=f"Today ({100 * pt[0]:+.0f}%)",
                  annotation_position="bottom right")

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Basket performance S/S₀", row=3, col=1, tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Legend — what the colours mean", expanded=False):
        st.markdown(
            "- **Green band (S/S₀ ≥ 100%)**: autocall zone. At any observation "
            "date the note will redeem at par and pay its accrued coupon; the "
            "FCN price plateaus near `N · (1 + coupon)` deep into this region.\n"
            "- **Red band (S/S₀ ≤ 70%)**: knock-in zone. The worst-of has "
            "broken the 70% strike at maturity; redemption falls 1:1 with the "
            "basket performance, which is what gives the FCN its short-put "
            "shape and most of its risk.\n"
            "- **Black dotted line**: the current basket position. Move the "
            "spot slider in the sidebar and the line glides across the curve.\n"
            "- **Δ slope** is the spot delta in dollars-per-dollar of basket "
            "spot. **Γ** is its derivative — and the kinks near the barriers "
            "are why this product is sensitive to barrier-noise in MC Greeks "
            "(see notebook 08 in the repo for the smoothed-payoff fix)."
        )

    st.caption(
        f"Slice taken at basket-vol shift = {v_shift:+.2f}, "
        f"correlation shift = {c_shift:+.2f}, projected from the current snapshot."
    )
