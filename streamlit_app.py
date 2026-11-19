"""Market Risk Lab — tableau de bord VaR, simulation de crises et backtesting."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk.backtest import FRTB_MAX_EXCEPTIONS, basel_zone, run_backtest
from risk.measures import MODEL_LABELS, diversification, expected_shortfall, historical_var, rolling_var
from risk.simulation import DESKS, PORTFOLIO_VAR_LIMIT, simulate_pnl
from risk.stress import (
    ALLOCATED_CAPITAL,
    DV01,
    FACTORS,
    SCENARIOS,
    STRESS_LIMIT,
    breaking_multiplier,
    stress_pnl,
)

st.set_page_config(page_title="Market Risk Lab", page_icon=":bar_chart:", layout="wide")

BLUE, ORANGE, AQUA, RED, CRIT, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#d03b3b", "#5b6270", "#9aa0aa"
MODEL_COLORS = {"historical": BLUE, "parametric": ORANGE, "ewma": AQUA}
WINDOW = 250


def eur(x: float, d: int = 1, sign: bool = False) -> str:
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−") + " M€"


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="x unified",
        separators=", ",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(zeroline=True)
    return fig


# --------------------------------------------------------------------------- paramètres
with st.sidebar:
    st.header("Market Risk Lab")
    st.caption("Portefeuille fictif, P&L simulés. Modifiez les paramètres pour régénérer l'historique.")
    seed = st.number_input("Graine aléatoire", value=218, step=1)
    nu = st.slider("Épaisseur des queues (ddl Student t)", 3.0, 30.0, 4.0, 0.5, help="Plus le nombre est faible, plus les queues sont épaisses.")
    stress_on = st.toggle("Épisode de stress de volatilité", value=True)
    intensity = st.slider("Intensité du stress", 2.0, 12.0, 7.0, 0.5, disabled=not stress_on)


@st.cache_data
def load(seed: int, nu: float, stress_on: bool, intensity: float) -> pd.DataFrame:
    return simulate_pnl(seed=int(seed), nu=nu, stress_start=150 if stress_on else None, stress_intensity=intensity)


@st.cache_data
def var_series(pnl: pd.Series, model: str, confidence: float) -> pd.Series:
    return rolling_var(pnl, model, confidence, WINDOW)


desk_pnl = load(seed, nu, stress_on, intensity)
pnl = desk_pnl.sum(axis=1).rename("Portefeuille")
last = pnl.iloc[-WINDOW:]
asof = pnl.index[-1]

st.title("Market Risk Lab")
st.caption(f"Portefeuille Marchés EUR · arrêté au {asof:%d/%m/%Y} · horizon 1 jour · fenêtre {WINDOW} jours · données simulées")

tab_dash, tab_stress, tab_bt, tab_method = st.tabs(["Tableau de bord", "Simulation de crises", "Backtesting", "Méthodologie"])

# --------------------------------------------------------------------------- tableau de bord
with tab_dash:
    var99 = historical_var(last, 0.99)
    es975 = expected_shortfall(last, 0.975)
    hist99 = var_series(pnl, "historical", 0.99)
    n_exc = run_backtest(pnl.iloc[-WINDOW:], hist99.iloc[-WINDOW:], 0.99).n_exceptions
    zone, plus = basel_zone(n_exc)

    c = st.columns(5)
    c[0].metric("VaR 99 % · 1 j", eur(var99), help="VaR historique, 250 scénarios", border=True, delta_arrow="off")
    c[1].metric("ES 97,5 % · 1 j", eur(es975), help="Moyenne des 7 pires P&L sur 250 jours", border=True, delta_arrow="off")
    c[2].metric("P&L du jour", eur(pnl.iloc[-1], sign=True), border=True, delta_arrow="off")
    c[3].metric("Utilisation limite VaR", f"{var99 / PORTFOLIO_VAR_LIMIT:.0%}".replace("%", " %"), f"limite {eur(PORTFOLIO_VAR_LIMIT, 0)}", delta_color="off", border=True, delta_arrow="off")
    c[4].metric("Exceptions 250 j", n_exc, f"zone {zone} · k = {3 + plus:.2f}".replace(".", ","), delta_color="off", border=True, delta_arrow="off")

    left, right = st.columns(2)
    with left:
        st.subheader("VaR 99 % sur 250 jours")
        h = hist99.iloc[-WINDOW:]
        fig = go.Figure(go.Scatter(x=h.index, y=h, name="VaR 99 %", line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(42,120,214,.12)"))
        fig.add_hline(y=PORTFOLIO_VAR_LIMIT, line=dict(color=CRIT, dash="dash"), annotation_text=f"Limite {eur(PORTFOLIO_VAR_LIMIT, 0)}", annotation_position="top left")
        fig.update_yaxes(title="M€", rangemode="tozero")
        st.plotly_chart(style(fig), width="stretch")
    with right:
        st.subheader("Distribution des P&L quotidiens")
        fig = go.Figure()
        fig.add_histogram(x=last[last >= -var99], xbins=dict(size=1), marker_color=BLUE, opacity=0.6, name="P&L")
        fig.add_histogram(x=last[last < -var99], xbins=dict(size=1), marker_color=RED, name="Queue au-delà de la VaR 99 %")
        fig.add_vline(x=-var99, line=dict(color=INK, width=2), annotation_text=f"VaR 99 % {eur(-var99)}", annotation_position="top right")
        fig.add_vline(x=-es975, line=dict(color=ORANGE, width=2, dash="dash"), annotation_text=f"ES 97,5 % {eur(-es975)}", annotation_position="top left")
        fig.update_layout(barmode="overlay", hovermode="closest")
        fig.update_xaxes(title="P&L (M€)")
        fig.update_yaxes(title="Jours")
        st.plotly_chart(style(fig), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("VaR par desk et diversification")
        standalone, total = diversification(desk_pnl.iloc[-WINDOW:], 0.99)
        limits = pd.Series({d.name: d.var_limit for d in DESKS})
        table = pd.DataFrame({"VaR 99 % (M€)": standalone, "Limite (M€)": limits, "Utilisation": 100 * standalone / limits})
        table.loc["Somme des VaR"] = [standalone.sum(), float("nan"), float("nan")]
        table.loc["Diversification"] = [total - standalone.sum(), float("nan"), float("nan")]
        table.loc["Portefeuille"] = [total, PORTFOLIO_VAR_LIMIT, 100 * total / PORTFOLIO_VAR_LIMIT]
        st.dataframe(
            table,
            width="stretch",
            column_config={
                "VaR 99 % (M€)": st.column_config.NumberColumn(format="%.1f"),
                "Limite (M€)": st.column_config.NumberColumn(format="%.1f"),
                "Utilisation": st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=150),
            },
        )
    with right:
        st.subheader("Sensibilités taux (DV01, k€/pb)")
        dv = DV01.copy()
        dv["Total"] = dv.sum(axis=1)
        dv.loc["Total"] = dv.sum()
        fig = go.Figure(go.Heatmap(z=DV01.to_numpy(), x=DV01.columns, y=DV01.index, colorscale=[[0, RED], [0.5, "#eceef1"], [1, BLUE]], zmid=0, zmin=-18, zmax=18, text=DV01.to_numpy(), texttemplate="%{text:+d}", hovertemplate="%{y} · %{x}<br>%{z:+d} k€ pour +1 pb<extra></extra>", colorbar=dict(title="k€/pb")))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 280), width="stretch")
        st.caption(f"DV01 total du portefeuille : {dv.loc['Total', 'Total']:+.0f} k€ par point de base.")

# --------------------------------------------------------------------------- stress tests
with tab_stress:
    st.markdown("Chocs instantanés sur six facteurs, calibrés sur la pire fenêtre de 10 jours de chaque épisode. Revalorisation par sensibilités (delta, gamma actions, vega).")
    names = [s.name for s in SCENARIOS] + ["Personnalisé"]
    choice = st.radio("Scénario", names, horizontal=True)
    base = next((s.shocks for s in SCENARIOS if s.name == choice), SCENARIOS[0].shocks)

    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("**Chocs appliqués**")
        shocks = {}
        for f in FACTORS:
            shocks[f.key] = st.slider(
                f"{f.label} ({f.unit})", float(f.lo), float(f.hi), float(base[f.key]), float(f.step),
                key=f"shock-{f.key}-{choice}", disabled=choice != "Personnalisé", format="%.1f" if f.step < 1 else "%.0f",
            )
    grid = stress_pnl(shocks)
    total = float(grid.to_numpy().sum())
    with col_out:
        m = st.columns(3)
        m[0].metric("P&L stressé", eur(total, sign=True), border=True, delta_arrow="off")
        m[1].metric("Consommation de la limite", f"{-total / STRESS_LIMIT:.0%}".replace("%", " %"), f"limite {eur(-STRESS_LIMIT, 0)}", delta_color="off", border=True, delta_arrow="off")
        m[2].metric("Fonds propres alloués", f"{-total / ALLOCATED_CAPITAL:.1%}".replace("%", " %").replace(".", ","), eur(ALLOCATED_CAPITAL, 0), delta_color="off", border=True, delta_arrow="off")
        k = breaking_multiplier(shocks)
        if total <= -STRESS_LIMIT:
            st.error(f"Limite de perte en stress dépassée. Stress inversé : la limite est atteinte dès {k:.0%} de l'amplitude du scénario.".replace("%", " %"))
        elif k is not None:
            st.info(f"Stress inversé : il faudrait amplifier ce scénario ×{k:.2f} pour atteindre la limite.".replace(".", ","))
        else:
            st.success("Le scénario reste sous la limite même amplifié huit fois.")

        by_factor = grid.sum()
        fig = go.Figure(go.Waterfall(
            x=list(by_factor.index) + ["Total"], y=list(by_factor.values) + [0],
            measure=["relative"] * len(by_factor) + ["total"],
            decreasing=dict(marker_color=RED), increasing=dict(marker_color=BLUE), totals=dict(marker_color=INK),
            texttemplate="%{delta:+.1f}", connector=dict(line=dict(color=GREY, width=1)), cliponaxis=False,
        ))
        fig.add_hline(y=-STRESS_LIMIT, line=dict(color=CRIT, dash="dash"), annotation_text="Limite de perte en stress", annotation_position="top left")
        running = by_factor.cumsum()
        fig.update_yaxes(title="M€", range=[min(-STRESS_LIMIT, running.min(), 0) * 1.15, max(running.max(), 0) * 1.15 + 3])
        fig.update_layout(hovermode="closest")
        st.plotly_chart(style(fig, 340), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Impact par desk")
        by_desk = grid.sum(axis=1).sort_values()
        fig = go.Figure(go.Bar(x=by_desk, y=by_desk.index, orientation="h", marker_color=[RED if v < 0 else BLUE for v in by_desk], text=[f"{v:+.1f}" for v in by_desk], textposition="outside", cliponaxis=False))
        pad = max(2.0, by_desk.abs().max() * 0.15)  # place pour les étiquettes
        fig.update_xaxes(title="M€", range=[min(0.0, by_desk.min()) - pad, max(0.0, by_desk.max()) + pad])
        fig.update_yaxes(ticksuffix="  ")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(style(fig, 280), width="stretch")
    with right:
        st.subheader("Bibliothèque de scénarios")
        lib = pd.DataFrame(
            [{"Scénario": s.name, "Période": s.period, "P&L (M€)": float(stress_pnl(s.shocks).to_numpy().sum())} for s in SCENARIOS]
        )
        lib["% de la limite"] = -100 * lib["P&L (M€)"] / STRESS_LIMIT
        st.dataframe(
            lib, hide_index=True, width="stretch",
            column_config={
                "P&L (M€)": st.column_config.NumberColumn(format="%.1f"),
                "% de la limite": st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=150),
            },
        )
    with st.expander("Matrice détaillée desk × facteur (M€)"):
        st.dataframe(grid.round(2), width="stretch")

# --------------------------------------------------------------------------- backtesting
with tab_bt:
    st.markdown("P&L hypothétique comparé à la VaR de la veille sur 250 jours. Une exception est une perte supérieure à la VaR.")
    a, b = st.columns([2, 1])
    model = a.segmented_control("Modèle", list(MODEL_LABELS), format_func=MODEL_LABELS.get, default="historical") or "historical"
    cl = b.segmented_control("Niveau de confiance", [0.99, 0.975], format_func=lambda c: f"{c:.1%}".replace(".", ",").replace("%", " %"), default=0.99) or 0.99

    v = var_series(pnl, model, cl).iloc[-WINDOW:]
    p = pnl.iloc[-WINDOW:]
    hit = -p > v
    fig = go.Figure()
    fig.add_bar(x=p.index, y=p, name="P&L quotidien", marker_color=[CRIT if h else BLUE for h in hit], opacity=0.7)
    fig.add_scatter(x=v.index, y=-v, name="−VaR (veille)", line=dict(color=ORANGE, width=2, shape="hv"))
    fig.add_scatter(x=p.index[hit], y=p[hit], mode="markers", name="Exception", marker=dict(color=CRIT, size=9, line=dict(color="white", width=1.5)))
    fig.update_yaxes(title="M€")
    st.plotly_chart(style(fig, 380), width="stretch")

    res = run_backtest(p, v, cl)
    c = st.columns(4)
    if cl == 0.99:
        zone, plus = basel_zone(res.n_exceptions)
        c[0].metric("Feux tricolores Bâle", f"Zone {zone}", f"{res.n_exceptions} exceptions · k = {3 + plus:.2f}".replace(".", ","), delta_color="off", border=True, delta_arrow="off")
    else:
        ok = res.n_exceptions <= FRTB_MAX_EXCEPTIONS[cl]
        c[0].metric("Seuil FRTB par desk", f"{res.n_exceptions} / {FRTB_MAX_EXCEPTIONS[cl]}", "éligible modèle interne" if ok else "inéligible", delta_color="off", border=True, delta_arrow="off")
    c[1].metric("Exceptions", res.n_exceptions, f"attendu {res.expected:.1f}".replace(".", ","), delta_color="off", border=True, delta_arrow="off")
    c[2].metric("Kupiec (couverture)", f"p = {res.kupiec_p:.3f}".replace(".", ","), "rejet à 5 %" if res.kupiec_p < 0.05 else "non rejeté", delta_color="inverse" if res.kupiec_p < 0.05 else "normal", border=True, delta_arrow="off")
    c[3].metric("Christoffersen (indépendance)", f"p = {res.ind_p:.3f}".replace(".", ","), "rejet à 5 %" if res.ind_p < 0.05 else "non rejeté", delta_color="inverse" if res.ind_p < 0.05 else "normal", border=True, delta_arrow="off")

    st.subheader("Comparaison des modèles")
    rows, cum = [], go.Figure()
    for mo, label in MODEL_LABELS.items():
        vm = var_series(pnl, mo, cl).iloc[-WINDOW:]
        r = run_backtest(p, vm, cl)
        rows.append({"Modèle": label, "VaR moyenne (M€)": vm.mean(), "Exceptions": r.n_exceptions, "Kupiec p": r.kupiec_p, "Indépendance p": r.ind_p, "Conditionnel p": r.conditional_p})
        cum.add_scatter(x=p.index, y=(-p > vm).cumsum(), name=label, line=dict(color=MODEL_COLORS[mo], width=3 if mo == model else 1.75, shape="hv"))
    cum.add_scatter(x=[p.index[0], p.index[-1]], y=[0, WINDOW * (1 - cl)], name="Attendu", line=dict(color=GREY, dash="dot"))
    cum.update_yaxes(title="Exceptions cumulées")
    fmt = {k: st.column_config.NumberColumn(format="%.3f") for k in ["Kupiec p", "Indépendance p", "Conditionnel p"]}
    fmt["VaR moyenne (M€)"] = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=fmt)
    st.plotly_chart(style(cum, 300), width="stretch")

# --------------------------------------------------------------------------- méthodologie
with tab_method:
    st.markdown(
        r"""
### Données
Les P&L quotidiens de cinq desks sont simulés. La volatilité suit un **GARCH(1,1)** commun
($\alpha = 0{,}08$, $\beta = 0{,}90$) et les innovations une **loi de Student** à $\nu$ degrés de liberté,
ce qui produit du regroupement de volatilité et des queues épaisses. Un épisode de stress multiplie
la variance de long terme pendant 22 jours, 150 jours avant la date d'arrêté.

### Mesures de risque (horizon 1 jour, fenêtre de 250 jours)
| Mesure | Formule |
|---|---|
| VaR historique | $\mathrm{VaR}_c = -q_{1-c}(\mathrm{P\&L}_{t-250..t-1})$ |
| VaR paramétrique | $\mathrm{VaR}_c = z_c \, \hat\sigma$ |
| VaR EWMA | $\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2$, $\lambda = 0{,}94$ (RiskMetrics) |
| Expected Shortfall | moyenne des $\lceil 250 \times 2{,}5\,\% \rceil = 7$ pires P&L |

### Stress tests
Revalorisation par sensibilités : $\Delta P = \sum_i \delta_i \, s_i + \tfrac12 \Gamma\, s_{\text{actions}}^2 + \nu\, s_{\text{vol}}$.
Le **stress inversé** cherche par dichotomie le multiplicateur $k$ tel que $\Delta P(k \cdot s) = -\text{limite}$.

### Backtesting
- **Kupiec (POF)** : $LR_{uc} = -2\ln\frac{(1-p)^{T-x}p^x}{(1-\hat p)^{T-x}\hat p^x} \sim \chi^2_1$
- **Christoffersen** : indépendance des exceptions (chaîne de Markov d'ordre 1), $LR_{ind} \sim \chi^2_1$
- **Feux tricolores Bâle** (99 %, 250 jours) : vert 0–4, jaune 5–9, rouge ≥ 10 exceptions ; multiplicateur $3 + k$
- **FRTB** : au plus 12 exceptions à 99 % et 30 à 97,5 % par desk
"""
    )
