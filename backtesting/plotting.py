# Tous les graphiques Plotly

import plotly.graph_objs as go
import plotly.express as px
import numpy as np


def plot_cumulative_pnl(cumulative_pnl_portfolio):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_pnl_portfolio.index,
        y=cumulative_pnl_portfolio.values,
        name="PnL"
    ))
    fig.update_layout(
        title="Cumulative PnL",
        xaxis_title="Date",
        yaxis_title="PnL (€)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def plot_drawdown(drawdown):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=drawdown, name="Drawdown", line_color="red"))
    fig.update_layout(title="Drawdown")
    return fig


def plot_cash_usage_breakdown(
    cumulative_flows,
    outflows_buy,
    inflows_sell,
    df_margin_call,
    outflows_transac_fees,
    outflows_repo,
    cash_costs,
    cash_gains,
):
    # Création du graphique Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cumulative_flows.index,
        y=outflows_buy.sum(axis=1).cumsum(),
        name='Achats (open long ; cover short)',
        stackgroup='out',
        mode='none',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=cumulative_flows.index,
        y=inflows_sell.sum(axis=1).cumsum(),
        name='Ventes (close long ; open short))',
        stackgroup='in',
        mode='none',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=cumulative_flows.index,
        y=cash_costs.cumsum(),
        name='Cash cost',
        stackgroup='out',
        mode='none',
        # line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=cumulative_flows.index,
        y=cash_gains.cumsum(),
        name='Cash gain',
        stackgroup='in',
        mode='none',
        # line=dict(color='green')
    ))


    fig.add_trace(go.Scatter(
        x=cumulative_flows.index,
        y=df_margin_call.sum(axis=1).cumsum(),
        name='Collat + Margin call',
        stackgroup='out',
        mode='none',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=cumulative_flows.index,
        y=outflows_transac_fees.sum(axis=1).cumsum(),
        name='Frais de transaction',
        stackgroup='out',
        mode='none',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=cumulative_flows.index,
        y=outflows_repo.sum(axis=1).cumsum(),
        name='Frais de repo',
        stackgroup='out',
        mode='none',
        line=dict(color='purple')
    ))

    fig.add_trace(go.Scatter(
        x=cumulative_flows.index,
        y=cumulative_flows.values,
        name='Cash total utilisé',
        mode='lines',
        line=dict(color='black', dash='dash')
    ))

    fig.update_layout(
        title='Utilisation cumulée du cash par catégorie',
        xaxis_title='Date',
        yaxis_title='Montant (€)',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )

    return fig

def plot_histo_returns_distrib(df_perf) :
    ## Plot l'histogramme de la répartition des perf

    raw_data_perf = df_perf.values.flatten() * 100
    data_cleaned = raw_data_perf[~np.isnan(raw_data_perf)]

    # Création de l'histogramme
    fig = px.histogram(data_cleaned, 
                    title="Distribution des perf des opérations", 
                    labels={"value": "Performance en %"}, 
                    template="plotly_dark")

    # Personnalisation
    fig.update_traces(marker=dict(color="deepskyblue", line=dict(width=1, color="white")))
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis=dict(showgrid=True, gridcolor="gray", title="Freq"),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14)
    )

    # Affichage
    return fig

