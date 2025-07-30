# Tous les graphiques Plotly

import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd

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

# TODO: something's wrong, the number do not add up
# def plot_cash_usage_breakdown(
#     cumulative_flows,
#     outflows_buy,
#     inflows_sell,
#     df_margin_call,
#     outflows_transac_fees,
#     outflows_repo,
#     cash_costs,
#     cash_gains,
# ):
#     # Création du graphique Plotly
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=cumulative_flows.index,
#         y=outflows_buy.sum(axis=1).cumsum(),
#         name='Achats (open long ; cover short)',
#         stackgroup='out',
#         mode='none',
#         line=dict(color='blue')
#     ))

#     fig.add_trace(go.Scatter(
#         x=cumulative_flows.index,
#         y=inflows_sell.sum(axis=1).cumsum(),
#         name='Ventes (close long ; open short))',
#         stackgroup='in',
#         mode='none',
#         line=dict(color='green')
#     ))

#     fig.add_trace(go.Scatter(
#         x=cumulative_flows.index,
#         y=cash_costs.cumsum(),
#         name='Cash cost',
#         stackgroup='out',
#         mode='none',
#         # line=dict(color='blue')
#     ))

#     fig.add_trace(go.Scatter(
#         x=cumulative_flows.index,
#         y=cash_gains.cumsum(),
#         name='Cash gain',
#         stackgroup='in',
#         mode='none',
#         # line=dict(color='green')
#     ))


#     fig.add_trace(go.Scatter(
#         x=cumulative_flows.index,
#         y=df_margin_call.sum(axis=1).cumsum(),
#         name='Collat + Margin call',
#         stackgroup='out',
#         mode='none',
#         line=dict(color='orange')
#     ))

#     fig.add_trace(go.Scatter(
#         x=cumulative_flows.index,
#         y=outflows_transac_fees.sum(axis=1).cumsum(),
#         name='Frais de transaction',
#         stackgroup='out',
#         mode='none',
#         line=dict(color='red')
#     ))

#     fig.add_trace(go.Scatter(
#         x=cumulative_flows.index,
#         y=outflows_repo.sum(axis=1).cumsum(),
#         name='Frais de repo',
#         stackgroup='out',
#         mode='none',
#         line=dict(color='purple')
#     ))

#     fig.add_trace(go.Scatter(
#         x=cumulative_flows.index,
#         y=cumulative_flows.values,
#         name='Cash total utilisé',
#         mode='lines',
#         line=dict(color='black', dash='dash')
#     ))

#     fig.update_layout(
#         title='Utilisation cumulée du cash par catégorie',
#         xaxis_title='Date',
#         yaxis_title='Montant (€)',
#         hovermode='x unified',
#         template='plotly_white',
#         height=600
#     )

#     return fig


def plot_distrib_ops_returns(perf_array):
    """
    Plot the histogram of operation performance distribution (in percentage).
    perf_array should contain decimal values (e.g., 0.15 for 15%).
    """
    fig = px.histogram(perf_array, 
                       title="Distribution des performances des opérations", 
                       labels={"value": "Performance (%)"}, 
                       template="plotly_dark")

    # Customization
    fig.update_traces(marker=dict(color="deepskyblue", line=dict(width=1, color="white")))
    fig.update_layout(
        xaxis=dict(
            showgrid=True, 
            gridcolor="gray", 
            tickformat=".0%",   # <- Format pourcentage
            title="Performance (%)"
        ),
        yaxis=dict(showgrid=True, gridcolor="gray", title="Fréquence"),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14)
    )

    return fig



def plot_pie_hit_ratio(dic_winners_losers_long_short) :
    ## On plot le graph camembert des hits ratio

    data_graph = dic_winners_losers_long_short

    # Créer un graphique en camembert
    fig = go.Figure(data=[go.Pie(labels=list(data_graph.keys()), values=list(data_graph.values()), hole=0.3, textinfo='percent+label')])

    # Personnalisation du graphique
    fig.update_layout(
        title=f"Répartition trades gagnants/perdants en fonction de la position (Long/Short)",
        template="plotly_dark",  # Thème sombre
        annotations=[dict(
            x=0.5,  # Position du texte au centre du graphique
            y=0.5,
            text="Total",
            font_size=20,
            showarrow=False
        )],
        showlegend=True
    )

    # Afficher le graphique
    return fig


def plot_volume_against_perf(df_metrics_per_ticker):
    """
    Crée un scatter plot du total return en fonction du volume $ attribué à chaque ticker.
    Affiche les infos au survol.
    """
    # Vérification et conversion en DataFrame si nécessaire
    if not isinstance(df_metrics_per_ticker, pd.DataFrame):
        df_metrics_per_ticker = pd.DataFrame(df_metrics_per_ticker)

    df_graph = df_metrics_per_ticker.dropna().reset_index()

    # Vérifier si la colonne "Nom" existe, sinon ne pas l'inclure dans hover_data
    hover_columns = ["Volume $", "Total_return"]
    if "index" in df_graph.columns:
        hover_columns.append("index")  # Ajoute "Nom" si disponible

    # Création du scatter plot
    fig = px.scatter(
        df_graph, x="Volume $", y="Total_return",
        title="Total return par ticker en fonction du volume $ qui lui a été attribué",
        labels={"Volume $": "Volume $", "Total_return": "Total_return"},
        template="plotly_dark",
        hover_data=hover_columns  # Afficher les colonnes existantes
    )

    # Personnalisation des points
    fig.update_traces(
        marker=dict(size=10, color="deepskyblue", line=dict(width=1, color="white"))
    )

    # Personnalisation globale
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis_tickformat=".0%",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=14)
    )

    # Affichage
    return fig



def plot_factor_exposition(coef_dict) :

    # Trier par valeur absolue et ne garder que les 10 plus importants
    top_coef_items = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    # Extraire noms et valeurs
    factors = [k for k, _ in top_coef_items]
    coefs = [v for _, v in top_coef_items]

    # Créer le bar plot
    fig = go.Figure(data=[go.Bar(
        x=factors,
        y=coefs,
        marker_color=['red' if c < 0 else 'green' for c in coefs]
    )])

    fig.update_layout(
        title='Top 10 coefficients Ridge par facteur (valeur absolue)',
        xaxis_title='Facteurs',
        yaxis_title='Coefficient',
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        template='plotly_white'
    )

    return fig