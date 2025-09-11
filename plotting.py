import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb, qualitative, sequential


def plot_curves_percent_with_months(
    df_wide: pd.DataFrame,
    title: str,
    show_legend: bool = True,
    legend_limit: int = 40,
    palette: str = "Gradient",
    base_color: str | None = None,
    line_width: int = 1,
):
    if df_wide.empty:
        return None

    idx = df_wide.index.to_numpy()
    name = (df_wide.index.name or "").upper()
    x_months = idx * 3 if name == "QOB" else idx

    Y = df_wide.to_numpy(dtype='float32')
    M, N = Y.shape
    target_points = 200_000
    step = max(1, int(np.ceil((M * N) / target_points)))
    if step > 1:
        x_months = x_months[::step]
        Y = Y[::step, :]

    def _generate_palette(base: str, n: int) -> list[str]:
        r, g, b = hex_to_rgb(base)
        palette = []
        for i in range(n):
            ratio = 0.2 + 0.8 * (i / max(n - 1, 1))
            ri = int(r + (255 - r) * ratio)
            gi = int(g + (255 - g) * ratio)
            bi = int(b + (255 - b) * ratio)
            palette.append(f'rgb({ri},{gi},{bi})')
        return palette

    palette_choice = palette.lower()
    base_color = base_color or "#1f77b4"
    if palette_choice == "gradient":
        palette_colors = _generate_palette(base_color, N)
    elif palette_choice == "plotly":
        palette_colors = qualitative.Plotly
    elif palette_choice == "viridis":
        palette_colors = sequential.Viridis
    else:
        palette_colors = qualitative.Plotly
    if len(palette_colors) < N:
        times = int(np.ceil(N / len(palette_colors)))
        palette_colors = (palette_colors * times)[:N]
    else:
        palette_colors = palette_colors[:N]

    fig = go.Figure()
    show_leg = show_legend and (df_wide.shape[1] <= legend_limit)
    for i, col in enumerate(df_wide.columns):
        fig.add_trace(go.Scatter(
            x=x_months,
            y=Y[:, i],
            mode='lines',
            name=str(col),
            line=dict(color=palette_colors[i], width=line_width),
            hovertemplate=f"Vintage: {col}<br>Month: %{{x}}<br>Default rate: %{{y:.2%}}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Deal Age (months)',
        yaxis=dict(title='Cumulative default rate', tickformat='.2%'),
        hovermode='x unified',
        showlegend=show_leg,
        legend=dict(bgcolor="white", font=dict(color="black")),
    )

    return fig
