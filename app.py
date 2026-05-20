import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Intelligence vs. Cost to Run", layout="wide")
st.title("Intelligence vs. Cost to Run — Artificial Analysis Intelligence Index")
st.caption("Artificial Analysis Intelligence Index · Cost to run Intelligence Index")


@st.cache_data(ttl=3600)
def fetch_openrouter_models():
    """Fetch model list and pricing from OpenRouter API."""
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]

    models = []
    for m in data:
        pricing = m.get("pricing", {})
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")

        if prompt_price is None or completion_price is None:
            continue

        try:
            prompt_price = float(prompt_price)
            completion_price = float(completion_price)
        except (ValueError, TypeError):
            continue

        # Skip free/router models and negative pricing
        if prompt_price <= 0 and completion_price <= 0:
            continue
        if prompt_price < 0 or completion_price < 0:
            continue

        # Blended price per million tokens (3:1 input:output ratio)
        blended_per_token = (3 * prompt_price + 1 * completion_price) / 4
        blended_per_million = blended_per_token * 1_000_000

        models.append(
            {
                "id": m["id"],
                "name": m["name"],
                "prompt_price_per_m": prompt_price * 1_000_000,
                "completion_price_per_m": completion_price * 1_000_000,
                "blended_price_per_m": blended_per_million,
                "context_length": m.get("context_length", 0),
                "provider": extract_provider(m["id"]),
            }
        )

    return pd.DataFrame(models)


def extract_provider(model_id: str) -> str:
    """Extract provider name from model ID."""
    provider_map = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Google",
        "deepseek": "DeepSeek",
        "meta-llama": "Meta",
        "mistralai": "Mistral",
        "qwen": "Alibaba",
        "x-ai": "xAI",
        "amazon": "Amazon",
        "nvidia": "NVIDIA",
        "minimax": "MiniMax",
        "moonshotai": "Kimi",
        "xiaomi": "Xiaomi",
        "z-ai": "Z AI",
        "cohere": "Cohere",
        "bytedance-seed": "ByteDance",
        "tencent": "Tencent",
        "inclusionai": "InclusionAI",
    }
    prefix = model_id.split("/")[0].lstrip("~")
    return provider_map.get(prefix, prefix.title())


@st.cache_data(ttl=3600)
def fetch_artificial_analysis_models():
    """Fetch intelligence index data from Artificial Analysis API."""
    import os

    api_key = os.environ.get("AA_API_KEY", "")
    resp = requests.get(
        "https://artificialanalysis.ai/api/v2/data/llms/models",
        headers={"x-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["data"]

    models = []
    for m in data:
        evals = m.get("evaluations") or {}
        intelligence = evals.get("artificial_analysis_intelligence_index")
        if intelligence is None:
            continue

        coding = evals.get("artificial_analysis_coding_index")

        creator = m.get("model_creator") or {}
        models.append(
            {
                "aa_name": m["name"],
                "aa_slug": m["slug"],
                "creator_slug": creator.get("slug", ""),
                "creator_name": creator.get("name", ""),
                "intelligence_index": intelligence,
                "coding_index": coding,
            }
        )

    return pd.DataFrame(models)


# Mapping from AA creator slugs to OpenRouter provider prefixes
AA_TO_OPENROUTER_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "deepseek": "deepseek",
    "meta": "meta-llama",
    "mistral": "mistralai",
    "alibaba": "qwen",
    "xai": "x-ai",
    "aws": "amazon",
    "nvidia": "nvidia",
    "minimax": "minimax",
    "kimi": "moonshotai",
    "xiaomi": "xiaomi",
    "zai": "z-ai",
    "cohere": "cohere",
    "bytedance_seed": "bytedance-seed",
    "tencent": "tencent",
    "inclusionai": "inclusionai",
    "inception": "inception",
}


def _normalize_slug(s):
    """Normalize a slug by replacing dots with hyphens and lowercasing."""
    return s.lower().replace(".", "-").rstrip("-")


def match_models(openrouter_df, aa_df):
    """Match OpenRouter models with Artificial Analysis intelligence data.

    Normalizes slugs (replacing '.' with '-') so that AA slug 'gpt-5-5'
    matches OpenRouter 'gpt-5.5'. Uses longest-match to pick the most
    specific OpenRouter model for each AA entry.
    """
    # Build a lookup: openrouter prefix -> list of (normalized_model_part, or_row)
    or_by_prefix = {}
    for _, row in openrouter_df.iterrows():
        or_id = row["id"]
        # Skip :free variants
        if ":" in or_id:
            continue
        # Skip aliases (~prefix)
        if or_id.startswith("~"):
            continue
        prefix = or_id.split("/")[0]
        model_part = or_id.split("/", 1)[1]
        norm = _normalize_slug(model_part)
        or_by_prefix.setdefault(prefix, []).append((norm, row))

    matched = []
    seen_keys = set()

    for _, aa_row in aa_df.iterrows():
        creator_slug = aa_row["creator_slug"]
        or_prefix = AA_TO_OPENROUTER_PREFIX.get(creator_slug)
        if not or_prefix:
            continue

        candidates = or_by_prefix.get(or_prefix, [])
        aa_norm = _normalize_slug(aa_row["aa_slug"])

        # Find longest matching OR model
        best = None
        best_len = 0
        for or_norm, or_row in candidates:
            if aa_norm == or_norm or aa_norm.startswith(or_norm + "-"):
                if len(or_norm) > best_len:
                    best_len = len(or_norm)
                    best = or_row

        if best is not None:
            key = (aa_row["aa_name"], best["id"])
            if key in seen_keys:
                continue
            seen_keys.add(key)

            matched.append(
                {
                    "model_name": aa_row["aa_name"],
                    "provider": extract_provider(best["id"]),
                    "intelligence_index": aa_row["intelligence_index"],
                    "coding_index": aa_row["coding_index"],
                    "is_reasoning": False,
                    "blended_price_per_m": best["blended_price_per_m"],
                    "prompt_price_per_m": best["prompt_price_per_m"],
                    "completion_price_per_m": best["completion_price_per_m"],
                    "openrouter_id": best["id"],
                }
            )

    return pd.DataFrame(matched)


# Provider color mapping (matching the image)
PROVIDER_COLORS = {
    "Alibaba": "#FF8C00",
    "Amazon": "#DAA520",
    "Anthropic": "#D2691E",
    "DeepSeek": "#0000CD",
    "Google": "#228B22",
    "Kimi": "#4169E1",
    "MiniMax": "#FF69B4",
    "Mistral": "#FF4500",
    "NVIDIA": "#9ACD32",
    "OpenAI": "#555555",
    "xAI": "#8A2BE2",
    "Xiaomi": "#FF6347",
    "Z AI": "#4682B4",
    "Tencent": "#2E8B57",
    "Inception": "#8B4513",
    "InclusionAI": "#5F9EA0",
    "ByteDance": "#6B8E23",
    "Meta": "#B8860B",
}


def main():
    # Fetch data
    with st.spinner("Fetching model data..."):
        openrouter_df = fetch_openrouter_models()
        aa_df = fetch_artificial_analysis_models()

    # Match models
    matched_df = match_models(openrouter_df, aa_df)

    if matched_df.empty:
        st.error("No models could be matched between data sources.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    score_metric = st.sidebar.radio(
        "Score metric",
        ["Intelligence Index", "Coding Index"],
        horizontal=True,
    )
    score_col = (
        "intelligence_index" if score_metric == "Intelligence Index" else "coding_index"
    )

    providers = sorted(matched_df["provider"].unique())
    selected_providers = st.sidebar.multiselect(
        "Providers",
        providers,
        default=providers,
    )

    # Filter
    filtered_df = matched_df[matched_df["provider"].isin(selected_providers)]
    if score_col == "coding_index":
        filtered_df = filtered_df[filtered_df["coding_index"].notna()]

    # Model selection — default to top N by selected score
    st.sidebar.header("Model Selection")
    top_n = st.sidebar.number_input(
        "Number of top models to show",
        min_value=1,
        max_value=len(filtered_df),
        value=min(30, len(filtered_df)),
        step=5,
    )
    top_models = (
        filtered_df.sort_values(score_col, ascending=False)
        .head(top_n)["model_name"]
        .tolist()
    )
    all_models = sorted(filtered_df["model_name"].tolist())
    selected_models = st.sidebar.multiselect(
        "Select models to display",
        all_models,
        default=sorted(top_models),
    )

    plot_df = filtered_df[filtered_df["model_name"].isin(selected_models)].copy()

    if plot_df.empty:
        st.warning("No models selected. Use the sidebar to select models.")
        return

    # Cost metric: input + output price per 1M tokens
    plot_df["cost_per_m"] = (
        plot_df["prompt_price_per_m"] + plot_df["completion_price_per_m"]
    )

    # Create tabs for the two charts
    tab1, tab2 = st.tabs(["Intelligence vs. Cost", "Cost to Run"])

    with tab1:
        # Create the scatter plot
        fig = go.Figure()

        # "Most attractive quadrant" shading (top-left: cheap & smart)
        x_quad_min = plot_df["cost_per_m"].min() * 0.5
        x_quad_max = 1.5
        y_mid = plot_df[score_col].median()
        y_max = plot_df[score_col].max() + 5

        fig.add_shape(
            type="rect",
            x0=x_quad_min,
            x1=x_quad_max,
            y0=y_mid,
            y1=y_max,
            fillcolor="rgba(144, 238, 144, 0.15)",
            line=dict(color="rgba(144, 238, 144, 0.5)", width=1),
            layer="below",
        )

        fig.add_annotation(
            x=x_quad_min * 1.2,
            y=y_max - 1,
            text="Most attractive quadrant",
            showarrow=False,
            font=dict(size=11, color="green"),
            xanchor="left",
        )

        # Plot each provider with its own color
        for provider in sorted(plot_df["provider"].unique()):
            provider_data = plot_df[plot_df["provider"] == provider]
            color = PROVIDER_COLORS.get(provider, "#888888")

            fig.add_trace(
                go.Scatter(
                    x=provider_data["cost_per_m"],
                    y=provider_data[score_col],
                    mode="markers+text",
                    name=provider,
                    text=provider_data["model_name"],
                    textposition="top right",
                    textfont=dict(size=11),
                    marker=dict(color=color, size=12),
                    customdata=provider_data[
                        ["prompt_price_per_m", "completion_price_per_m"]
                    ].values,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"{score_metric}: " + "%{y:.1f}<br>"
                        "Input: $%{customdata[0]:.2f}/M tokens<br>"
                        "Output: $%{customdata[1]:.2f}/M tokens<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Price per 1M Tokens — Input + Output (USD)",
                    font=dict(size=14),
                ),
                gridcolor="rgba(200,200,200,0.3)",
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title=dict(
                    text=f"Artificial Analysis {score_metric}",
                    font=dict(size=14),
                ),
                gridcolor="rgba(200,200,200,0.3)",
                tickfont=dict(size=12),
            ),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=12),
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(150,150,150,1)"),
            margin=dict(t=80, l=60, r=40, b=60),
        )

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        # Info note
        st.caption(
            "Data sources: Model pricing from [OpenRouter API](https://openrouter.ai/api/v1/models) · "
            "Intelligence Index from [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models)"
        )

    with tab2:
        # --- Bar chart: OpenRouter Pricing per 1M tokens (Input + Output) ---
        st.subheader("OpenRouter Model Pricing")
        st.caption("Price per 1M tokens (USD) — Input and Output from OpenRouter API")

        bar_df = plot_df.copy()
        bar_df["total_price"] = (
            bar_df["prompt_price_per_m"] + bar_df["completion_price_per_m"]
        )
        bar_df = bar_df.sort_values("total_price", ascending=False)

        bar_df["label"] = bar_df["model_name"]

        bar_fig = go.Figure()

        # Output price (bottom of stack)
        bar_fig.add_trace(
            go.Bar(
                x=bar_df["label"],
                y=bar_df["completion_price_per_m"],
                name="Output",
                marker_color="#1f77b4",
                hovertemplate="<b>%{x}</b><br>Output: $%{y:,.2f}/M tokens<extra></extra>",
            )
        )

        # Input price (top of stack)
        bar_fig.add_trace(
            go.Bar(
                x=bar_df["label"],
                y=bar_df["prompt_price_per_m"],
                name="Input",
                marker_color="#ff7f0e",
                hovertemplate="<b>%{x}</b><br>Input: $%{y:,.2f}/M tokens<extra></extra>",
            )
        )

        # Add total price labels on top of each bar
        bar_fig.add_trace(
            go.Scatter(
                x=bar_df["label"],
                y=bar_df["total_price"],
                mode="text",
                text=bar_df["total_price"].apply(lambda v: f"${v:,.2f}"),
                textposition="top center",
                textfont=dict(size=11),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        bar_fig.update_layout(
            barmode="stack",
            height=600,
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title=dict(text="Price per 1M Tokens (USD)", font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=12),
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(150,150,150,1)"),
            margin=dict(t=60, l=60, r=40, b=150),
        )

        st.plotly_chart(bar_fig, use_container_width=True, theme="streamlit")

    with st.expander("Price-to-Performance Ratio", expanded=True):
        perf_df = plot_df[
            [
                "model_name",
                "provider",
                score_col,
                "prompt_price_per_m",
                "completion_price_per_m",
                "cost_per_m",
            ]
        ].copy()
        perf_df["ratio"] = perf_df["cost_per_m"] / perf_df[score_col]
        perf_df = perf_df.sort_values("ratio", ascending=True).reset_index(drop=True)

        # Calculate gain/loss vs previous row (the better-value model above)
        prev_score = perf_df[score_col].shift(1)
        prev_cost = perf_df["cost_per_m"].shift(1)
        perf_df["next_score_gain"] = perf_df[score_col] - prev_score
        perf_df["next_cost_gain"] = perf_df["cost_per_m"] - prev_cost

        perf_df = perf_df.rename(
            columns={
                "model_name": "Model",
                "provider": "Provider",
                score_col: score_metric,
                "prompt_price_per_m": "Input $/M",
                "completion_price_per_m": "Output $/M",
                "cost_per_m": "Total $/M",
                "ratio": f"$ per {score_metric}",
                "next_score_gain": f"Next Δ {score_metric}",
                "next_cost_gain": "Next Δ $/M",
            }
        )

        # Format delta columns with arrows and colors
        def _fmt_delta(val, invert=False):
            """Format a delta value with colored arrow emoji. invert=True means lower is better."""
            if pd.isna(val):
                return "—"
            positive_is_good = not invert
            if val > 0:
                arrow = "🟢 ↑" if positive_is_good else "🔴 ↑"
            elif val < 0:
                arrow = "🔴 ↓" if positive_is_good else "🟢 ↓"
            else:
                return "0"
            return f"{arrow} {val:+.1f}" if not invert else f"{arrow} ${val:+.2f}"

        score_delta_col = f"Next Δ {score_metric}"
        cost_delta_col = "Next Δ $/M"
        perf_df[score_delta_col] = perf_df[score_delta_col].apply(
            lambda v: _fmt_delta(v, invert=False)
        )
        perf_df[cost_delta_col] = perf_df[cost_delta_col].apply(
            lambda v: _fmt_delta(v, invert=True)
        )

        st.dataframe(
            perf_df.style.format(
                {
                    score_metric: "{:.1f}",
                    "Input $/M": "${:.2f}",
                    "Output $/M": "${:.2f}",
                    "Total $/M": "${:.2f}",
                    f"$ per {score_metric}": "${:.3f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"**Next Δ {score_metric}**: score change if you switch to the next row's model · "
            "**Next Δ $/M**: cost change (positive = more expensive)"
        )

    # Show data table
    with st.expander("View raw data"):
        display_df = plot_df[
            [
                "model_name",
                "provider",
                score_col,
                "prompt_price_per_m",
                "completion_price_per_m",
                "cost_per_m",
            ]
        ].copy()
        display_df.columns = [
            "Model",
            "Provider",
            score_metric,
            "Input $/M tokens",
            "Output $/M tokens",
            "Total $/M tokens",
        ]
        display_df = display_df.sort_values(score_metric, ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("View model matching (AA → OpenRouter)"):
        match_display = matched_df[
            [
                "model_name",
                "openrouter_id",
                "provider",
                score_col,
                "prompt_price_per_m",
                "completion_price_per_m",
            ]
        ].copy()
        match_display.columns = [
            "AA Model Name",
            "OpenRouter ID",
            "Provider",
            score_metric,
            "Input $/M",
            "Output $/M",
        ]
        match_display = match_display.sort_values(score_metric, ascending=False)
        st.dataframe(match_display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
