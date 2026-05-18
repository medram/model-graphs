import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup

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
def fetch_artificial_analysis_intelligence():
    """Scrape intelligence index from Artificial Analysis leaderboard page."""
    url = "https://artificialanalysis.ai/leaderboards/models"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try to extract JSON data from script tags (Next.js pattern)
        scripts = soup.find_all("script")
        for script in scripts:
            if script.string and "intelligence" in script.string.lower():
                # Try to find JSON data
                pass

        # If page scraping doesn't work reliably, use curated data
        # from the leaderboard (as observed on the website)
        return None
    except Exception:
        return None


def get_intelligence_data():
    """
    Get intelligence index data.
    Since Artificial Analysis doesn't have a public API,
    we use curated data from their leaderboard.
    This data is updated based on their published rankings.
    """
    # Data sourced from artificialanalysis.ai/leaderboards/models
    # Format: (display_name, provider, intelligence_index, is_reasoning)
    data = [
        ("GPT-5.5 (xhigh)", "OpenAI", 60, True),
        ("GPT-5.5 (high)", "OpenAI", 59, True),
        ("Claude Opus 4.7 (max)", "Anthropic", 57, True),
        ("Gemini 3.1 Pro Preview", "Google", 57, True),
        ("GPT-5.5 (medium)", "OpenAI", 57, True),
        ("Kimi K2.6", "Kimi", 54, True),
        ("MiMo-V2.5-Pro", "Xiaomi", 54, True),
        ("GPT-5.3 Codex (xhigh)", "OpenAI", 54, True),
        ("Grok 4.3 (high)", "xAI", 53, True),
        ("Claude Opus 4.7 (Non-reasoning, high)", "Anthropic", 52, False),
        ("Qwen3.6 Max Preview", "Alibaba", 52, True),
        ("Claude Sonnet 4.6 (max)", "Anthropic", 52, True),
        ("DeepSeek V4 Pro (Max)", "DeepSeek", 52, True),
        ("GLM-5.1", "Z AI", 51, True),
        ("GPT-5.5 (low)", "OpenAI", 51, True),
        ("Qwen3.6 Plus", "Alibaba", 50, True),
        ("DeepSeek V4 Pro (High)", "DeepSeek", 50, True),
        ("GLM-5", "Z AI", 50, True),
        ("MiniMax-M2.7", "MiniMax", 50, True),
        ("MiMo-V2.5", "Xiaomi", 49, True),
        ("GPT-5.4 mini (xhigh)", "OpenAI", 49, True),
        ("GPT-5.4 (low)", "OpenAI", 48, True),
        ("DeepSeek V4 Flash (Max)", "DeepSeek", 47, True),
        ("Gemini 3 Flash", "Google", 46, True),
        ("DeepSeek V4 Flash (High)", "DeepSeek", 46, True),
        ("Qwen3.6 27B", "Alibaba", 46, True),
        ("Qwen3.5 397B A17B", "Alibaba", 45, True),
        ("MiMo-V2-Omni-0327", "Xiaomi", 45, True),
        ("Claude Sonnet 4.6 (Non-reasoning)", "Anthropic", 44, False),
        ("GPT-5.4 nano (xhigh)", "OpenAI", 44, True),
        ("Qwen3.6 35B A3B", "Alibaba", 43, True),
        ("Kimi K2.6 (non-reasoning)", "Kimi", 43, False),
        ("Hy3-preview", "Tencent", 42, True),
        ("Qwen3.5 122B A10B", "Alibaba", 42, True),
        ("MiMo-V2-Flash (Feb 2026)", "Xiaomi", 41, True),
        ("MiniMax-M2.5", "MiniMax", 41, True),
        ("Qwen3.5 27B", "Alibaba", 41, True),
        ("DeepSeek V4 Pro", "DeepSeek", 39, False),
        ("Mistral Medium 3.5", "Mistral", 39, True),
        ("Gemma 4 31B", "Google", 39, True),
        ("o3", "OpenAI", 38, True),
        ("GPT-5.4 nano", "OpenAI", 38, False),
        ("GPT-5.4 mini (medium)", "OpenAI", 38, True),
        ("Kimi K2.5", "Kimi", 37, True),
        ("Claude 4.5 Haiku", "Anthropic", 37, True),
        ("DeepSeek V4 Flash", "DeepSeek", 36, False),
        ("NVIDIA Nemotron 3 Super", "NVIDIA", 36, True),
        ("Nova 2.0 Pro Preview (medium)", "Amazon", 36, True),
        ("Gemini 2.5 Pro", "Google", 35, True),
        ("Gemini 3 Flash (non-reasoning)", "Google", 35, False),
        ("gpt-oss-120B (high)", "OpenAI", 33, True),
        ("Mercury 2", "Inception", 33, True),
        ("Qwen3.5 9B", "Alibaba", 32, True),
        ("Gemma 4 31B (non-reasoning)", "Google", 32, False),
        ("Gemma 4 26B A4B", "Google", 31, True),
        ("Claude 4.5 Haiku (non-reasoning)", "Anthropic", 31, False),
        ("Grok 4.3", "xAI", 31, False),
        ("MiMo-V2-Flash", "Xiaomi", 30, True),
        ("gpt-oss-20B (high)", "OpenAI", 24, True),
        ("gpt-oss-120B (low)", "OpenAI", 24, False),
        ("NVIDIA Nemotron 3 Nano", "NVIDIA", 24, True),
    ]

    return pd.DataFrame(
        data, columns=["model_name", "provider", "intelligence_index", "is_reasoning"]
    )


def match_models(openrouter_df, intelligence_df):
    """Match OpenRouter models with Artificial Analysis intelligence data."""
    # Mapping from Artificial Analysis names to OpenRouter IDs
    name_to_id = {
        "GPT-5.5 (xhigh)": "openai/gpt-5.5",
        "GPT-5.5 (high)": "openai/gpt-5.5",
        "GPT-5.5 (medium)": "openai/gpt-5.5",
        "GPT-5.5 (low)": "openai/gpt-5.5",
        "GPT-5.5 (Non-reasoning)": "openai/gpt-5.5",
        "Claude Opus 4.7 (max)": "anthropic/claude-opus-4.7",
        "Claude Opus 4.7 (Non-reasoning, high)": "anthropic/claude-opus-4.7",
        "Gemini 3.1 Pro Preview": "google/gemini-3.1-pro-preview",
        "Kimi K2.6": "moonshotai/kimi-k2.6",
        "Kimi K2.6 (non-reasoning)": "moonshotai/kimi-k2.6",
        "MiMo-V2.5-Pro": "xiaomi/mimo-v2-pro",
        "GPT-5.3 Codex (xhigh)": "openai/gpt-5.3-codex",
        "Grok 4.3 (high)": "x-ai/grok-4.3",
        "Grok 4.3": "x-ai/grok-4.3",
        "Qwen3.6 Max Preview": "qwen/qwen3.6-max-preview",
        "Claude Sonnet 4.6 (max)": "anthropic/claude-sonnet-4.6",
        "Claude Sonnet 4.6 (Non-reasoning)": "anthropic/claude-sonnet-4.6",
        "Claude Sonnet 4.6 (Non-reasoning, Low Effort)": "anthropic/claude-sonnet-4.6",
        "DeepSeek V4 Pro (Max)": "deepseek/deepseek-v4-pro",
        "DeepSeek V4 Pro (High)": "deepseek/deepseek-v4-pro",
        "DeepSeek V4 Pro": "deepseek/deepseek-v4-pro",
        "GLM-5.1": "z-ai/glm-5",
        "GLM-5": "z-ai/glm-5",
        "GPT-5.5 Pro (xhigh)": "openai/gpt-5.5-pro",
        "Qwen3.6 Plus": "qwen/qwen3.5-plus",
        "MiniMax-M2.7": "minimax/minimax-m2.7",
        "MiniMax-M2.5": "minimax/minimax-m2.5",
        "MiMo-V2.5": "xiaomi/mimo-v2.5-pro",
        "GPT-5.4 mini (xhigh)": "openai/gpt-5.4-mini",
        "GPT-5.4 mini (medium)": "openai/gpt-5.4-mini",
        "GPT-5.4 (low)": "openai/gpt-5.4",
        "GPT-5.4 nano (xhigh)": "openai/gpt-5.4-nano",
        "GPT-5.4 nano": "openai/gpt-5.4-nano",
        "DeepSeek V4 Flash (Max)": "deepseek/deepseek-v4-flash",
        "DeepSeek V4 Flash (High)": "deepseek/deepseek-v4-flash",
        "DeepSeek V4 Flash": "deepseek/deepseek-v4-flash",
        "Gemini 3 Flash": "google/gemini-3-flash-preview",
        "Gemini 3 Flash (non-reasoning)": "google/gemini-3-flash-preview",
        "Qwen3.6 27B": "qwen/qwen3.6-27b",
        "Qwen3.5 397B A17B": "qwen/qwen3.5-397b-a17b",
        "MiMo-V2-Omni-0327": "xiaomi/mimo-v2-omni",
        "Qwen3.6 35B A3B": "qwen/qwen3.6-35b-a3b",
        "Hy3-preview": "tencent/hy3-preview",
        "Qwen3.5 122B A10B": "qwen/qwen3.5-122b-a10b",
        "MiMo-V2-Flash": "xiaomi/mimo-v2-flash",
        "MiMo-V2-Flash (Feb 2026)": "xiaomi/mimo-v2-flash",
        "Qwen3.5 27B": "qwen/qwen3.5-27b",
        "Mistral Medium 3.5": "mistralai/mistral-medium-3.1",
        "Gemma 4 31B": "google/gemma-4-31b",
        "Gemma 4 31B (non-reasoning)": "google/gemma-4-31b",
        "o3": "openai/o3",
        "Claude 4.5 Haiku": "anthropic/claude-3.5-haiku",
        "Claude 4.5 Haiku (non-reasoning)": "anthropic/claude-3.5-haiku",
        "Kimi K2.5": "moonshotai/kimi-k2.5-0127",
        "NVIDIA Nemotron 3 Super": "nvidia/nemotron-3-super-120b-a12b",
        "Gemini 2.5 Pro": "google/gemini-2.5-pro",
        "Nova 2.0 Pro Preview (medium)": "amazon/nova-2-pro-v1",
        "gpt-oss-120B (high)": "openai/gpt-oss-120b",
        "gpt-oss-120B (low)": "openai/gpt-oss-120b",
        "gpt-oss-20B (high)": "openai/gpt-oss-20b",
        "Mercury 2": "inception/mercury-2",
        "Qwen3.5 9B": "qwen/qwen3.5-9b",
        "Gemma 4 26B A4B": "google/gemma-4-26b-a4b",
        "NVIDIA Nemotron 3 Nano": "nvidia/nemotron-nano-9b-v2",
    }

    matched = []
    for _, row in intelligence_df.iterrows():
        model_name = row["model_name"]
        openrouter_id = name_to_id.get(model_name)

        if openrouter_id:
            or_match = openrouter_df[openrouter_df["id"] == openrouter_id]
            if not or_match.empty:
                or_row = or_match.iloc[0]
                matched.append(
                    {
                        "model_name": model_name,
                        "provider": row["provider"],
                        "intelligence_index": row["intelligence_index"],
                        "is_reasoning": row["is_reasoning"],
                        "blended_price_per_m": or_row["blended_price_per_m"],
                        "prompt_price_per_m": or_row["prompt_price_per_m"],
                        "completion_price_per_m": or_row["completion_price_per_m"],
                        "openrouter_id": openrouter_id,
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
    "OpenAI": "#1A1A1A",
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
    with st.spinner("Fetching model data from OpenRouter..."):
        openrouter_df = fetch_openrouter_models()

    intelligence_df = get_intelligence_data()

    # Match models
    matched_df = match_models(openrouter_df, intelligence_df)

    if matched_df.empty:
        st.error("No models could be matched between data sources.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    providers = sorted(matched_df["provider"].unique())
    selected_providers = st.sidebar.multiselect(
        "Providers",
        providers,
        default=providers,
    )

    show_reasoning = st.sidebar.checkbox("Show reasoning models", value=True)
    show_non_reasoning = st.sidebar.checkbox("Show non-reasoning models", value=True)

    # Filter
    filtered_df = matched_df[matched_df["provider"].isin(selected_providers)]
    if not show_reasoning:
        filtered_df = filtered_df[~filtered_df["is_reasoning"]]
    if not show_non_reasoning:
        filtered_df = filtered_df[filtered_df["is_reasoning"]]

    # Model selection — default to top 30 by intelligence index
    st.sidebar.header("Model Selection")
    top_30 = (
        filtered_df.sort_values("intelligence_index", ascending=False)
        .head(30)["model_name"]
        .tolist()
    )
    all_models = sorted(filtered_df["model_name"].tolist())
    selected_models = st.sidebar.multiselect(
        "Select models to display",
        all_models,
        default=sorted(top_30),
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
        y_mid = plot_df["intelligence_index"].median()
        y_max = plot_df["intelligence_index"].max() + 5

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

            # Reasoning models get a different marker
            reasoning = provider_data[provider_data["is_reasoning"]]
            non_reasoning = provider_data[~provider_data["is_reasoning"]]

            if not reasoning.empty:
                fig.add_trace(
                    go.Scatter(
                        x=reasoning["cost_per_m"],
                        y=reasoning["intelligence_index"],
                        mode="markers+text",
                        name=f"{provider}",
                        text=reasoning["model_name"] + "💡",
                        textposition="top right",
                        textfont=dict(size=11),
                        marker=dict(color=color, size=12),
                        customdata=reasoning[
                            ["prompt_price_per_m", "completion_price_per_m"]
                        ].values,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Intelligence: %{y}<br>"
                            "Input: $%{customdata[0]:.2f}/M tokens<br>"
                            "Output: $%{customdata[1]:.2f}/M tokens<br>"
                            "<extra></extra>"
                        ),
                    )
                )

            if not non_reasoning.empty:
                fig.add_trace(
                    go.Scatter(
                        x=non_reasoning["cost_per_m"],
                        y=non_reasoning["intelligence_index"],
                        mode="markers+text",
                        name=f"{provider} (non-reasoning)",
                        text=non_reasoning["model_name"],
                        textposition="top right",
                        textfont=dict(size=11),
                        marker=dict(color=color, size=12, symbol="circle"),
                        customdata=non_reasoning[
                            ["prompt_price_per_m", "completion_price_per_m"]
                        ].values,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Intelligence: %{y}<br>"
                            "Input: $%{customdata[0]:.2f}/M tokens<br>"
                            "Output: $%{customdata[1]:.2f}/M tokens<br>"
                            "<extra></extra>"
                        ),
                    )
                )

        # Set axis range based on actual data
        x_data_min = plot_df["cost_per_m"].min()
        x_data_max = plot_df["cost_per_m"].max()
        import math

        x_log_min = max(0, math.floor(math.log10(x_data_min * 0.5)))
        x_log_max = math.ceil(math.log10(x_data_max * 2))

        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Price per 1M Tokens — Input + Output (USD, Log Scale)",
                    font=dict(size=14),
                ),
                type="log",
                range=[x_log_min, x_log_max],
                gridcolor="rgba(200,200,200,0.3)",
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title=dict(
                    text="Artificial Analysis Intelligence Index", font=dict(size=14)
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
            plot_bgcolor="white",
            margin=dict(t=80, l=60, r=40, b=60),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Info note
        st.caption("💡 Reasoning models are indicated by a lightbulb icon.")
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

        # Build display labels with lightbulb for reasoning models
        bar_df["label"] = bar_df.apply(
            lambda r: f"{r['model_name']} 💡" if r["is_reasoning"] else r["model_name"],
            axis=1,
        )

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

        # Reasoning indicator (zero-height trace for legend)
        bar_fig.add_trace(
            go.Bar(
                x=bar_df["label"],
                y=[0] * len(bar_df),
                name="Reasoning",
                marker_color="#2ca02c",
                hoverinfo="skip",
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
            plot_bgcolor="white",
            margin=dict(t=60, l=60, r=40, b=150),
        )

        st.plotly_chart(bar_fig, use_container_width=True)
        st.caption("💡 Reasoning models are indicated by a lightbulb icon.")

    # Show data table
    with st.expander("View raw data"):
        display_df = plot_df[
            [
                "model_name",
                "provider",
                "intelligence_index",
                "is_reasoning",
                "prompt_price_per_m",
                "completion_price_per_m",
                "cost_per_m",
            ]
        ].copy()
        display_df.columns = [
            "Model",
            "Provider",
            "Intelligence Index",
            "Reasoning",
            "Input $/M tokens",
            "Output $/M tokens",
            "Total $/M tokens",
        ]
        display_df = display_df.sort_values("Intelligence Index", ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
