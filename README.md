# Model Graphs

Interactive dashboard comparing AI model intelligence and coding performance against pricing, using data from [Artificial Analysis](https://artificialanalysis.ai/) and [OpenRouter](https://openrouter.ai/).

![Intelligence vs. Cost screenshot](screenshot.png)

**Live demo:** [model-graphs.streamlit.app](https://model-graphs.streamlit.app/)

## Features

- **Intelligence vs. Cost** — Scatter plot of model scores against pricing per 1M tokens
- **Cost to Run** — Stacked bar chart of input/output pricing across models
- **Price-to-Performance Ratio** — Table sorted by best value, with delta indicators showing marginal gains between models
- Filter by provider, score metric (Intelligence Index / Coding Index), and number of top models
- Configurable top-N model selector (default: 30)

## Setup

```bash
# Clone and install
git clone https://github.com/medram/model-graphs.git
cd model-graphs
make install
```

Create a `.env` file with your API key:

```
AA_API_KEY=your_artificial_analysis_api_key
```

## Usage

```bash
# Run the app
make run

# Run with auto-reload on save
make dev
```

## Data Sources

- **Model pricing**: [OpenRouter API](https://openrouter.ai/api/v1/models)
- **Intelligence & Coding Index**: [Artificial Analysis API](https://artificialanalysis.ai/leaderboards/models)
