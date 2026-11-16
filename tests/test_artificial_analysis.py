import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st


try:
    importlib.import_module("plotly.graph_objects")
except ModuleNotFoundError:
    plotly = types.ModuleType("plotly")
    plotly.__path__ = []
    graph_objects = types.ModuleType("plotly.graph_objects")
    setattr(plotly, "graph_objects", graph_objects)
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = plotly.graph_objects


with (
    patch.object(st, "set_page_config"),
    patch.object(st, "title"),
    patch.object(st, "caption"),
    patch.object(st, "cache_data", side_effect=lambda **kwargs: lambda function: function),
):
    app = importlib.import_module("app")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class ArtificialAnalysisApiTest(unittest.TestCase):

    def test_fetches_all_free_api_pages_and_matches_models_by_creator_name(self):
        first_page = FakeResponse(
            {
                "tier": "free",
                "intelligence_index_version": 5,
                "pagination": {"page": 1, "page_size": 200, "total_pages": 2, "has_more": True},
                "data": [
                    {
                        "id": "aa-openai-id",
                        "name": "GPT-5.5",
                        "slug": "gpt-5-5",
                        "model_creator": {"id": "creator-openai", "name": "OpenAI"},
                        "evaluations": {
                            "artificial_analysis_intelligence_index": 70.1,
                            "artificial_analysis_coding_index": None,
                            "artificial_analysis_engineering_index": 55.2,
                        },
                    }
                ],
            }
        )
        second_page = FakeResponse(
            {
                "tier": "free",
                "intelligence_index_version": 5,
                "pagination": {"page": 2, "page_size": 200, "total_pages": 2, "has_more": False},
                "data": [
                    {
                        "id": "aa-google-id",
                        "name": "Gemini 3.0",
                        "slug": "gemini-3-0",
                        "model_creator": {"id": "creator-google", "name": "Google"},
                        "evaluations": {
                            "artificial_analysis_intelligence_index": 68.4,
                            "artificial_analysis_coding_index": None,
                            "artificial_analysis_engineering_index": None,
                        },
                    },
                    {
                        "id": "aa-unscored-id",
                        "name": "Unscored Model",
                        "slug": "unscored-model",
                        "model_creator": {"id": "creator-google", "name": "Google"},
                        "evaluations": {
                            "artificial_analysis_intelligence_index": None,
                            "artificial_analysis_coding_index": None,
                        },
                    },
                ],
            }
        )
        openrouter_df = pd.DataFrame(
            [
                {
                    "id": "openai/gpt-5-5",
                    "blended_price_per_m": 10.0,
                    "prompt_price_per_m": 5.0,
                    "completion_price_per_m": 20.0,
                },
                {
                    "id": "google/gemini-3-0",
                    "blended_price_per_m": 8.0,
                    "prompt_price_per_m": 4.0,
                    "completion_price_per_m": 16.0,
                },
            ]
        )

        with patch.dict(os.environ, {"AA_API_KEY": "test-key"}), patch.object(
            app.requests, "get", side_effect=[first_page, second_page]
        ) as get:
            aa_df = app.fetch_artificial_analysis_models()
            matched_df = app.match_models(openrouter_df, aa_df)

        self.assertEqual(2, len(matched_df))
        self.assertNotIn("Unscored Model", aa_df["aa_name"].tolist())
        self.assertEqual(
            [70.1, 68.4],
            matched_df["intelligence_index"].tolist(),
        )
        self.assertEqual(
            ["intelligence_index", "engineering_index"],
            [
                column
                for column in matched_df.columns
                if column.endswith("_index") and matched_df[column].notna().any()
            ],
        )
        self.assertEqual(
            ["openai/gpt-5-5", "google/gemini-3-0"],
            matched_df["openrouter_id"].tolist(),
        )
        self.assertEqual(
            "https://artificialanalysis.ai/api/v2/language/models/free",
            get.call_args_list[0].args[0],
        )
        self.assertEqual(
            [{"page": 1}, {"page": 2}],
            [call.kwargs["params"] for call in get.call_args_list],
        )
        self.assertEqual(
            [{"x-api-key": "test-key"}, {"x-api-key": "test-key"}],
            [call.kwargs["headers"] for call in get.call_args_list],
        )


if __name__ == "__main__":
    unittest.main()
