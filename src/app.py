import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from algorithms import (
    get_feature_relevance,
    get_feature_relevance_tree,
    get_decision_tree,
    get_correlation,
)


class App:
    def __init__(self):
        self.sections = {
            "Feature Analysis": self.feature_relevance,
            "Timeseries Analysis": self.timeseries_analysis,
        }

    def sample_datasets(self):
        return {
            "Cars (classification)": "/src/data/car.data",
            "Monthly beer (timeseries)": "/src/data/monthly-beer-production-in-austria.csv",
            "Electric production (timeseries)": "/src/data/Electric_Production.csv",
        }

    def run(self):
        st.write("## Input data")

        section = st.sidebar.selectbox("Section", list(self.sections))
        dataset = st.file_uploader("Dataset (CSV)")

        if not dataset:
            st.info(
                "You can load a dataset in CSV format to activate the possible analysis or select a sample dataset from below."
            )
            dataset_name = st.selectbox(
                "Select sample dataset", list(self.sample_datasets())
            )
            dataset = Path(self.sample_datasets()[dataset_name])

        dataset = pd.read_csv(dataset)

        st.write("### Raw data")
        if st.checkbox("Show raw data"):
            st.write(dataset)

        self.sections[section](dataset)

    def feature_relevance(self, dataset: pd.DataFrame):
        st.write("## Feature relevance analysis")

        columns = list(dataset.columns)
        predict_column = st.selectbox("Prediction column", columns, len(columns) - 1)

        y = dataset[predict_column]
        X = list(dataset.drop(columns=predict_column).to_dict("index").values())

        st.write("#### Numerical feature relevance")

        features = pd.DataFrame(
            [
                dict(feature=k, relevance=v, target=c)
                for c, weights in get_feature_relevance(X, y)
                for k, v in weights.items()
            ]
        )
        st.write(features)

        features["sign"] = (features["relevance"] > 0).astype(bool)
        features["abs"] = features["relevance"].abs()

        st.write("#### Graphical representation")

        st.write(
            alt.Chart(features)
            .mark_circle()
            .encode(x="feature", y="target", color="sign", size="abs",)
        )

        st.write("#### Decision Tree")

        depth = st.number_input("Max tree depth", 1, 10, 3)
        st.graphviz_chart(get_decision_tree(X, y, depth=depth))

        st.write("#### Features correlation")

        X = list(dataset.to_dict("index").values())
        corr = get_correlation(X)

        if st.checkbox("Show correlation table"):
            st.write(corr)

        st.write(
            alt.Chart(corr.reset_index().melt(id_vars="index"))
            .mark_rect()
            .encode(x="index", y="variable", color="value")
        )

    def timeseries_analysis(self, dataset: pd.DataFrame):
        st.write("## Timeseries Analysis")

        time_index = st.selectbox("Select the timeseries index", list(dataset.columns))
        value_index = st.selectbox(
            "Select the value column", list(set(dataset.columns) - set([time_index]))
        )

        start, end = st.slider("Range to visualize", 0, len(dataset), (0, len(dataset)))
        st.write(
            f"Selected `{end - start}` values in the range `{dataset[time_index][start]}` - `{dataset[time_index][start]}`"
        )

        st.altair_chart(
            alt.Chart(dataset[start:end])
            .mark_line()
            .encode(x=time_index, y=value_index,)
            .interactive(),
            use_container_width=True,
        )


App().run()
