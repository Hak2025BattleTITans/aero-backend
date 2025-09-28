import pandas as pd
import plotly.express as px


class Plotter:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, sep=";")

        for col in ["Доход пасс", "Пассажиры"]:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df["Дата вылета"] = pd.to_datetime(
            self.df["Дата вылета"], errors="coerce"
        )

    def avg_check(self):

        #   Средний чек = Доход пассажироов / Пассажиры
        #   Круговая диаграмма
        #   Агрегация по коду кабины

        df = self.df.copy()
        df = df[df["Пассажиры"] > 0]
        df["Средний чек"] = df["Доход пасс"] / df["Пассажиры"]

        avg_df = df.groupby("Код кабины", as_index=False)["Средний чек"].mean()

        fig = px.pie(
            avg_df,
            names="Код кабины",
            values="Средний чек",
            title="Средний чек по коду кабины"
        )

        spec = fig.to_dict()
        return spec

    def dyn_income(self):

        #   Динамика дохода по коду кабины по дням
        #   Линейный график

        df = self.df.copy()

        daily_revenue = df.groupby(['Дата вылета', 'Код кабины'])['Доход пасс'].sum().reset_index()

        fig = px.line(
            daily_revenue,
            x="Дата вылета",
            y="Доход пасс",
            color="Код кабины",
            facet_row="Код кабины",
            title="Динамика дохода по коду кабины (сумма за день)"
        )

        fig.update_yaxes(showticklabels=True)
        fig.update_xaxes(showticklabels=True)


        fig.update_layout(
            height=800,
            hovermode="x unified"
        )

        spec = fig.to_dict()
        return spec

    def dyn_passenger(self):
        #   Динамика пассажиропотока по коду кабины по дням
        #   Линейный график

        df = self.df.copy()

        daily_passengers = df.groupby(['Дата вылета', 'Код кабины'])['Пассажиры'].sum().reset_index()

        fig = px.line(
            daily_passengers,
            x="Дата вылета",
            y="Пассажиры",
            color="Код кабины",
            facet_row="Код кабины",
            title="Динамика пассажиропотока по коду кабины (сумма за день)"
        )

        fig.update_yaxes(showticklabels=True)
        fig.update_xaxes(showticklabels=True)

        fig.update_layout(
            height=800,
            hovermode="x unified"
        )

        spec = fig.to_dict()
        return spec
