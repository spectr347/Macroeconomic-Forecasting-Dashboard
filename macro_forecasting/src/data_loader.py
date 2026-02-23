import requests
import pandas as pd


def load_world_bank_data(country, indicator):
    url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=1000"

    try:
        response = requests.get(url)
        data = response.json()

        if len(data) < 2:
            return None

        records = data[1]
        df = pd.DataFrame(records)[["date", "value"]]
        df = df.dropna()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df.set_index("date", inplace=True)

        # Ensure annual frequency
        df = df.asfreq("YS")

        return df

    except:
        return None