import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as pl
import pandas as pd
import seaborn as sb
from fastquant import (
    get_pse_data_cache,
    get_stock_data,
    datestring_to_datetime,
)
from fastquant import Network


class Screener(Network):
    def __init__(
        self,
        symbol=None,
        sector=None,
        start_date="2020-01-01",
        end_date=None,
        data_type="close",
        indicator="pct_change",
        exclude_symbols=None,
        verbose=True,
        clobber=False,
        update_cache=False,
    ):
        super().__init__(
            symbol=None,
            sector=sector,
            start_date=start_date,
            end_date=None,
            exclude_symbols=None,
            indicator=data_type,
            verbose=verbose,
            clobber=clobber,
            update_cache=update_cache,
        )
        """
        """
        self.symbol = symbol
        self.data_type = data_type
        self.indicator = indicator
        self.cache = get_pse_data_cache()
        self.data = self.get_data(data_type)
        self.data = self.filter_date()

        first_date_entry = self.data.index[0].strftime("%Y-%m-%d")
        last_date_entry = self.data.index[-1].strftime("%Y-%m-%d")
        self.start_date = (
            first_date_entry if start_date is None else start_date
        )
        self.end_date = last_date_entry if end_date is None else end_date

    def get_data(self, data_type=None):
        if data_type is None:
            return self.cache.xs("close", level=1, axis=1)
        else:
            return self.cache.xs(data_type, level=1, axis=1)

    def filter_date(self, start_date=None, end_date=None):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date

        data = self.data.copy()
        data_filter = (data.index >= start_date) & (data.index <= end_date)
        data_recent = data[data_filter].dropna(how="all", axis=1)
        return data_recent

    def get_technical_indicator_data(
        self,
        indicator=None,
        sector=None,
        start_date=None,
        end_date=None,
        daily=False,
    ):
        indicator = self.indicator if indicator is None else indicator
        data_recent = self.filter_date(
            start_date=start_date, end_date=end_date
        )
        if sector is not None:
            symbols = self.get_symbols_of_a_sector(sector)
            data_recent = data_recent.loc[:, symbols]

        if indicator == self.data_type:
            # return orig
            return data_recent

        elif indicator == "pct_change":
            daily_pct_change = data_recent.pct_change().apply(
                lambda x: x * 100
            )
            if daily:
                return daily_pct_change
            else:
                pct_change = daily_pct_change.mean(axis=0).sort_values(
                    ascending=False
                )
                pct_change.name = "pct_change"
                return pct_change

        elif indicator == "sharpe_ratio":
            errmsg = "set daily=False"
            assert daily is False, errmsg
            # sharpe ratio, volatility?
            sharpe_ratio = (
                data_recent.pct_change().mean()
                / data_recent.pct_change().std()
            )
            sharpe_ratio.name = "sharpe_ratio"
            return sharpe_ratio
        else:
            raise ValueError("{} not found!".format(indicator))

    def plot_subsector(
        self,
        subsector,
        kind="line",
        start_date=None,
        end_date=None,
        dt_format="%b %d",
        ax=None,
        figsize=(8, 8),
    ):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date

        if ax is None:
            fig, ax = pl.subplots(figsize=figsize)

        symbols = self.get_symbols_of_a_sector(subsector, subsector=True)
        data = self.filter_date(start_date, end_date)[symbols]
        pc = data.pct_change().apply(lambda x: x * 100)

        if kind == "line":
            ax = pc.plot(ax=ax, marker="o")
            pl.setp(
                ax,
                xlabel="",
                ylabel="Price change (%)",
                xlim=(start_date, end_date),
            )
            ax.legend()
            ax.set_title(f"{subsector} subsector")
            # set ticks every week
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            # set major ticks format
            ax.xaxis.set_major_formatter(mdates.DateFormatter(dt_format))
        elif (kind == "density") | (kind == "kde"):
            ax = pc.plot(ax=ax, kind="density", subplots=False)
            ax.legend(title=f"{subsector} subsector")

            for n, m in enumerate(pc.mean()):
                c = ax.get_lines()[n].get_color()
                ax.axvline(m, 0, 1, ls="--", lw=2, c=c)

            # ax.set_xlim(-10,10)
            ax.set_title(f"Data since {start_date}")
            ax.set_xlabel("Price change (%)")

        else:
            raise ValueError(f"{kind} not found!")
        return ax

    def plot_sectors(
        self,
        sector,
        per_symbol=False,
        indicator=None,
        start_date=None,
        end_date=None,
        daily=True,
        figsize=(10, 8),
        ax=None,
    ):
        indicator = self.indicator if indicator is None else indicator

        if indicator == "pct_change":
            indicator_label = "CLOSE CHANGE (%)"
        elif indicator == "close":
            indicator_label = "CLOSE"
        elif indicator == "sharpe_ratio":
            indicator_label = "Sharpe ratio"
        else:
            raise ValueError("indicator={} not found!".format(indicator))

        data = self.get_technical_indicator_data(
            indicator=indicator,
            start_date=start_date,
            end_date=end_date,
            daily=daily,
        )
        if ax is None:
            fig, ax = pl.subplots(1, 1, figsize=figsize)

        if (sector == "all") | (isinstance(sector, list)):
            sectors = self.all_sectors if sector == "all" else sector
            for sector in sectors:
                try:
                    symbols = self.get_symbols_of_a_sector(sector)
                    if per_symbol:
                        d = data.loc[:, symbols]
                    else:
                        d = data.loc[:, symbols].mean(axis=1)
                    _ = d.plot(ax=ax, marker="o", label=sector)
                    ax.legend()
                except Exception as e:
                    print(e)
        else:
            symbols = self.get_symbols_of_a_sector(sector)
            if per_symbol:
                d = data.loc[:, symbols]
            else:
                d = data.loc[:, symbols].mean(axis=1)
            _ = d.plot(ax=ax, marker="o", label=sector)
        ax.set_ylabel("{}".format(indicator_label))
        return ax

    def get_indicator_per_sector(
        self, indicator=None, sector=None, start_date=None, end_date=None
    ):
        indicator = self.indicator if indicator is None else indicator

        df = []
        for sector in self.all_sectors:
            d = self.get_technical_indicator_data(
                daily=True,
                start_date=start_date,
                indicator=indicator,
                sector=sector,
            ).mean()
            d.name = indicator
            d = d.reset_index()
            d["sector"] = sector
            df.append(d)

        return pd.concat(df)

    def plot_sectors_boxplot(
        self,
        sector=None,
        indicator=None,
        start_date=None,
        end_date=None,
        figsize=(10, 8),
        ax=None,
    ):
        indicator = self.indicator if indicator is None else indicator
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        sector = self.sector if sector is None else sector.lower()
        # add sector column
        df = self.get_indicator_per_sector(
            indicator=indicator,
            sector=sector,
            start_date=start_date,
            end_date=end_date,
        )
        if ax is None:
            fig, ax = pl.subplots(figsize=figsize)

        if sector is None:
            d = df.copy()
        else:
            symbols = self.get_symbols_of_a_sector(sector)
            d = df[df.Symbol.isin(symbols)]

        _ = sb.boxplot(
            ax=ax, x="sector", y="pct_change", data=d, palette="vlag"
        )
        ax.set_title(
            "Mean {} ({} - {})".format(indicator, start_date, end_date)
        )
        return ax

    def plot_sectors_kde(
        self,
        indicator=None,
        start_date=None,
        end_date=None,
        figsize=(10, 8),
        ax=None,
    ):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        indicator = self.indicator if indicator is None else indicator

        fig, ax = pl.subplots(1, 1, figsize=(10, 6))

        for sector in self.all_sectors:
            try:
                symbols = self.get_symbols_of_a_sector(sector=sector)
                data = self.filter_date()[symbols]
                daily_pct_change = (
                    data.pct_change().apply(lambda x: x * 100).mean(axis=1)
                )

                daily_pct_change.plot.kde(ax=ax, label=sector)
                ax.legend()
                ax.set_xlim(-10, 10)
            except Exception as e:
                print(e)
        ax.set_title(
            "Mean {} ({} - {})".format(indicator, start_date, end_date)
        )
        ax.set_xlabel("Percent change")
        return ax
