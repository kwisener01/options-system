from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pytz
import logging

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER

logger = logging.getLogger(__name__)


class AlpacaClient:
    def __init__(self):
        self.trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=IS_PAPER)
        self.data = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        logger.info("AlpacaClient initialized — paper=%s", IS_PAPER)

    # ---------- Account ----------

    def get_account(self):
        return self.trading.get_account()

    def get_buying_power(self) -> float:
        acct = self.get_account()
        return float(acct.buying_power)

    def get_portfolio_value(self) -> float:
        acct = self.get_account()
        return float(acct.portfolio_value)

    def get_positions(self) -> list:
        return self.trading.get_all_positions()

    def get_position(self, symbol: str):
        try:
            return self.trading.get_open_position(symbol)
        except Exception:
            return None

    # ---------- Orders ----------

    def buy(self, symbol: str, dollar_amount: float) -> dict:
        """Place a notional (dollar-based) market buy order."""
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(dollar_amount, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading.submit_order(req)
        logger.info("BUY %s $%.2f — order_id=%s", symbol, dollar_amount, order.id)
        return order

    def sell(self, symbol: str, qty: float = None, dollar_amount: float = None) -> dict:
        """Close a position by qty or dollar amount. If neither provided, closes full position."""
        if qty is None and dollar_amount is None:
            order = self.trading.close_position(symbol)
        else:
            side_kwargs = {"qty": str(qty)} if qty else {"notional": round(dollar_amount, 2)}
            req = MarketOrderRequest(
                symbol=symbol,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                **side_kwargs,
            )
            order = self.trading.submit_order(req)
        logger.info("SELL %s — order_id=%s", symbol, order.id)
        return order

    def close_all_positions(self):
        self.trading.close_all_positions(cancel_orders=True)
        logger.warning("All positions closed.")

    def get_open_orders(self) -> list:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return self.trading.get_orders(req)

    # ---------- Market Data ----------

    def get_latest_quote(self, symbol: str) -> dict:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self.data.get_stock_latest_quote(req)
        q = quotes[symbol]
        return {"bid": float(q.bid_price), "ask": float(q.ask_price), "mid": (float(q.bid_price) + float(q.ask_price)) / 2}

    def get_bars(self, symbol: str, days: int = 60, timeframe: TimeFrame = TimeFrame.Day) -> list:
        end = datetime.now(pytz.UTC)
        start = end - timedelta(days=days)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
        bars = self.data.get_stock_bars(req)
        return bars[symbol] if symbol in bars else []

    # ---------- Market Status ----------

    def is_market_open(self) -> bool:
        clock = self.trading.get_clock()
        return clock.is_open
