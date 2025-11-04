from ibapi.client import *
from ibapi.wrapper import *
from ibapi.contract import Contract
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from ibapi.execution import Execution

port = 7497
MARGIN = True

contracts = {
    "SPY": {"symbol": "SPY", "secType": "STK", "currency": "USD", "exchange": "SMART", "primaryExchange": "ARCA"},
    "VWO": {"symbol": "VWO", "secType": "STK", "currency": "USD", "exchange": "SMART", "primaryExchange": "ARCA"},
    "EXSA": {"symbol": "EXSA", "secType": "STK", "currency": "EUR", "exchange": "SMART", "primaryExchange": "IBIS"}
}

class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.symbol_map = {}
        self.df_final = pd.DataFrame()
        self.completed_symbols = set()
        self.orderId = None
        self.account_values = {}
        self.capital = 0
        self.vwoCurrentPos = 0
        self.spyCurrentPos = 0
        self.exsaCurrentPos = 0
        self.actualPositions = [0,0,0]
        self.nextOrderId = None
        self.parentorderId = None
        self.y2 = None
        self.max_leverage_ibkr = 0
        self.buyingpower = 0
        self.NetLiquidation = 0

    def nextValidId(self, orderId):
        self.orderId = orderId
        self.start()

        spyContract = Contract()
        spyContract.symbol = contracts["SPY"]['symbol']
        spyContract.secType = contracts["SPY"]["secType"]
        spyContract.exchange = contracts["SPY"]["exchange"]
        spyContract.currency = contracts["SPY"]["currency"]

        vwoContract = Contract()
        vwoContract.symbol = contracts["VWO"]['symbol']
        vwoContract.secType = contracts["VWO"]["secType"]
        vwoContract.exchange = contracts["VWO"]["exchange"]
        vwoContract.currency = contracts["VWO"]["currency"]

        exsaContract = Contract()
        exsaContract.symbol = contracts["EXSA"]['symbol']
        exsaContract.secType = contracts["EXSA"]["secType"]
        exsaContract.exchange = contracts["EXSA"]["exchange"]
        exsaContract.currency = contracts["EXSA"]["currency"]

        self.OrderActionStep(posx= self.actualPositions[0], contr=spyContract, orderId=orderId)
        orderId = self.nextId()
        self.OrderActionStep(posx= self.actualPositions[1], contr=vwoContract, orderId=orderId)
        orderId = self.nextId()
        self.OrderActionStep(posx= self.actualPositions[2], contr=exsaContract, orderId=orderId)

    def OrderActionStep(self, posx: float, contr : Contract, orderId, SL:bool=False):
        if not SL:
            pass
        if posx!=0:
            qty = int(round(posx))
            orderId = self.nextId()
            print(f"{orderId}: updating {contr.symbol} position: {qty}.")
            parent = Order()
            parent.orderId = orderId
            parent.orderType = "MKT"
            parent.tif = "OPG"
            parent.totalQuantity = qty
            parent.transmit = True
            self.parentorderId = parent.orderId
            if qty>0:
                parent.action = "BUY"
                
            elif qty<0:
                parent.action = "SELL"
                parent.totalQuantity = -qty

            self.placeOrder(parent.orderId, contr, parent)
            
            print(f"{orderId}: {parent.action}ING {qty} shares of {contr.symbol}")
            exec = Execution()
            exec.execId = str(parent.orderId)
            exec.orderId = parent.orderId
            exec.side = parent.action
            exec.shares = abs(qty)
            exec.cumQty = abs(qty)
            exec.price = self.y2.iloc[-1][contr.symbol]
            exec.time = datetime.now().strftime("%Y%m%d %H:%M:%S")
            self.execDetails(parent.orderId, contr, exec)
        else:
            print(f"no operation on {contr.symbol}")

    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        buy_price = int(round(execution.price))
        print(execution.side)
        if execution.side == "BUY":
            stop_price = round(0.6 * buy_price)
            action = "SELL"
        else:  # short
            stop_price = round(1.6 * buy_price)
            action = "BUY"

        stop_loss = Order()
        stop_loss.orderId = self.nextId()
        stop_loss.parentId = self.parentorderId
        stop_loss.action = action
        stop_loss.orderType = "STP"
        stop_loss.auxPrice = stop_price
        stop_loss.totalQuantity = execution.cumQty
        stop_loss.transmit = True

        print(f"Placing stop order {stop_loss.orderId} @ {stop_price} for {execution.cumQty} shares of {contract.symbol}")

        self.placeOrder(stop_loss.orderId, contract, stop_loss)
        
        print(f"execDetails. reqId: {reqId}, contract: {contract}, execution: {execution}")

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState: OrderState):
        print(f"ordreId: {orderId}, contract: {contract}, order: {order}")

    def orderStatus(self, orderId: OrderId, status: str, filled: float, remaining: float, avgFillPrice: float, permId:int,
                    parentId: int, lastFillPrice: float, clientId: int, whyHeld:str, mktCapPrice: float):
        print(f"orderStatus. orderId: {orderId}, status: {status}, filled: {filled}, remaining: {remaining}, avgFillPrice:{avgFillPrice}, permId: {permId}, parentId: {parentId}, lastFillPrice: {lastFillPrice}, clientId: {clientId}, whyHeld: {whyHeld}, mktCapPrice: {mktCapPrice}")

    def nextId(self):
        self.orderId +=1
        return self.orderId
    
    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderReject):
        print(f"[{errorTime}] reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, advancedOrderReject: {advancedOrderReject}")

    def updatePortfolio(self, contract: Contract, position: float , marketPrice : float, marketValue : float,
                        averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str):
        if contract.symbol == "SPY":
            self.spyCurrentPos = position
        elif contract.symbol == "VWO":
            self.vwoCurrentPos = position
        elif contract.symbol == "EXSA":
            self.exsaCurrentPos = position
        print("UpdatePortfolio.", "Symbol:", contract.symbol, "secType:", contract.secType, "Exchange", contract.exchange,
              "Position:", position, "MarketPrice", marketPrice, "MarketValue:", marketValue, "AverageCost:", averageCost,
              "UnrealizedPNL:", unrealizedPNL, "RealizedPNL:", realizedPNL, "AccountName:", accountName)
        
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        self.account_values[key] = {"value": val, "currency": currency}

        if key in ["BuyingPower"]:
            self.buyingpower = float(val)

        if key in ["NetLiquidation"]:
            self.NetLiquidation = float(val)


        if key in ["TotalCashValue"]:
            print(f"{key}: {val} {currency}")
            self.capital = float(val)

        #self.max_leverage_ibkr = self.buyingpower / self.NetLiquidation

    def updateAccountTime(self, timeStamp: str):
        print("UpdateAccountTime. Time:", timeStamp)
    
    def accountDownloadEnd(self, accountName: str):
        print("AccountDownloadEnd. Account:", accountName)

    def start(self):
        self.reqAccountUpdates(True, "")

    def stop(self):
        self.reqAccountUpdates(False, "")
        self.done = True
        self.disconnect()

    def historicalData(self, reqId, bar):
        symbol = self.symbol_map.get(reqId, "UNKNOWN")
        if symbol not in self.data:
            self.data[symbol] = []
        self.data[symbol].append({
            "date": bar.date,
            "open": bar.open,
        })
    
    def historicalDataEnd(self, reqId, start, end):
        symbol = self.symbol_map.get(reqId, "UNKNOWN")
        print(f"Historical Data ended for {symbol}: {start} → {end}")
        if symbol in self.data:
            df = pd.DataFrame(self.data[symbol])
            df["symbol"] = symbol
            self.df_final = pd.concat([self.df_final, df], ignore_index=True)
        self.completed_symbols.add(symbol)
        self.cancelHistoricalData(reqId)

    def createContract(self, info):
        contract = Contract()
        contract.symbol = info["symbol"]
        contract.secType = info["secType"]
        contract.currency = info["currency"]
        contract.exchange = info["exchange"]
        contract.primaryExchange = info["primaryExchange"]
        return contract

    def warmup(self):
        for name, info in contracts.items():
            contract = self.createContract(info)
            reqId = self.nextId()
            self.symbol_map[reqId] = name

            if info["currency"] == "USD":
                tz = "America/New_York"
                end_time = datetime.now(ZoneInfo(tz)).strftime("%Y%m%d %H:%M:%S") + " US/Eastern"
            else:
                tz = "Europe/Berlin"
                end_time = datetime.now(ZoneInfo(tz)).strftime("%Y%m%d %H:%M:%S") + " MET"

            self.reqHistoricalData(
                reqId, contract, end_time,
                "105 D", "1 day", "TRADES", 0, 1, False, []
            )
            print(f"Warmed up for {info['symbol']} (reqId={reqId})")

        while len(self.completed_symbols) < len(contracts):
            time.sleep(1)

        dfs = []
        for symbol, bars in self.data.items():
            df = pd.DataFrame(bars)
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"open": symbol})
            dfs.append(df.set_index("date"))

        # Fusion propre sur la colonne date
        merged = pd.concat(dfs, axis=1).sort_index()
        merged = merged.ffill().tail(100)
        return merged
    
    def kalman_filter (self):
        data = self.warmup()
        tickers = data.columns
        delta = 0.009
        Ve = 0.01
        q_pow = -1.0
        print(f"Running Kalman Fliter for parameters : delta : {delta}, Ve : {Ve}, q_pow : {q_pow}")

        n = data.shape[0]
        p = len(tickers) - 1

        # Construire la matrice X (tickers explicatifs + offset)
        X = np.zeros((n, p + 1))
        for j, ticker in enumerate(tickers[:-1]):
            X[:, j] = data[ticker]
        X[:, -1] = 1  # colonne offset

        # Initialisation des variables Kalman
        yhat = np.full_like(data[tickers[-1]], np.nan, dtype=float)
        e = np.full_like(data[tickers[-1]], np.nan, dtype=float)
        Q = np.full_like(data[tickers[-1]], np.nan, dtype=float)
        
        R = np.zeros((data.shape[1], data.shape[1]))
        P = np.zeros((data.shape[1], data.shape[1]))
        beta = np.full((data.shape[1], len(X)), np.nan)
        Vw = (delta / (1 - delta)) * np.eye(data.shape[1])

        beta[:, 0] = 0

        # === Boucle Kalman ===
        for t in range(data.shape[0]):
            if t > 0:
                beta[:, t] = beta[:, t - 1]           # Eq. 3.7
                R = P + Vw                            # Eq. 3.8

            yhat[t] = X[t, :] @ beta[:, t]            # Eq. 3.9
            Q[t] = X[t, :] @ R @ X[t, :].T + Ve       # Eq. 3.10

            e[t] = (data[tickers[-1]].iloc[t] - yhat[t])                     # erreur

            K = R @ X[t, :].T / Q[t]                  # gain
            beta[:, t] = beta[:, t] + K * e[t]        # Eq. 3.11
            P = R - np.outer(K, X[t, :]) @ R          # Eq. 3.12

        Q = Q**(q_pow)

        return e, Q, beta, X, data

    def _compute_positions(self):
        e, Q, beta, X, data = self.kalman_filter()
        tickers = data.columns
        y2 = np.zeros((data.shape[0], len(tickers)))

        for j,ticker in enumerate(tickers[:-1]):
            y2[:, j] = X[:, j]

        y2[:,-1] = data[tickers[-1]]

        longsEntry = e < -np.sqrt(Q)
        longsExit = e > -np.sqrt(Q)
        shortsEntry = e > np.sqrt(Q)
        shortsExit = e < np.sqrt(Q)

        numUnitsLong = np.full(len(y2), np.nan)
        numUnitsShort = np.full(len(y2), np.nan)

        numUnitsLong[0] = 0
        numUnitsLong[longsEntry] = 1
        numUnitsLong[longsExit] = 0
        numUnitsLong = np.array(pd.DataFrame(numUnitsLong).ffill())

        numUnitsShort[0] = 0
        numUnitsShort[shortsEntry] = -1
        numUnitsShort[shortsExit] = 0
        numUnitsShort = np.array(pd.DataFrame(numUnitsShort).ffill())

        numUnits = numUnitsLong + numUnitsShort

        hedge_ratios = np.zeros((data.shape[0], len(tickers)))

        for j,ticker in enumerate(tickers[:-1]):
            hedge_ratios[:, j] = -beta[j, :]

        hedge_ratios[:,-1] = np.ones(len(beta[0, :]))

        positions = np.round(np.tile(numUnits.reshape(-1, 1), (1, y2.shape[1])) * hedge_ratios * y2,2)
        print("Positions Computed.")
        norm_positions = (np.array(positions[-1, :]))/(np.abs(np.array(positions[-1, :])).sum())
        currentPrices = np.array(data.iloc[-1].values)
        return norm_positions, currentPrices, positions, data
    
    def _calculate_orders(self):
        positions, currentPrices,positions_b,y2 = self._compute_positions()
        qty = np.round((self.capital*(positions))/currentPrices,2)
        toBuy = [float(qty[0]-float(self.spyCurrentPos)),float(qty[1]-float(self.vwoCurrentPos)),float(qty[2]-float(self.exsaCurrentPos))]
        return toBuy, positions_b, y2
    
    def Kelly(self):
        toBuy, positions, y2 = self._calculate_orders()
        self.y2 = y2
        lag_positions = np.roll(positions, 1, axis=0)
        lag_y2 = np.roll(y2, 1, axis=0)

        # Mettre NaN sur la première ligne (pas de précédent)
        lag_positions[0, :] = np.nan
        lag_y2[0, :] = np.nan

        pnl = np.nansum(lag_positions * (y2 - lag_y2) / lag_y2, axis=1)
        ret = pnl / np.nansum(np.abs(lag_positions), axis=1)
        ret[np.isnan(ret)] = 0
        ret = pd.Series(ret)
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        ret_v = ret.copy() - (-3.5 / 252)
        m = (ret_v.rolling(100).mean()*100).dropna()
        v = (100*(ret_v.rolling(100).std())**2).dropna()
        f = m/(2*v)
        max_leverage_th = 10.74

        max_leverage = 6 #IBKR's net margin limit
        if MARGIN :
            #f = float(f.clip(lower = -1, upper=1).iloc[0])
            f = float(f.clip(lower = -max_leverage, upper = max_leverage).iloc[0])
        else:
            f= float(f.clip(lower=-1, upper=1).iloc[0])
        self.actualPositions = np.round([f*toBuy[0], f*toBuy[1], float(np.clip((f*toBuy[2]),-55_000, 55_000)) ])
        print("actualPositions:",self.actualPositions)
        self.nextValidId(self.nextId())

    # === 5. EXPORT / CHECK ===
    def summary(self):
        return pd.DataFrame(
            self.data,
            index=self.data.index
        )

app = TestApp()
app.connect("54.242.57.2", port, 0)

threading.Thread(target=app.run).start()
time.sleep(2)
ordercancelor = OrderCancel()
app.reqGlobalCancel(ordercancelor)
app.Kelly()

time.sleep(2)

app.disconnect()
