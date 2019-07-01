import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from pylab import rcParams

#无风险收益率，在计算夏普比率时使用
Rf = 0.04
x = []
count = 0
#Reaction Trend System
class RTS:
    dealData = {'date':[], 'position':[], 'cashCapital':[], 'totalCapital':[], 'dailyYield':[], 'profit':[], 'predSitu':[]}
    tradingCost = 0
    slipPage = 0
    position = 0
    initCapital = 0
    cashCapital = 0
    totalCapital = 0
    dailyYield = 0
    lastClose = 0
    def __init__(self, tC, sP, iC):
        self.tradingCost = tC
        self.slipPage = sP
        self.initCapital = iC
        self.cashCapital = iC

    #根据fastAvg与slowAvg来进行交易变动仓位，同时记录信息
    def dealByMa(self, date, fastAvg, slowAvg, openPrice, closePrice):
        #变动仓位
        if(fastAvg > slowAvg and self.position != 1):
            buyPrice = ((openPrice + self.slipPage) * (1 + self.tradingCost)) * (1 - self.position)
            if(self.cashCapital > buyPrice):
                self.cashCapital -= buyPrice
                self.position = 1
        elif(fastAvg < slowAvg and self.position != -1):
            sellPrice = ((openPrice - self.slipPage) * (1 - self.tradingCost)) * (self.position + 1)
            self.cashCapital += sellPrice
            self.position = -1
        else:
            pass
        if((fastAvg > slowAvg and closePrice > openPrice) or (fastAvg < slowAvg and closePrice < openPrice)):
            self.dealData['predSitu'].append(1)
        else:
            self.dealData['predSitu'].append(0)
        #更新记录
        self.totalCapital = self.cashCapital + self.position*closePrice
        self.lastClose = closePrice
        self.dailyYield = 0 if len(self.dealData['totalCapital']) == 0 else (self.totalCapital - self.dealData['totalCapital'][-1])/self.dealData['totalCapital'][-1]
        self.profit = (self.totalCapital - self.initCapital)/self.initCapital
        self.dealData['date'].append(date)
        self.dealData['position'].append(self.position)
        self.dealData['cashCapital'].append(self.cashCapital)
        self.dealData['totalCapital'].append(self.totalCapital)
        self.dealData['profit'].append(self.profit)
        self.dealData['dailyYield'].append(self.dailyYield)

    def dealByPred(self, date, preClosePrice, openPrice, closePrice, predClosePrice):
        if(predClosePrice > preClosePrice and self.position != 1):
            buyPrice = ((openPrice + self.slipPage) * (1 + self.tradingCost)) * (1 - self.position)
            if(self.cashCapital > buyPrice):
                self.cashCapital -= buyPrice
                self.position = 1
        elif(predClosePrice < preClosePrice and self.position != -1):
            sellPrice = ((openPrice - self.slipPage) * (1 - self.tradingCost)) * (self.position + 1)
            self.cashCapital += sellPrice
            self.position = -1
        else:
            pass
        if((predClosePrice > preClosePrice and closePrice > preClosePrice) or (predClosePrice < preClosePrice and closePrice < preClosePrice)):
            self.dealData['predSitu'].append(1)
        else:
            self.dealData['predSitu'].append(0)
        #更新记录
        self.totalCapital = self.cashCapital + self.position*closePrice
        self.lastClose = closePrice
        self.dailyYield = 0 if len(self.dealData['totalCapital']) == 0 else (self.totalCapital - self.dealData['totalCapital'][-1])/self.dealData['totalCapital'][-1]
        self.profit = (self.totalCapital - self.initCapital)/self.initCapital
        self.dealData['date'].append(date)
        self.dealData['position'].append(self.position)
        self.dealData['cashCapital'].append(self.cashCapital)
        self.dealData['totalCapital'].append(self.totalCapital)
        self.dealData['profit'].append(self.profit)
        self.dealData['dailyYield'].append(self.dailyYield)

    def dealByPredV2(self, date, preClosePrice, openPrice, closePrice, predTrend):
        if(predTrend == 1 and self.position != 1):
            buyPrice = ((openPrice + self.slipPage) * (1 + self.tradingCost)) * (1 - self.position)
            if(self.cashCapital > buyPrice):
                self.cashCapital -= buyPrice
                self.position = 1
        elif(predTrend == -1 and self.position != -1):
            sellPrice = ((openPrice - self.slipPage) * (1 - self.tradingCost)) * (self.position + 1)
            self.cashCapital += sellPrice
            self.position = -1
        else:
            pass
        if((predTrend == 1 and closePrice > preClosePrice) or (predTrend == -1 and closePrice < preClosePrice)):
            self.dealData['predSitu'].append(1)
        else:
            self.dealData['predSitu'].append(0)
        #更新记录
        self.totalCapital = self.cashCapital + self.position*closePrice
        self.lastClose = closePrice
        self.dailyYield = 0 if len(self.dealData['totalCapital']) == 0 else (self.totalCapital - self.dealData['totalCapital'][-1])/self.dealData['totalCapital'][-1]
        self.profit = (self.totalCapital - self.initCapital)/self.initCapital
        self.dealData['date'].append(date)
        self.dealData['position'].append(self.position)
        self.dealData['cashCapital'].append(self.cashCapital)
        self.dealData['totalCapital'].append(self.totalCapital)
        self.dealData['profit'].append(self.profit)
        self.dealData['dailyYield'].append(self.dailyYield)

    def dealByPredV3(self, date, closePrice, preClosePrice, predTrend):
        if(predTrend == 1 and self.position != 1):
            buyPrice = ((preClosePrice + self.slipPage) * (1 + self.tradingCost)) * (1 - self.position)
            if(self.cashCapital > buyPrice):
                self.cashCapital -= buyPrice
                self.position = 1
        elif(predTrend == -1 and self.position != -1):
            sellPrice = ((preClosePrice - self.slipPage) * (1 - self.tradingCost)) * (self.position + 1)
            self.cashCapital += sellPrice
            self.position = -1
        else:
            pass
        if((predTrend == 1 and closePrice > preClosePrice) or (predTrend == -1 and closePrice < preClosePrice)):
            self.dealData['predSitu'].append(1)
        else:
            self.dealData['predSitu'].append(0)
        #更新记录
        self.totalCapital = self.cashCapital + self.position*closePrice
        self.lastClose = closePrice
        self.dailyYield = 0 if len(self.dealData['totalCapital']) == 0 else (self.totalCapital - self.dealData['totalCapital'][-1])/self.dealData['totalCapital'][-1]
        self.profit = (self.totalCapital - self.initCapital)/self.initCapital
        self.dealData['date'].append(date)
        self.dealData['position'].append(self.position)
        self.dealData['cashCapital'].append(self.cashCapital)
        self.dealData['totalCapital'].append(self.totalCapital)
        self.dealData['profit'].append(self.profit)
        self.dealData['dailyYield'].append(self.dailyYield)

    def dealBymaxReturn(self, date, preClosePrice, openPrice, closePrice, predTrend):
        if(closePrice - openPrice > 0 and self.position != 1):
            buyPrice = ((openPrice + self.slipPage) * (1 + self.tradingCost)) * (1 - self.position)
            if(self.cashCapital > buyPrice):
                self.cashCapital -= buyPrice
                self.position = 1
        elif(closePrice - openPrice < 0 and self.position != -1):
            sellPrice = ((openPrice - self.slipPage) * (1 - self.tradingCost)) * (self.position + 1)
            self.cashCapital += sellPrice
            self.position = -1
        else:
            pass
        if((closePrice - preClosePrice > 0 and closePrice > openPrice) or (closePrice - preClosePrice < 0 and closePrice < openPrice)):
            self.dealData['predSitu'].append(1)
        else:
            self.dealData['predSitu'].append(0)
        #更新记录
        self.totalCapital = self.cashCapital + self.position*closePrice
        self.lastClose = closePrice
        self.dailyYield = 0 if len(self.dealData['totalCapital']) == 0 else (self.totalCapital - self.dealData['totalCapital'][-1])/self.dealData['totalCapital'][-1]
        self.profit = (self.totalCapital - self.initCapital)/self.initCapital
        self.dealData['date'].append(date)
        self.dealData['position'].append(self.position)
        self.dealData['cashCapital'].append(self.cashCapital)
        self.dealData['totalCapital'].append(self.totalCapital)
        self.dealData['profit'].append(self.profit)
        self.dealData['dailyYield'].append(self.dailyYield)

    #得到对应属性关于日期的曲线
    def getPlot(self, name):
        df = pd.DataFrame(self.dealData)
        df.plot(x='date', y=name)
        plt.show()
    
    #得到年化收益率、最大回撤与夏普比率的指标
    def getAnalysis(self):
        #计算年收益率
        startDate = self.dealData['date'][0].split('/')
        endDate = self.dealData['date'][-1].split('/')
        print(endDate, startDate)
        peroid = int(endDate[0]) - int(startDate[0]) + (int(endDate[1]) - int(startDate[1]))/12 + (int(endDate[2]) - int(startDate[2]))/365
        #annualReturn = (self.totalCapital / self.initCapital) ** (1/peroid) - 1
        annualReturn = (self.totalCapital / self.initCapital) ** (250/len(self.dealData['date'])) - 1
        #计算最大回撤
        tmpMax = self.dealData['totalCapital'][0]
        maxDrawdown = 0
        for capital in self.dealData['totalCapital']:
            tmpMax = capital if capital > tmpMax else tmpMax
            maxDrawdown = (tmpMax - capital)/tmpMax if (tmpMax - capital)/tmpMax > maxDrawdown else maxDrawdown
        #计算夏普比率
        sharpeRatio = (annualReturn - Rf)/np.std(self.dealData['dailyYield'],ddof=1)
        print("Annual Return: %.2f %%" % (annualReturn*100))
        print("Max DrawDown: %.2f %%" % (maxDrawdown*100))
        print("SharpeRatio: %.2f" % sharpeRatio)
        accuracy = sum(self.dealData['predSitu'])/len(self.dealData['predSitu'])
        print("Pred Accuracy: %.2f %%" % (accuracy*100))
        return annualReturn

#MA回撤的主要函数，读取filePath对应文件中的数据进行计算并创建RTS实例，使用其进行交易（slowPeriod 必须比 fastPeriod 大）
def MaBackTest(fastPeriod, slowPeriod, filePath, tradingCost, slipPage, capital):
    rts = RTS(tradingCost, slipPage, capital)
    df = pd.read_csv(filePath)
    
    # x, y = [], []
    # for i in range(len(df)):
    #     if(df.iloc[i]['close']-df.iloc[i]['open']>0):
    #         x.append(1)
    #     elif(df.iloc[i]['close']-df.iloc[i]['open']<0):
    #         x.append(-1)
    #     else:
    #         x.append(0)
    #     if(i>=1 and df.iloc[i]['close']-df.iloc[i-1]['open']>0):
    #         y.append(1)
    #     elif(i>=1 and df.iloc[i]['close']-df.iloc[i-1]['open']<0):
    #         y.append(-1)
    #     else:
    #         y.append(0)
    # x = np.array(x).reshape(-1,1)
    # y = np.array(y).reshape(-1,1)
    # df=pd.DataFrame(np.hstack((x, y)))
    # print(x,y,np.hstack((x, y)))
    # df.columns=['inday_trend','outday_trend']
    # df.to_csv('./trend.csv')
    # return
    fastClose = [0]*fastPeriod
    slowClose = [0]*slowPeriod
    fastIndex, slowIndex = 0, 0
    fastAvg, slowAvg = 0, 0
    for i in range(len(df)):
        row = df.iloc[i]
        if(i < slowPeriod):
            slowClose[i] = row['close']
            if(i < fastPeriod):
                fastClose[i] = row['close']
        else:
            fastAvg = sum(fastClose)/fastPeriod
            slowAvg = sum(slowClose)/slowPeriod
        fastClose[fastIndex%fastPeriod] = row['close']
        slowClose[slowIndex%slowPeriod] = row['close']
        fastIndex += 1
        slowIndex += 1
        rts.dealByMa(row['date'], fastAvg, slowAvg, row['open'], row['close'])
    #可以根据需要修改下面一行中参数的名字来得到不同曲线
    rts.getPlot('totalCapital')
    return(rts.getAnalysis())

def PredBackTest(filePath, tradingCost, slipPage, capital):
    rts = RTS(tradingCost, slipPage, capital)
    df = pd.read_csv(filePath)
    for i in range(1, len(df)):
        lastRow = df.iloc[i-1]
        row = df.iloc[i]
        rts.dealByPred(row['date'], lastRow['close'], row['open'], row['close'], row['pred_close'])
        # rts.dealByPredV3(row['Date'], row['Adj_Close_stock'], lastRow['Adj_Close_stock'], row['Random Forest(Multiple Input)'])
    #可以根据需要修改下面一行中参数的名字来得到不同曲线
    rts.getPlot('totalCapital')
    return(rts.getAnalysis())

#测试用，得到期货走势曲线
def getClose(path):
    df = pd.read_csv(path)
    df.plot(x='date', y='close')
    plt.show()

def getPredClose(path):
    df = pd.read_csv(path)
    ax = df.plot(x='date', y='pred_close', legend='pred_close')
    df.plot(x='date', y='real_close', legend='real_close', ax=ax)
    plt.show()

#使用遍历参数的方法得到年化最大的长周期与短周期参数
def optimizeMaParam():
    bestFast, bestSlow, maxReturn = 0, 0, 0
    for i in range(2, 9):
        for j in range(10, 50):
            tmpReturn = MaBackTest(i, j, './rb000.csv', 0.0002, 1, 10000)
            if(tmpReturn > maxReturn):
                bestFast, bestSlow, maxReturn = i, j, tmpReturn
    print(bestFast, bestSlow, maxReturn)

def getPredRes(path):
    df = pd.read_csv(path)
    rc = df.loc[:, "real_close"]
    pc = df.loc[:, "pred_close"]
    count = 0
    for i in range(1, rc.shape[0]):
        if((pc[i]-pc[i-1])*(rc[i]-rc[i-1]) > 0):
            count+=1
    print(count, rc.shape[0]-1)
    print(count/(rc.shape[0]-1))
    rcParams['figure.figsize'] = 10, 8
    ax = df.plot(y=['real_close','pred_close'], color=['red','blue'], grid=True)
    plt.show()

def main():
    # MaBackTest(5, 20, './rb000.csv', 0.0002, 1, 10000)
    getClose("rb000.csv")
    # PredBackTest('pred_data.csv', 0.0002, 1, 10000)
    
# df=pd.read_csv('./trend.csv')
# count = 0
# for i in range(20, len(df)):
#     if(df.iloc[i]['outday_trend'] == df.iloc[i]['inday_trend']):
#         count+=1
# print(count,len(df))
# print(count/(len(df)))

main()