import numpy as np
import matplotlib.pyplot as plt


class MinEnv1:
    def __init__(self):
        self.Dataset = None
        self.ProductDataset = None

        # Process Status
        self.DayRange = 0
        self.StepRange = 0
        self.TotalRange = 0

        self.DayPointer = 0
        self.StepPointer = 0

        # Trade Status
        self.InitBalance = 0
        self.TradeFee = 0
        self.ProductNumber = 0

        self.CashFlow = 0
        self.StockValues = 0
        self.QuantityStatus = None  # list(list(price, quantity))
        self.TradeRecord = None  # list(index, price, quantity, time)
        self.TradeTime = 0
        self.TotalTradeFee = 0

        # Env Setting
        self.ShortOption = True
        self.NegativeStop = False
        self.OnlyPlotProfit = False
        self.OverNight = False
        self.End = True
        self.Theta = 0

        # Plot Setting
        self.MultiPlot = False
        self.CashFlowHistory = []
        self.StockValuesHistory = []
        self.ActionHistory = []

    def _get_status_dict(self, end=True) -> dict:
        self.End = end
        status = None if self.End else self.Dataset[self.DayPointer][self.StepPointer]
        product_status = None if self.End else self.ProductDataset[self.DayPointer, self.StepPointer]
        info = {
            'process_status': {
                'day_pointer': self.DayPointer,
                'step_pointer': self.StepPointer,
                'total_step': self.TotalRange,
                'process_step': self.DayPointer * self.StepRange + self.StepPointer
            },
            'trade_status': {
                'cash_flow': self.CashFlow,
                'stock_values': self.StockValues,
                'stock_quantity': self.QuantityStatus,  # price, quantity
                'total_trade_fee': self.TotalTradeFee,
            },
            'env_status': status,
            'product_status': product_status,
            'end': self.End
        }
        return info

    def get_next_state(self):
        return None if self.End else self.Dataset[self.DayPointer][self.StepPointer]

    def init_env(self,
                 dataset: np.array,
                 product_dataset: np.array,
                 balance=0,
                 trade_fee=0,
                 theta=0,
                 short_option=True,
                 negative_balance_stop=False,
                 only_plot_profit=False,
                 over_night=False) -> dict:
        if len(dataset.shape) != 3:
            msg = "Get dimension of dataset is {}, expect 3\n".format(len(dataset.shape))
            msg += "[day[min[feature[]]]"
            raise ValueError(msg)

        # Process Status
        self.Dataset = dataset
        self.DayRange = len(self.Dataset)
        self.StepRange = len(self.Dataset[0])
        self.TotalRange = self.DayRange * self.StepRange

        self.DayPointer = 0
        self.StepPointer = 0

        if len(product_dataset.shape) != 3:
            msg = "Get dimension of product dataset is {}, expect 3\n".format(len(dataset.shape))
            msg += "[day[min[values]]]"
            raise ValueError(msg)

        if product_dataset.shape[0:2] != dataset.shape[0:2]:
            msg = "time step of product_dataset not equal to dataset"
            raise ValueError(msg)

        self.ProductDataset = product_dataset

        # Trade Status
        self.InitBalance = balance
        self.TradeFee = trade_fee
        self.ProductNumber = int(self.ProductDataset.shape[-1])

        self.TradeTime = 0
        self.TotalTradeFee = 0
        self.CashFlow = self.InitBalance
        self.QuantityStatus = np.zeros((self.ProductNumber, 2))  # list(quantity)
        self.TradeRecord = []  # list(index, price, quantity, time)

        # Env Setting
        self.ShortOption = short_option
        self.NegativeStop = negative_balance_stop
        self.OnlyPlotProfit = only_plot_profit
        self.OverNight = over_night
        self.End = False
        self.Theta = theta

        # Plot Setting
        self.CashFlowHistory = []
        self.StockValuesHistory = []
        self.ActionHistory = []

        return self._get_status_dict(False)

    def step(self, quantity_action=None):
        if self.End:
            msg = 'The Env is End, please init_env to reset the env'
            raise ValueError(msg)

        if len(quantity_action) != self.ProductNumber:
            msg = "length of quantity_action is {}, expect {}".format(len(quantity_action), self.ProductNumber)
            raise ValueError(msg)

        end = False

        # Trade Action
        action_note = []
        for i, cq in zip(range(self.ProductNumber), quantity_action):
            avg_p = self.QuantityStatus[i][0]
            tq = self.QuantityStatus[i][1]

            # Check Input is valid
            if not isinstance(int(cq), int):
                msg = "type of param quantity_action is {}, expect 'int'".format(type(cq))
                raise ValueError(msg)

            # Check is End -> sell all
            if not self.OverNight:
                if self.StepPointer + 1 >= self.StepRange - 1:
                    cq = - tq

            # Check can Short
            if not self.ShortOption:
                if cq < 0 and tq - cq < 0:
                    msg = "You can not Sell more than you have"
                    raise ValueError(msg)

            if cq > 0:
                action_note.append(1)
            elif cq == 0:
                action_note.append(0)
            else:
                action_note.append(-1)

            if cq != 0:
                p = self.ProductDataset[self.DayPointer, self.StepPointer, i]

                # Trade Fee
                self.CashFlow += - self.TradeFee
                self.TotalTradeFee += self.TradeFee
                self.TradeTime += 1
                self.TradeRecord.append([i, p, cq, "{}-{}".format(self.DayPointer, self.StepPointer)])

                # Reduce
                if tq > 0 > cq or tq < 0 < cq:
                    # To Zero When Over +/-
                    if tq + cq > 0 > tq or tq + cq < 0 < tq:
                        self.CashFlow += (p - avg_p) * tq + avg_p * abs(tq)
                        cq += tq
                        tq = 0
                        # Trade Fee
                        self.CashFlow += - self.TradeFee
                        self.TotalTradeFee += self.TradeFee
                    # Normal Reduce
                    else:
                        self.CashFlow += (p - avg_p) * (-cq) + avg_p * abs(cq)
                        tq += cq
                        cq = 0

                # Increase
                if cq != 0:
                    self.CashFlow -= abs(cq) * p
                    avg_p = (avg_p * tq + p * cq) / (tq + cq)
                    tq += cq

                # Sum
                self.QuantityStatus[i][0] = avg_p
                self.QuantityStatus[i][1] = tq

        # Cal Stock Value
        values = 0
        for i in range(self.ProductNumber):
            avg_p = self.QuantityStatus[i][0]
            tq = self.QuantityStatus[i][1]
            p = self.ProductDataset[self.DayPointer, self.StepPointer, i]
            values += (p - avg_p) * tq + avg_p * abs(tq)
        self.StockValues = values

        # Reduce Theta
        self.CashFlow -= self.Theta

        self.CashFlowHistory.append(self.CashFlow)
        self.StockValuesHistory.append(self.StockValues)
        self.ActionHistory.append(action_note)

        if self.CashFlow < 0 and self.NegativeStop:
            end = True

        # Next Day // Step + 1
        self.StepPointer += 1
        if self.StepPointer >= self.StepRange - 1:
            self.DayPointer += 1
            self.StepPointer = 0
            if self.DayPointer >= self.DayRange:
                end = True

        # After Values
        if not end:
            values = 0
            for i in range(self.ProductNumber):
                avg_p = self.QuantityStatus[i][0]
                tq = self.QuantityStatus[i][1]
                p = self.ProductDataset[self.DayPointer, self.StepPointer, i]
                values += (p - avg_p) * tq + avg_p * abs(tq)
            self.StockValues = values

        return self._get_status_dict(end)

    def plot_result(self, plot_product_index=None, pause=0.0):
        def plot_action(i, ax):
            index = np.where(action_note[:, i] == 1)[0]
            ax.scatter(x=index, y=product_info[index, i], marker='o', color='g')
            index = np.where(action_note[:, i] == -1)[0]
            ax.scatter(x=index, y=product_info[index, i], marker='o', color='r')

        action_history = np.array(self.ActionHistory).reshape(-1, self.ProductNumber)
        if plot_product_index is not None:
            if len(plot_product_index) > 2:
                msg = "plot product can not over 2"
                raise ValueError(msg)
            if len([x for x in plot_product_index if x >= self.ProductNumber]) > 0:
                msg = "product index out of range"
                raise ValueError(msg)
            action_note = action_history[:, plot_product_index]
        else:
            plot_product_index = [x for x in range(self.ProductNumber) if x < 2]
            action_note = action_history[:, plot_product_index]

        product_info = self.ProductDataset[:, :-1, plot_product_index].reshape(-1, len(plot_product_index))

        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax1.plot(product_info[:, 0])
        plot_action(0, ax1)

        if len(plot_product_index) == 2:
            ax2 = ax1.twinx()
            ax2.plot(product_info[:, 1])
            plot_action(1, ax2)

        plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
        total = np.array(self.CashFlowHistory) + np.array(self.StockValuesHistory)
        if self.OnlyPlotProfit:
            plt.plot(total - self.InitBalance)
        else:
            plt.plot(total)
            plt.plot(self.CashFlowHistory, label="Cash Flow")
            plt.plot(self.StockValuesHistory, label="Stock Values")
        if pause == 0:
            plt.show()
            plt.close()
        else:
            plt.pause(pause)
            plt.clf()


    # Example 1
    '''
    from mymodule import FutuFunction, DataAnalysis

    with FutuFunction.FutuFunction() as futu:
        min_datas = futu.get_kline_data('HK.999010', 3, 1, futu.Opt.KLType.K_1M, to_list=True)

    min_datan = []
    for min_data in min_datas:
        min_data = DataAnalysis.time_cutter(min_data, ('09:00:00', '16:30:00'))
        min_datan.append(np.array(min_data[['close']]))
    min_datan = np.array(min_datan)

    env = MinEnv1()
    s = env.init_env(min_datan, (min_datan - 20000), 2000, 0, negative_balance_stop=False, over_night=False)
    # print(s)
    cot50 = 0
    done = False
    while not done:
        s = env.step(np.random.choice([-1, 0, 1], 1, p=[0.01, 0.98, 0.01]))
        print(s)
        done = s['end']
        # env.plot_result(pause=0.1)
    print(env.TradeRecord)
    env.plot_result()
    '''


class DayEnv1:
    def __init__(self):
        self.Dataset = None
        self.ProductDataset = None

        # Process Status
        self.DayRange = 0
        self.DayPointer = 0

        # Trade Status
        self.InitBalance = 0
        self.TradeFee = 0
        self.ProductNumber = 0

        self.CashFlow = 0
        self.StockValues = 0
        self.QuantityStatus = None  # list(list(price, quantity))
        self.TradeRecord = None  # list(index, price, quantity, time)
        self.TradeTime = 0
        self.TotalTradeFee = 0

        # Env Setting
        self.ShortOption = True
        self.NegativeStop = False
        self.OnlyPlotProfit = False
        self.End = True

        # Plot Setting
        self.CashFlowHistory = []
        self.StockValuesHistory = []
        self.ActionHistory = []

    def _get_status_dict(self, end=True) -> dict:
        self.End = end
        status = None if self.End else self.Dataset[self.DayPointer]
        product_status = None if self.End else self.ProductDataset[self.DayPointer]
        info = {
            'process_status': {
                'day_pointer': self.DayPointer,
                'step_pointer': None,
                'total_step': None,
                'process_step': self.DayPointer
            },
            'trade_status': {
                'cash_flow': self.CashFlow,
                'stock_values': self.StockValues,
                'stock_quantity': self.QuantityStatus,
                'total_trade_fee': self.TotalTradeFee,
            },
            'env_status': status,
            'product_status': product_status,
            'end': self.End
        }
        return info

    def init_env(self,
                 dataset: np.ndarray,
                 product_dataset: np.ndarray,
                 balance=0,
                 trade_fee=0,
                 short_option=True,
                 negative_balance_stop=False,
                 only_plot_profit=False) -> dict:
        if len(dataset.shape) != 2:
            msg = "Get dimension of dataset is {}, expect 2\n".format(len(dataset.shape))
            msg += "[min[feature[]]"
            raise ValueError(msg)

        # Process Status
        self.Dataset = dataset
        self.DayRange = len(self.Dataset)

        self.DayPointer = 0

        if len(product_dataset.shape) != 2:
            msg = "Get dimension of product dataset is {}, expect 2\n".format(len(product_dataset.shape))
            msg += "[min[values]]"
            raise ValueError(msg)

        if product_dataset.shape[0] != dataset.shape[0]:
            msg = "time step of product_dataset not equal to dataset"
            raise ValueError(msg)

        self.ProductDataset = product_dataset

        # Trade Status
        self.InitBalance = balance
        self.TradeFee = trade_fee
        self.ProductNumber = int(self.ProductDataset.shape[-1])

        self.TradeTime = 0
        self.TotalTradeFee = 0
        self.CashFlow = self.InitBalance
        self.QuantityStatus = np.zeros((self.ProductNumber, 2))  # list(quantity)
        self.TradeRecord = []  # list(index, price, quantity, time)

        # Env Setting
        self.ShortOption = short_option
        self.NegativeStop = negative_balance_stop
        self.OnlyPlotProfit = only_plot_profit
        self.End = False

        # Plot Setting
        self.CashFlowHistory = []
        self.StockValuesHistory = []
        self.ActionHistory = []

        return self._get_status_dict(False)

    def step(self, quantity_action=None):
        if self.End:
            msg = 'The Env is End, please init_env to reset the env'
            raise ValueError(msg)

        if len(quantity_action) != self.ProductNumber:
            msg = "length of quantity_action is {}, expect {}".format(len(quantity_action), self.ProductNumber)
            raise ValueError(msg)

        end = False

        action_note = []
        for i, cq in zip(range(self.ProductNumber), quantity_action):
            avg_p = self.QuantityStatus[i][0]
            tq = self.QuantityStatus[i][1]

            # Check Input is valid
            if not isinstance(int(cq), int):
                msg = "type of param quantity_action is {}, expect 'int'".format(type(cq))
                raise ValueError(msg)

            # Check can Short
            if not self.ShortOption:
                if cq < 0 and tq - cq < 0:
                    msg = "You can not Sell more than you have"
                    raise ValueError(msg)

            if cq > 0:
                action_note.append(1)
            elif cq == 0:
                action_note.append(0)
            else:
                action_note.append(-1)

            if cq != 0:
                p = self.ProductDataset[self.DayPointer, i]

                # Trade Fee
                self.CashFlow += - self.TradeFee
                self.TradeTime += 1
                self.TotalTradeFee += self.TradeFee
                self.TradeRecord.append([i, p, cq, "{}".format(self.DayPointer)])
                # Reduce
                if tq > 0 > cq or tq < 0 < cq:
                    # Negative to Positive // Positive to Negative
                    if tq + cq > 0 > tq or tq + cq < 0 < tq:
                        self.CashFlow += (p - avg_p) * tq + avg_p * abs(tq)
                        cq += tq
                        tq = 0
                    # Normal Reduce
                    else:
                        self.CashFlow += (p - avg_p) * (-cq) + avg_p * abs(cq)
                        tq += cq
                        cq = 0

                # Increase
                if cq != 0:
                    self.CashFlow -= abs(cq) * p
                    avg_p = (avg_p * tq + p * cq) / (tq + cq)
                    tq += cq

                # Sum
                self.QuantityStatus[i][0] = avg_p
                self.QuantityStatus[i][1] = tq

        values = 0
        for i in range(self.ProductNumber):
            avg_p = self.QuantityStatus[i][0]
            tq = self.QuantityStatus[i][1]
            p = self.ProductDataset[self.DayPointer, i]
            values += (p - avg_p) * tq + avg_p * abs(tq)
        self.StockValues = values

        self.CashFlowHistory.append(self.CashFlow)
        self.StockValuesHistory.append(self.StockValues)
        self.ActionHistory.append(action_note)

        if self.CashFlow < 0 and self.NegativeStop:
            end = True
        # Step + 1
        self.DayPointer += 1
        if self.DayPointer >= self.DayRange:
            end = True

        # After Values
        # if not end:
        #     values = 0
        #     for i, tq in zip(range(self.ProductNumber), self.QuantityStatus):
        #         p = self.ProductDataset[self.DayPointer, self.StepPointer]
        #         values += tq * p
        #     self.StockValues = values

        return self._get_status_dict(end)

    def plot_result(self, plot_product_index=None, pause=0.0):
        def plot_action(i, ax):
            index = np.where(action_note[:, i] == 1)[0]
            ax.scatter(x=index, y=product_info[index, i], marker='o', color='g')
            index = np.where(action_note[:, i] == -1)[0]
            ax.scatter(x=index, y=product_info[index, i], marker='o', color='r')

        action_history = np.array(self.ActionHistory).reshape(-1, self.ProductNumber)
        if plot_product_index is not None:
            if len(plot_product_index) > 2:
                msg = "plot product can not over 2"
                raise ValueError(msg)
            if len([x for x in plot_product_index if x >= self.ProductNumber]) > 0:
                msg = "product index out of range"
                raise ValueError(msg)
            action_note = action_history[:, plot_product_index]
        else:
            plot_product_index = [x for x in range(self.ProductNumber) if x < 2]
            action_note = action_history[:, plot_product_index]

        product_info = self.ProductDataset[:, plot_product_index].reshape(-1, len(plot_product_index))

        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax1.plot(product_info[:, 0])
        plot_action(0, ax1)

        if len(plot_product_index) == 2:
            ax2 = ax1.twinx()
            ax2.plot(product_info[:, 1])
            plot_action(1, ax2)

        plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
        total = np.array(self.CashFlowHistory) + np.array(self.StockValuesHistory)
        if self.OnlyPlotProfit:
            plt.plot(total - self.InitBalance)
        else:
            plt.plot(total)
            plt.plot(self.CashFlowHistory, label="Cash Flow")
            plt.plot(self.StockValuesHistory, label="Stock Values")
        if pause == 0:
            plt.show()
            plt.close()
        else:
            plt.pause(pause)
            plt.clf()


if __name__ == '__main__':
    from mymodule import FutuFunction

    with FutuFunction.FutuFunction() as futu:
        day_data = futu.get_kline_data('HK.999010', 1200, 1, futu.Opt.KLType.K_DAY)

    dataset = np.array(day_data[['close']])

    env = DayEnv1()
    s = env.init_env(dataset, dataset, 0, 0, negative_balance_stop=False)
    # print(s)
    cot50 = 0
    done = False
    while not done:
        s = env.step(np.random.choice([-1, 0, 1], 1, p=[0.01, 0.98, 0.01]))
        print(s)
        done = s['end']
        # env.plot_result(pause=0.1)
    print(env.TradeRecord)
    env.plot_result()
