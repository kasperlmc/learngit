from gmsdk.api import StrategyBase
from gmsdk import md
from gmsdk.enums import *
import arrow

#每次开仓量
OPEN_VOL = 5


class R_Breaker(StrategyBase):
    def __init__(self,*args,**kwargs):
        super(R_Breaker,self).__init__(*args,**kwargs)
        self.__get_param()
        self.__init_data()

    def __get_param(self):
        '''
        获取配置参赛
        :return:
        '''

        self.trade_symbol = self.config.get("para","trade_symbol")
        pos = self.trade_symbol.find(".")

        # 策略的一些阀值
        self.exchange = self.trade_symbol[:pos]
        self.sec_id = self.trade_symbol[pos+1:]
        self.observe_size = self.config.getfloat("para","observe_size")
        self.reversal_size = self.config.getfloat("para","reversal_size")
        self.break_size = self.config.getfloat("para","break_size")

        # 交易开始和结束时间
        FMT = "%sT%s"
        today = arrow.now().date()
        begin_time = self.config.get("para","begin_time")





