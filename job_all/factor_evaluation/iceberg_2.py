import ccxt
import time
import os
import asyncio
good_coin = ['BTC', 'ETH', 'XRP', 'BCH', 'EOS', 'XLM', 'LTC', 'ADA', 'XMR', 'TRX', 'BNB', 'ONT', 'NEO', 'DCR']
# good_coin = ['BTC', 'ETH', 'XRP']
good_exchange_name = ['binance', 'fcoin', 'gateio', 'huobipro', 'kucoin', 'okex','bcex','bibox','bigone','bitfinex','bitforex',
                      'bithumb','bitkk','cex','coinbase','coinex','cointiger','exx','gdax','gemini','hitbtc','rightbtc',
                      'theocean','uex']
# good_exchange_name = ['huobipro', 'kucoin', 'okex']

def_quote = 'USDT'
# delay 2 seconds
delay = 2

all_exchange = ccxt.exchanges
gate = ccxt.gateio()
print(gate)
gate_markets = gate.load_markets()
print('gateio markets is {}'.format(gate.markets))


# test demo
def test_demo():
    # from variable id
    exchange_id = 'okcoincny'
    # os.environ.setdefault('http_proxy', 'http://127.0.0.1:1080')
    # os.environ.setdefault('https_proxy', 'http://127.0.0.1:1080')
    # print('http_proxy is {},https_proxy is {}'.format(os.environ.get('http_proxy'), os.environ.get('https_proxy')))
    binance = getattr(ccxt, exchange_id)()
    binance_markets = binance.load_markets()
    print ('gateio markets is {}'.format(binance_markets))
    # print ('gateio all tickers is {}'.format(gate.fetch_tickers()))
    # print ('BTC/USDT is {},\n ETH/USDT is {},\n ETH/BTC is {}'.format(gate_markets['BTC/USDT'],
    # gate_markets['ETH/USDT'],gate_markets['ETH/BTC']))
    # print('exchange gate is {}'.format(dir(gate)))
    # print(gate_markets)
    exit()
    for symbol in gate.markets:
        # print(symbol)
        base = str(symbol).split('/')[0]
        quote = str(symbol).split('/')[1]
        if base in good_coin and quote == def_quote:
            orderbook = gate.fetch_order_book(symbol)
            bid1 = orderbook['bids'][0][0] if len (orderbook['bids']) > 0 else None
            ask1 = orderbook['asks'][0][0] if len (orderbook['asks']) > 0 else None
            spread = (ask1 - bid1) if (bid1 and ask1) else None
            print('symbol is {},bid1 is {},ask1 is {},spread is {}'.format(symbol,bid1,ask1,spread))
        # print ('symbol is {},\n order_book is {}'.format(symbol,gate.fetch_order_book (symbol)))
        time.sleep(delay)  # rate limit

test_demo()












