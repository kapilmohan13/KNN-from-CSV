import pandas as pd
# import talib
import pandas as pd
from stockstats import wrap



def main():
    # # Load CSV file into a DataFrame
    # df = pd.read_csv('..//data//new.csv')
    #
    # # Display the first few rows of the DataFrame
    # print(df)

    print("Reading input..", end=" ")
    data = pd.read_csv('..//data//new.csv')
    print("done.")
    df = wrap(data)
    print("Calculating indicators..", end=" ")
    rsi = df.get('rsi_6')
    data['rsi_6'] = rsi.values
    macd = df.get('macd')
    data['macd'] = macd.values
    close_16_sma = df.get('close_16_sma')
    data['close_16_sma'] = close_16_sma.values
    stochrsi = df.get('stochrsi')
    data['stochrsi'] = stochrsi.values
    close_10_roc = df.get('close_10_roc')
    data['close_10_roc'] = close_10_roc.values
    close_10_mad = df.get('close_10_mad')
    data['close_10_mad'] = close_10_mad.values

    #Williams overbought/oversold index
    wr = df.get('wr')
    data['wr'] = wr.values

    #VR - Volume Variation Index
    vr = df.get('vr')
    data['vr'] = vr.values

    #VR - Volume Variation Index
    vwma = df.get('vwma')
    data['vwma'] = vwma.values

    #Linear Regression Moving Average
    close_10_lrma = df.get('close_10_lrma')
    data['close_10_lrma'] = close_10_lrma.values

    #Relative Vigor Index (RVGI)
    rvgi_5 = df.get('rvgi_5')
    data['rvgi_5'] = rvgi_5.values
    print("done.")

    print("Adding prediction values..", end=" ")
    #add high-low, the next day prediction value
    data['highlowNextDay'] = (data['high'] - data['low']).shift(-1)
    print("done.")

    #write file
    print("Writing indicators.csv..", end=" ")
    data.to_csv('..//data//indicators.csv', index=True)
    print("done.")

# %%
if __name__ =="__main__":
    main()

# %%