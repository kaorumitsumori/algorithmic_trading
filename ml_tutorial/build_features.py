from pandas import read_pickle
import talib



def calc_talib_features(df):
    op = df['op']
    hi = df['hi']
    lo = df['lo']
    cl = df['cl']
    vl = df['vl']
    hl = (hi + lo) / 2
    df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = \
        talib.BBANDS(cl, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BBANDS_upperband'] -= hl
    df['BBANDS_middleband'] -= hl
    df['BBANDS_lowerband'] -= hl
    df['DEMA'] = talib.DEMA(cl, timeperiod=30) - hl
    df['EMA'] = talib.EMA(cl, timeperiod=30) - hl
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(cl) - hl
    df['KAMA'] = talib.KAMA(cl, timeperiod=30) - hl
    df['MA'] = talib.MA(cl, timeperiod=30, matype=0) - hl
    df['MIDPOINT'] = talib.MIDPOINT(cl, timeperiod=14) - hl
    df['SMA'] = talib.SMA(cl, timeperiod=30) - hl
    df['T3'] = talib.T3(cl, timeperiod=5, vfactor=0) - hl
    df['TEMA'] = talib.TEMA(cl, timeperiod=30) - hl
    df['TRIMA'] = talib.TRIMA(cl, timeperiod=30) - hl
    df['WMA'] = talib.WMA(cl, timeperiod=30) - hl
    df['ADX'] = talib.ADX(hi, lo, cl, timeperiod=14)
    df['ADXR'] = talib.ADXR(hi, lo, cl, timeperiod=14)
    df['APO'] = talib.APO(cl, fastperiod=12, slowperiod=26, matype=0)
    df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(
        hi, lo, timeperiod=14)
    df['AROONOSC'] = talib.AROONOSC(hi, lo, timeperiod=14)
    df['BOP'] = talib.BOP(op, hi, lo, cl)
    df['CCI'] = talib.CCI(hi, lo, cl, timeperiod=14)
    df['DX'] = talib.DX(hi, lo, cl, timeperiod=14)
    df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(
        cl, fastperiod=12, slowperiod=26, signalperiod=9)
    # skip MACDEXT MACDFIX たぶん同じなので
    df['MFI'] = talib.MFI(hi, lo, cl, vl, timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(hi, lo, cl, timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(hi, lo, timeperiod=14)
    df['MOM'] = talib.MOM(cl, timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(hi, lo, cl, timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(hi, lo, timeperiod=14)
    df['RSI'] = talib.RSI(cl, timeperiod=14)
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(
        hi, lo, cl, fastk_period=5, slowk_period=3,
        slowk_matype=0, slowd_period=3, slowd_matype=0
    )
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(
        hi, lo, cl, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(
        cl, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['TRIX'] = talib.TRIX(cl, timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(
        hi, lo, cl, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = talib.WILLR(hi, lo, cl, timeperiod=14)
    df['AD'] = talib.AD(hi, lo, cl, vl)
    df['ADOSC'] = talib.ADOSC(
        hi, lo, cl, vl, fastperiod=3, slowperiod=10)
    df['OBV'] = talib.OBV(cl, vl)
    df['ATR'] = talib.ATR(hi, lo, cl, timeperiod=14)
    df['NATR'] = talib.NATR(hi, lo, cl, timeperiod=14)
    df['TRANGE'] = talib.TRANGE(hi, lo, cl)
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(cl)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(cl)
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(
        cl)
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(cl)
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(cl)
    df['BETA'] = talib.BETA(hi, lo, timeperiod=5)
    df['CORREL'] = talib.CORREL(hi, lo, timeperiod=30)
    df['LINEARREG'] = talib.LINEARREG(cl, timeperiod=14) - cl
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(cl, timeperiod=14)
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(
        cl, timeperiod=14) - cl
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(cl, timeperiod=14)
    df['STDDEV'] = talib.STDDEV(cl, timeperiod=5, nbdev=1)
    return df


def main():
    dataset = read_pickle('dataset.pkl')
    features = calc_talib_features(dataset)
    features.to_pickle('features.pkl')


if __name__ == '__main__':
    main()