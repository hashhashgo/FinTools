import pandas as pd

name_map = {
    '南华综合指数': 'NHCI',
    '南华工业品指数': 'NHII',
    '南华能化指数': 'NHECI',
    '南华金属指数': 'NHMI',
    '南华农产品指数': 'NHAI',
    '南华贵金属指数': 'NHPMI',
    '南华能源指数': 'NHEI',
    '南华石油化工指数': 'NHPCI',
    '南华煤制化工指数': 'NHCCI',
    '南华有色金属指数': 'NHNFI',
    '南华黑色产业指数': 'NHFI',
    '南华黑色原材料指数': 'NHFMI',
    '南华建材指数': 'NHBMI',
    '南华油脂油料指数': 'NHOOI',
    '南华经济作物指数': 'NHAECI',
    '南华新材料指数': 'NHNMI',
    '南华迷你综合指数': 'NHCIMi',
    '南华风险均衡商品指数': 'NHRECI',
    '南华QFII商品指数': 'NHQFII',
    '南华商品指数': 'NHCI',
    '南华黑色指数': 'NHFI'
}

symbol_map = {
    '黄大豆1号': 'A',
    '黄大豆一号': 'A',
    '黄大豆': 'A',
    '大豆': 'A',
    '豆一': 'A',
    '白银': 'AG',
    '铝': 'AL',
    '氧化铝': 'AO',
    '苹果': 'AP',
    '黄金': 'AU',
    '黄大豆2号': 'B',
    '胶合板': 'BB',
    '国际铜': 'BC',
    '丁二烯橡胶': 'BR',
    '石油沥青': 'BU',
    '玉米': 'C',
    '棉花': 'CF',
    '一号棉': 'CF',
    '红枣': 'CJ',
    '玉米淀粉': 'CS',
    '铜': 'CU',
    '棉纱': 'CY',
    '苯乙烯': 'EB',
    '乙二醇': 'EG',
    '纤维板': 'FB',
    '玻璃': 'FG',
    '燃料油': 'FU',
    '燃油': 'FU',
    '热轧卷板': 'HC',
    '铁矿石': 'I',
    '焦炭': 'J',
    '鸡蛋': 'JD',
    '焦煤': 'JM',
    '粳稻': 'JR',
    '聚乙烯': 'L',
    '塑料': 'L',
    '碳酸锂': 'LC',
    '原木': 'LG',
    '生猪': 'LH',
    '晚籼稻': 'LR',
    '低硫燃料油': 'LU',
    '豆粕': 'M',
    '甲醇': 'MA',
    '镍': 'NI',
    '20号胶': 'NR',
    '菜籽油': 'OI',
    '棕榈油': 'P',
    '铅': 'PB',
    '短纤': 'PF',
    '液化石油气': 'PG',
    '花生': 'PK',
    '普麦': 'PM',
    '聚丙烯': 'PP',
    '瓶片': 'PR',
    '多晶硅': 'PS',
    '对二甲苯': 'PX',
    '螺纹钢': 'RB',
    '螺纹': 'RB',
    '早籼稻': 'RI',
    '菜籽粕': 'RM',
    '菜粕': 'RM',
    '粳米': 'RR',
    '油菜籽': 'RS',
    '天然橡胶': 'RU',
    '橡胶': 'RU',
    '纯碱': 'SA',
    '原油': 'SC',
    '硅铁': 'SF',
    '烧碱': 'SH',
    '工业硅': 'SI',
    '锰硅': 'SM',
    '锡': 'SN',
    '纸浆': 'SP',
    '白糖': 'SR',
    '不锈钢': 'SS',
    'PTA': 'TA',
    '尿素': 'UR',
    '聚氯乙烯': 'V',
    'PVC': 'V',
    '强麦': 'WH',
    '线材': 'WR',
    '豆油': 'Y',
    '动力煤': 'ZC',
    '锌': 'ZN'
}

def convert(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=name_map)
    df = df.dropna(subset=['symbol'])
    df['symbol'] = df['symbol'].map(lambda x: symbol_map.get(x, x))
    df = df.set_index("symbol").stack().rename("weight").reset_index().rename(columns={"level_1": "index"})
    if not df['symbol'].isin(symbol_map.values()).all():
        missing_symbols = df.loc[~df['symbol'].isin(symbol_map.values()), 'symbol'].unique()
        raise ValueError(f"Some symbols are not mapped correctly: {missing_symbols}")
    if not df['index'].isin(name_map.values()).all():
        missing_indices = df.loc[~df['index'].isin(name_map.values()), 'index'].unique()
        raise ValueError(f"Some indices are not mapped correctly: {missing_indices}")
    return df

w2025 = convert(pd.read_excel("nanhua2025.xlsx", skiprows=2, usecols="B:U").rename(columns={'Unnamed: 1': 'symbol'}))
w2025['year'] = 2025

w2024 = convert(pd.read_excel("nanhua2024.xlsx", skiprows=2, usecols="B:T").rename(columns={'Unnamed: 1': 'symbol'}))
w2024['year'] = 2024

w2023 = convert(pd.read_excel("nanhua2023.xlsx", skiprows=2, usecols="B:R").rename(columns={'Unnamed: 1': 'symbol'}))
w2023['year'] = 2023

w2022 = convert(pd.read_excel("nanhua2022.xlsx", skiprows=1).rename(columns={'品种名称': 'symbol'}))
w2022['year'] = 2022

w2021 = convert(pd.read_excel("nanhua2021.xlsx", skiprows=1).rename(columns={'品种名称': 'symbol'}))
w2021['year'] = 2021

w2020 = convert(pd.read_excel("nanhua2020.xlsx", skiprows=1).rename(columns={'品种名称': 'symbol'}))
w2020['year'] = 2020

w2019 = convert(pd.read_excel("nanhua2019.xlsx", skiprows=1).rename(columns={'品种名称': 'symbol'}))
w2019['year'] = 2019

w2018 = convert(pd.read_excel("nanhua2018.xlsx", skiprows=1).rename(columns={'品种名称': 'symbol'}))
w2018['year'] = 2018

w2017 = convert(pd.read_excel("nanhua2017.xlsx", skiprows=1).rename(columns={'品种名称': 'symbol'}))
w2017['year'] = 2017

w2016 = convert(pd.read_excel("nanhua2016.xlsx", skiprows=1).rename(columns={'品种名称': 'symbol'}))
w2016['year'] = 2016

w2015 = convert(pd.read_excel("nanhua2015.xlsx", skiprows=1).rename(columns={'品种名称': 'symbol'}))
w2015['year'] = 2015

w2014 = convert(pd.read_excel("nanhua2014.xlsx", skiprows=1).rename(columns={'品种': 'symbol'}))
w2014['year'] = 2014

w2013 = convert(pd.read_excel("nanhua2013.xlsx", skiprows=1).rename(columns={'品种': 'symbol'}))
w2013['year'] = 2013

w2012 = convert(pd.read_excel("nanhua2012.xlsx", skiprows=1).rename(columns={'品种': 'symbol'}))
w2012['year'] = 2012

w2011 = convert(pd.read_excel("nanhua2011.xlsx", skiprows=1).rename(columns={'品种': 'symbol'}))
w2011['year'] = 2011

wnhci = pd.read_excel("NHCI.xls")
wnhci = wnhci.rename(columns={'时间': 'year'})
wnhci['year'] = wnhci['year'].dt.year
wnhci = wnhci[wnhci['year'] < 2011]
wnhci = wnhci.set_index("year").stack().rename("weight").reset_index().rename(columns={"level_1": "symbol"})
wnhci['symbol'] = wnhci['symbol'].map(lambda x: symbol_map.get(x, x))
wnhci['index'] = 'NHCI'
if not wnhci['symbol'].isin(symbol_map.values()).all():
    missing_symbols = wnhci.loc[~wnhci['symbol'].isin(symbol_map.values()), 'symbol'].unique()
    raise ValueError(f"Some symbols are not mapped correctly: {missing_symbols}")
if not wnhci['index'].isin(name_map.values()).all():
    missing_indices = wnhci.loc[~wnhci['index'].isin(name_map.values()), 'index'].unique()
    raise ValueError(f"Some indices are not mapped correctly: {missing_indices}")

wall = pd.concat([w2025, w2024, w2023, w2022, w2021, w2020, w2019, w2018, w2017, w2016, w2015, w2014, w2013, w2012, w2011, wnhci], ignore_index=True)
wall = wall.sort_values(['year', 'index', 'symbol']).reset_index(drop=True)
wall = wall.drop(wall[wall['weight'] == 0].index)

assert wall.groupby(['year', 'index'])['weight'].sum().sub(1).abs().max() < 1e-6, "Weights do not sum to 1 for some year/index combinations."

s = set()
for year in wall['year'].unique():
    for ss in s:
        if wall[(wall['year'] == year) & (wall['index'] == ss)].empty:
            w_new = wall[(wall['year'] == (year - 1)) & (wall['index'] == ss)].copy()
            assert not w_new.empty, f"Missing weight for symbol {ss} in year {year}, and no previous year data to carry forward."
            w_new['year'] = year
            wall = pd.concat([wall, w_new], ignore_index=True)
    for index in wall[wall['year'] == year]['index'].unique(): s.add(index)

wall.to_csv("nanhua_weights.csv", index=False)