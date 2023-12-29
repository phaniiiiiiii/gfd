# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:32:54 2023

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

weather = pd.read_csv("DailyDelhiClimateTrain.csv",index_col="date")
weather.index = pd.to_datetime(weather.index,format="%d/%m/%Y",dayfirst=True)

null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
valid_columns = weather.columns[null_pct < .05] # find the column that has null value < 5%

weather = weather[valid_columns].copy()

weather = weather.query('meanpressure > 980').copy()
weather = weather.query('meanpressure < 1200').copy()
weather = weather.query('wind_speed < 30').copy()

weather["target_0"] = weather.shift(-1)["meantemp"]
weather["target_1"] = weather.shift(-1)["humidity"]
weather["target_2"] = weather.shift(-1)["wind_speed"]
weather["target_3"] = weather.shift(-1)["meanpressure"]
weather = weather.ffill()

def compute_rolling(weather, num, col):# calculate past avg of a few day
    label = f"rolling_{num}_{col}"
    weather[label] = weather[col].rolling(num, min_periods=1).mean()
    return weather
    
def create_features(weather):# improve the mean squared error, decrease the range of forecast
    weather['quarter'] = weather.index.quarter
    weather['month'] = weather.index.month
    weather['year'] = weather.index.year

    cols=["meantemp", "meanpressure", "wind_speed", "humidity" ]

    rolling_nums = [7, 14]
    for num in rolling_nums:
        for col in cols:
            weather = compute_rolling(weather, num, col)
        
    weather.fillna(0)
    
    return weather

def add_lag(weather):
    weather['quarter'] = weather.index.quarter
    weather['month'] = weather.index.month
    weather['year'] = weather.index.year
    
    target_map_0 = weather['target_0'].to_dict()
    weather['lag_01'] = (weather.index - pd.Timedelta('364 days')).map(target_map_0)
    weather['lag_02'] = (weather.index - pd.Timedelta('728 days')).map(target_map_0)

    target_map_1 = weather['target_1'].to_dict()
    weather['lag_11'] = (weather.index - pd.Timedelta('364 days')).map(target_map_1)
    weather['lag_12'] = (weather.index - pd.Timedelta('728 days')).map(target_map_1)

    target_map_2 = weather['target_2'].to_dict()
    weather['lag_21'] = (weather.index - pd.Timedelta('364 days')).map(target_map_2)
    weather['lag_22'] = (weather.index - pd.Timedelta('728 days')).map(target_map_2)

    target_map_3 = weather['target_3'].to_dict()
    weather['lag_31'] = (weather.index - pd.Timedelta('364 days')).map(target_map_3)
    weather['lag_32'] = (weather.index - pd.Timedelta('728 days')).map(target_map_3)
    return weather

def rdmforest(weather,predictors,target, n):
    all_predictions = []

    train = weather.iloc[:-n]
    test = weather.iloc[-n:]
     
    rf = RandomForestRegressor(n_estimators = 1000, criterion = 'squared_error', max_depth = 8, 
                               min_samples_split = 2, min_samples_leaf = 1)
    
    rf.fit(train[predictors], train[target])
    
    preds = rf.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    combined = pd.concat([test[target], preds], axis=1)
    combined.columns = ["actual", "prediction"]
    combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
    all_predictions.append(combined)
    
    # # Get feature importances
    # importances = list(rf.feature_importances_)
    # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(predictors, importances)]
    # feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    return pd.concat(all_predictions)

def rid(weather, predictors,target, n):
    all_predictions = []
    train = weather.iloc[:-n]
    test = weather.iloc[-n:]

    rr = Ridge(alpha=0.1) 
        
    rr.fit(train[predictors], train[target])
        
    preds = rr.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    combined = pd.concat([test[target], preds], axis=1)
    combined.columns = ["actual", "prediction"]
    combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        
    all_predictions.append(combined)
    return pd.concat(all_predictions)

def xg(weather, predictors,target, n):
    all_predictions = []
    
    train = weather.iloc[:-n]
    test = weather.iloc[-n:]

    reg = xgb.XGBRegressor(tree_method="hist",    
                           n_estimators=1000,
                           objective='reg:squarederror',
                           max_depth=4,
                           learning_rate=0.01)
    reg.fit(train[predictors], train[target],
            eval_set=[(train[predictors], train[target])],
            verbose=None)
    
    preds = reg.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    combined = pd.concat([test[target], preds], axis=1)
    combined.columns = ["actual", "prediction"]
    combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        
    all_predictions.append(combined)

    # # Get feature importances
    # importances = list(reg.feature_importances_)
    # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(predictors, importances)]
    # feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    return pd.concat(all_predictions)
print("New Delhi weather forecast")
print('Choose your model (empty for Ridge Regression):\nRidge Regression\nRandom Forest\nXGBoost')
model=input('Paste your choice here:')
print()
if model == '' or model == 'Ridge Regression':
    weather = create_features(weather)
    predictors = weather.columns[~weather.columns.isin(["target_0","target_1","target_2","target_3"])]
    target = "target_0"
    predictions = rid(weather, predictors,target, 100)

    score = mean_absolute_error(predictions['actual'], predictions['prediction'])
    print(f'Mean error: {score:0.2f}')
    score_2 = mean_squared_error(predictions['actual'], predictions['prediction'])
    print(f'Mean squared error: {score_2:0.2f}')
    print('Accuracy example by testing on meantemp')
    ax = predictions[["actual"]].plot(figsize=(15, 5))
    predictions["prediction"].plot(ax=ax)
    predictions["diff"].plot(ax=ax)
    plt.legend(['actual', 'prediction','diff'])
    ax.set_title('Ridge Regression')
    plt.show()
    # use xgboost, random forrest,... to improve the model


    print("(This model is only accurate enough if the date today is less\nthan 7 days after data end, which is at 24/04/2017 in our case)")
    print()
    print("(emtpy for the end day of the data:24-04-2017)")
    date = input("today date(dd-mm-yy):")
    print()
    if date == '':
        prediction_0 = rid(weather, predictors,"target_0", 1)
        prediction_1 = rid(weather, predictors,"target_1", 1)
        prediction_2 = rid(weather, predictors,"target_2", 1)
        prediction_3 = rid(weather, predictors,"target_3", 1)
        print("predictions for 25-04-2017:")
        print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
        print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
        print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
        print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))
    else:
        date = pd.to_datetime(date,format="%d-%m-%Y",dayfirst=True)
        end_date = weather.index.max()
        if date-end_date > pd.Timedelta("7d"):
            print("This date is too far pass the last date of this data\nThe data needs updating to be up-to-date")
        elif date-end_date == pd.Timedelta("0"):
            prediction_0 = rid(weather, predictors,"target_0", 1)
            prediction_1 = rid(weather, predictors,"target_1", 1)
            prediction_2 = rid(weather, predictors,"target_2", 1)
            prediction_3 = rid(weather, predictors,"target_3", 1)
            print("predictions for 25-04-2017:")
            print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
            print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
            print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
            print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))
            
        else:
            future = pd.date_range(date,date, freq='1d')
            future_df = pd.DataFrame(index=future)
            df_and_future = pd.concat([weather, future_df])

            df_and_future["meantemp"].iloc[-1]=float(input("today mean temperature:"))
            df_and_future["humidity"].iloc[-1]=float(input("today humidity:"))
            df_and_future["wind_speed"].iloc[-1]=float(input("today wind speed:"))
            df_and_future["meanpressure"].iloc[-1]=float(input("today meanpressure:"))
            df_and_future = create_features(df_and_future)
            print()
            prediction_0 = rid(df_and_future, predictors,"target_0", 1)
            prediction_1 = rid(df_and_future, predictors,"target_1", 1)
            prediction_2 = rid(df_and_future, predictors,"target_2", 1)
            prediction_3 = rid(df_and_future, predictors,"target_3", 1)
            b = date + pd.Timedelta("1d")
            b = b.date()
            b = b.strftime("%d-%m-%Y")
            print("predictions for "+str(b)+":")
            print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
            print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
            print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
            print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))
            
elif model == "Random Forest":
    weather = create_features(weather)
    predictors = weather.columns[~weather.columns.isin(["target_0","target_1","target_2","target_3"])]
    target = "target_0"

    td = rdmforest(weather,predictors,target, 100)
    #mse
    score = mean_absolute_error(td['actual'], td['prediction'])
    print(f'Mean error: {score:0.2f}')
    score_2 = mean_squared_error(td['actual'], td['prediction'])
    print(f'Mean squared error: {score_2:0.2f}')
    print('Accuracy example by testing on meantemp')
    ax = td[['actual']].plot(figsize=(15, 5))
    td['prediction'].plot(ax=ax)
    td["diff"].plot(ax=ax)
    plt.legend(['actual', 'prediction','diff'])
    ax.set_title('Random Forest')
    plt.show()

    print("(This model is only accurate enough if the date today is less\nthan 7 days after data end, which is at 24/04/2017 in our case)")
    print()
    print("(emtpy for the end day of the data:24-04-2017)")
    date = input("today date(dd-mm-yy):")
    print()
    if date == '':
        prediction_0 = rdmforest(weather, predictors,"target_0", 1)
        prediction_1 = rdmforest(weather, predictors,"target_1", 1)
        prediction_2 = rdmforest(weather, predictors,"target_2", 1)
        prediction_3 = rdmforest(weather, predictors,"target_3", 1)
        print("predictions for 25-04-2017:")
        print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
        print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
        print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
        print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))
    else:
        date = pd.to_datetime(date,format="%d-%m-%Y",dayfirst=True)
        end_date = weather.index.max()
        if date-end_date > pd.Timedelta("7d"):
            print("This date is too far pass the last date of this data\nThe data needs updating to be up-to-date")
        elif date-end_date == pd.Timedelta("0"):
            prediction_0 = rdmforest(weather, predictors,"target_0", 1)
            prediction_1 = rdmforest(weather, predictors,"target_1", 1)
            prediction_2 = rdmforest(weather, predictors,"target_2", 1)
            prediction_3 = rdmforest(weather, predictors,"target_3", 1)
            print("predictions for 25-04-2017:")
            print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
            print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
            print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
            print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))
            
        else:
            future = pd.date_range(date,date, freq='1d')
            future_df = pd.DataFrame(index=future)
            df_and_future = pd.concat([weather, future_df])

            df_and_future["meantemp"].iloc[-1]=float(input("today mean temperature:"))
            df_and_future["humidity"].iloc[-1]=float(input("today humidity:"))
            df_and_future["wind_speed"].iloc[-1]=float(input("today wind speed:"))
            df_and_future["meanpressure"].iloc[-1]=float(input("today meanpressure:"))
            df_and_future = create_features(df_and_future)
            print()
            prediction_0 = rdmforest(df_and_future, predictors,"target_0", 1)
            prediction_1 = rdmforest(df_and_future, predictors,"target_1", 1)
            prediction_2 = rdmforest(df_and_future, predictors,"target_2", 1)
            prediction_3 = rdmforest(df_and_future, predictors,"target_3", 1)
            b = date + pd.Timedelta("1d")
            b = b.date()
            b = b.strftime("%d-%m-%Y")
            print("predictions for "+str(b)+":")
            print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
            print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
            print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
            print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))
            
elif model == "XGBoost":
    weather = add_lag(weather)
    predictors = weather.columns[~weather.columns.isin(["target_0","target_1","target_2","target_3"])]
    target = "target_0"

    td = xg(weather, predictors, target, 100)

    score = mean_absolute_error(td['actual'], td['prediction'])
    print(f'Mean error: {score:0.2f}')
    score_2 = mean_squared_error(td['actual'], td['prediction'])
    print(f'Mean squared error: {score_2:0.2f}')
    print('Accuracy example by testing on meantemp')
    ax = td[['actual']].plot(figsize=(15, 5))
    td['prediction'].plot(ax=ax)
    td["diff"].plot(ax=ax)
    plt.legend(['actual', 'prediction','diff'])
    ax.set_title('XGboost')
    plt.show()
    
    print("(This model is only accurate enough if the date today is less\nthan 7 days after data end, which is at 24/04/2017 in our case)")
    print()
    print("(emtpy for the end day of the data:24-04-2017)")
    date = input("today date(dd-mm-yy):")
    print()
    if date == '':
        prediction_0 = xg(weather, predictors,"target_0", 1)
        prediction_1 = xg(weather, predictors,"target_1", 1)
        prediction_2 = xg(weather, predictors,"target_2", 1)
        prediction_3 = xg(weather, predictors,"target_3", 1)
        print("predictions for 25-04-2017:")
        print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
        print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
        print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
        print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))
    else:
        date = pd.to_datetime(date,format="%d-%m-%Y",dayfirst=True)
        end_date = weather.index.max()
        if date-end_date > pd.Timedelta("7d"):
            print("This date is too far pass the last date of this data\nThe data needs updating to be up-to-date")
        elif date-end_date == pd.Timedelta("0"):
            prediction_0 = xg(weather, predictors,"target_0", 1)
            prediction_1 = xg(weather, predictors,"target_1", 1)
            prediction_2 = xg(weather, predictors,"target_2", 1)
            prediction_3 = xg(weather, predictors,"target_3", 1)
            print("predictions for 25-04-2017:")
            print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
            print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
            print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
            print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))
            
        else:
            future = pd.date_range(date,date, freq='1d')
            future_df = pd.DataFrame(index=future)
            df_and_future = pd.concat([weather, future_df])

            df_and_future["meantemp"].iloc[-1]=float(input("today mean temperature:"))
            df_and_future["humidity"].iloc[-1]=float(input("today humidity:"))
            df_and_future["wind_speed"].iloc[-1]=float(input("today wind speed:"))
            df_and_future["meanpressure"].iloc[-1]=float(input("today meanpressure:"))
            df_and_future = add_lag(df_and_future)
            print()
            prediction_0 = xg(df_and_future, predictors,"target_0", 1)
            prediction_1 = xg(df_and_future, predictors,"target_1", 1)
            prediction_2 = xg(df_and_future, predictors,"target_2", 1)
            prediction_3 = xg(df_and_future, predictors,"target_3", 1)
            b = date + pd.Timedelta("1d")
            b = b.date()
            b = b.strftime("%d-%m-%Y")
            print("predictions for "+str(b)+":")
            print("mean temperature:",round(float(prediction_0["prediction"].iloc[-1]),2))
            print("humidity:",round(float(prediction_1["prediction"].iloc[-1]),2))
            print("wind speed:",round(float(prediction_2["prediction"].iloc[-1]),2))
            print("mean pressure:",round(float(prediction_3["prediction"].iloc[-1]),2))