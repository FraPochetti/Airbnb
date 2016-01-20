import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse as sp

def make_user_features(train, test):
    #encoding country destinations in train set
    outcome = train.country_destination
    labels = outcome.values
    le = LabelEncoder()
    y = le.fit_transform(labels)
    train = train.drop(['country_destination'], axis=1)

    #storing user ids in test set
    id_test = test['id']

    #appending test to train and dropping date first booking which is redundant
    data = pd.concat((train, test), axis=0, ignore_index=True)
    data = data.drop(['date_first_booking'], axis=1)

    #extracting features from date_account_created
    data['dac_year'] = data.date_account_created.apply(lambda x: x.year)
    data['dac_month'] = data.date_account_created.apply(lambda x: x.month)
    data['dac_weekday'] = data.date_account_created.apply(lambda x: x.weekday())
    data = data.drop(['date_account_created'], axis=1)

    #extracting features from timestamp_first_active
    data['tfa_year'] = data.timestamp_first_active.apply(lambda x: x.year)
    data['tfa_month'] = data.timestamp_first_active.apply(lambda x: x.month)
    data['tfa_weekday'] = data.timestamp_first_active.apply(lambda x: x.weekday())
    data = data.drop(['timestamp_first_active'], axis=1)

    #filling age nan with age median
    data.age = data.age.fillna(data.age.median())

    #binning age column
    bins = list(np.arange(15, 85, 5))
    bins.insert(0,0)
    bins.append(int(max(data.age)))
    group_names = ['<15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50',
                   '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '>80']
    data['age_bucket'] = pd.cut(data['age'], bins, labels=group_names)

    #cleaning gender column and filling nan in all dataframe with 'unknown'
    data.gender = data.gender.replace('-unknown-','unknown')
    data.ix[:, data.columns != 'age_bucket'] = data.ix[:, data.columns != 'age_bucket'].fillna('unknown')

    #generating dummy variables in top of categorical columns
    to_be_dummified = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser','age_bucket']
    for f in to_be_dummified:
        dummies = pd.get_dummies(data[f], prefix=f)
        data = data.drop([f], axis=1)
        data = pd.concat((data, dummies), axis=1)

    return data[:train.shape[0]], data[train.shape[0]:], y, le

def make_sessions_features(data, df_sessions):
    # Drop row with nan values from the "user_id" column as they're useless
    df_sessions = df_sessions.dropna(subset=["user_id"])

    # Frequency of devices - by user
    device_freq = df_sessions.groupby('user_id').device_type.value_counts()

    # Frequency of actions taken - by user
    action_freq = df_sessions.groupby('user_id').action.value_counts()

    # Total list of users
    users = data.id

    def feature_dict(df):
        f_dict = dict(list(df.groupby(level='user_id')))
        res = {}
        for k, v in f_dict.items():
            v.index = v.index.droplevel('user_id')
            res[k] = v.to_dict()
        return res

    # Make a dictionary with the frequencies { 'user_id' : {"IPhone": 2, "Windows": 1}}
    action_dict = feature_dict(action_freq)
    device_dict = feature_dict(device_freq)

    # Transform to a list of dictionaries
    action_rows = [action_dict.get(k, {}) for k in users]
    device_rows = [device_dict.get(k, {}) for k in users]

    device_transf = DictVectorizer()
    tf = device_transf.fit_transform(device_rows)

    action_transf = DictVectorizer()
    tf2 = action_transf.fit_transform(action_rows)

    # Concatenate the two datasets
    # Those are row vectors with the frequencies of both device and actions [0, 0, 0, 2, 0, 1, ...]
    features = sp.hstack([tf, tf2])

    # We create a dataframe with the new features and we write it to disk
    df_sess_features = pd.DataFrame(features.todense())
    df_sess_features['id'] = users

    #left joining data and sessions on user_id
    final = pd.merge(data, df_sess_features, how='left', left_on='id', right_on='id')
    final.ix[:, final.columns != 'age_bucket'].fillna(-1, inplace=True)

    # Using inplace because I have 8GB of RAM
    # final.ix[:, final.columns != 'age_bucket'] = final.ix[:, final.columns != 'age_bucket'].fillna(-1)

    final.drop(['id'], axis=1, inplace=True)
    return final
