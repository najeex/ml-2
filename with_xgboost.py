

import pandas as pd
import numpy as np
import re
import datetime
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb

#load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

unix_cols = ['deadline','state_changed_at','launched_at','created_at']

for x in unix_cols:
    train[x] = train[x].apply(lambda k: datetime.datetime.fromtimestamp(int(k)).strftime('%Y-%m-%d %H:%M:%S'))
    test[x] = test[x].apply(lambda k: datetime.datetime.fromtimestamp(int(k)).strftime('%Y-%m-%d %H:%M:%S'))

cols_to_use = ['name','desc']
len_feats = ['name_len','desc_len']
count_feats = ['name_count','desc_count']

for i in np.arange(2):
    train[len_feats[i]] = train[cols_to_use[i]].apply(str).apply(len)
    test[len_feats[i]] = test[cols_to_use[i]].apply(str).apply(len)

train['name_count'] = train['name'].str.split().str.len()
train['desc_count'] = train['desc'].str.split().str.len()

test['name_count'] = test['name'].str.split().str.len()
test['desc_count'] = test['desc'].str.split().str.len()

train['keywords_len'] = train['keywords'].str.len()
train['keywords_count'] = train['keywords'].str.split('-').str.len()

test['keywords_len'] = test['keywords'].str.len()
test['keywords_count'] = test['keywords'].str.split('-').str.len()

unix_cols = ['deadline','state_changed_at','launched_at','created_at']

for x in unix_cols:
    train[x] = train[x].apply(lambda k: datetime.datetime.strptime(k, '%Y-%m-%d %H:%M:%S'))
    test[x] = test[x].apply(lambda k: datetime.datetime.strptime(k, '%Y-%m-%d %H:%M:%S'))

time1 = []
time3 = []
for i in np.arange(train.shape[0]):
    time1.append(np.round((train.loc[i, 'launched_at'] - train.loc[i, 'created_at']).total_seconds()).astype(int))
    time3.append(np.round((train.loc[i, 'deadline'] - train.loc[i, 'launched_at']).total_seconds()).astype(int))

train['time1'] = np.log(time1)
train['time3'] = np.log(time3)

time5 = []
time6 = []
for i in np.arange(test.shape[0]):
    time5.append(np.round((test.loc[i, 'launched_at'] - test.loc[i, 'created_at']).total_seconds()).astype(int))
    time6.append(np.round((test.loc[i, 'deadline'] - test.loc[i, 'launched_at']).total_seconds()).astype(int))

test['time1'] = np.log(time5)
test['time3'] = np.log(time6)

feat = ['disable_communication','country']

for x in feat:
    le = LabelEncoder()
    le.fit(list(train[x].values) + list(test[x].values))
    train[x] = le.transform(list(train[x]))
    test[x] = le.transform(list(test[x]))

train['goal'] = np.log1p(train['goal'])
test['goal'] = np.log1p(test['goal'])

kickdesc = pd.Series(train['desc'].tolist() + test['desc'].tolist()).astype(str)


def desc_clean(word):
    p1 = re.sub(pattern='(\W+)|(\d+)|(\s+)',repl=' ',string=word)
    p1 = p1.lower()
    return p1

kickdesc = kickdesc.map(desc_clean)

stop = set(stopwords.words('english'))
kickdesc = [[x for x in x.split() if x not in stop] for x in kickdesc]

stemmer = SnowballStemmer(language='english')
kickdesc = [[stemmer.stem(x) for x in x] for x in kickdesc]

kickdesc = [[x for x in x if len(x) > 2] for x in kickdesc]

kickdesc = [' '.join(x) for x in kickdesc]
cv = CountVectorizer(max_features=650)

alldesc = cv.fit_transform(kickdesc).todense()
combine = pd.DataFrame(alldesc)
combine.rename(columns= lambda x: 'variable_'+ str(x), inplace=True)

train_text = combine[:train.shape[0]]
test_text = combine[train.shape[0]:]

test_text.reset_index(drop=True,inplace=True)

cols_to_use = ['name_len','desc_len','keywords_len','name_count','desc_count','keywords_count','time1','time3','goal']

target = train['final_status']

train = train.loc[:,cols_to_use]
test = test.loc[:,cols_to_use]

X_train = pd.concat([train, train_text],axis=1)
X_test = pd.concat([test, test_text],axis=1)

print X_train.shape
print X_test.shape

dtrain = xgb.DMatrix(data=X_train, label = target)
dtest = xgb.DMatrix(data=X_test)


params = {
    'objective':'binary:logistic',
    'eval_metric':'error',
    'eta':0.025,
    'max_depth':6,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':5
    
}
bst = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=40,nfold=5L,verbose_eval=10)

bst_train = xgb.train(params, dtrain, num_boost_round=1000)

p_test = bst_train.predict(dtest)


sub = pd.DataFrame()
sub['project_id'] = test['project_id']
sub['final_status'] = p_test

sub['final_status'] = [1 if x > 0.5 else 0 for x in sub['final_status']]

sub.to_csv("with_xgboost.csv",index=False) #0.70






