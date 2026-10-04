"""Step 4 (model training): time-based split, tuning with season-expanding CV, final test. Run from project root."""
import pandas as pd, numpy as np, json, joblib, warnings, itertools
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, f1_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
P='data/processed'
f=pd.read_csv(f'{P}/model_features.csv',parse_dates=['MatchDate'])
ALL=pd.read_csv(f'{P}/feature_list.csv').feature.tolist()
COMPACT=list(dict.fromkeys(['HomeElo','AwayElo','Elo_diff','H2H_HomePPG','Home_promoted','Away_promoted']+[c for c in ALL if c.endswith('_diff')]))
FSETS={'all':ALL,'compact':COMPACT}

# ---- time-based split (never random: that would let the model see the future) ----
dev  = f[f.Season<='2023-24'].reset_index(drop=True)             # train + validation seasons (2017-18..2023-24)
test = f[f.Season.isin(['2024-25','2025-26'])].reset_index(drop=True)   # untouched until the very end
live = f[f.Season=='2026-27'].reset_index(drop=True)             # 20 current-season games, kept aside
folds=[(dev.index[dev.Season<v],dev.index[dev.Season==v]) for v in ['2021-22','2022-23','2023-24']]  # expanding window
print(f'dev {len(dev)} | test {len(test)} | live {len(live)} | CV folds validate on 2021-22, 2022-23, 2023-24')

models={
 'Logistic Regression':(lambda **p:make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,**p)),
                        [{'C':c} for c in (0.003,0.01,0.03,0.1,1)]),
 'Random Forest':(lambda **p:RandomForestClassifier(n_estimators=300,n_jobs=2,random_state=42,**p),
                  [{'max_depth':d,'min_samples_leaf':l} for d,l in itertools.product((4,6),(10,25))]),
 'XGBoost':(lambda **p:XGBClassifier(objective='multi:softprob',subsample=0.8,colsample_bytree=0.6,random_state=42,n_jobs=2,eval_metric='mlogloss',**p),
            [{'max_depth':d,'learning_rate':lr,'n_estimators':n} for d,lr,n in itertools.product((2,3),(0.02,0.05),(150,300))]),
}
def cv_score(make,params,cols):
    ll=[];acc=[]
    for tr,va in folds:
        m=make(**params).fit(dev.loc[tr,cols],dev.loc[tr,'Result']); p=m.predict_proba(dev.loc[va,cols])
        ll.append(log_loss(dev.loc[va,'Result'],p,labels=[0,1,2])); acc.append(accuracy_score(dev.loc[va,'Result'],p.argmax(1)))
    return np.mean(ll),np.mean(acc)

rows=[];best={}
base=DummyClassifier(strategy='prior').fit(dev[ALL],dev.Result)
for name,(make,grid) in models.items():
    res=[]
    for fs,cols in FSETS.items():
        for params in grid:
            ll,acc=cv_score(make,params,cols); res.append((ll,acc,fs,params))
    res.sort(key=lambda r:r[0]); ll,acc,fs,params=res[0]
    best[name]=dict(feature_set=fs,params=params,cv_logloss=ll,cv_acc=acc)
    print(f'{name:20s} best CV logloss {ll:.4f} acc {acc:.3f} | features={fs} params={params}')

# ---- final: refit each tuned model on ALL dev seasons, evaluate once on the test seasons ----
X_t,y_t=test,test.Result; out=[]
def row(name,p,extra={}):
    return dict(model=name,test_accuracy=accuracy_score(y_t,p.argmax(1)),test_logloss=log_loss(y_t,p,labels=[0,1,2]),
                test_macro_f1=f1_score(y_t,p.argmax(1),average='macro'),**extra)
out.append(row('Baseline: prior (always Home win)',base.predict_proba(X_t[ALL])))
fitted={}
for name,(make,_) in models.items():
    b=best[name]; cols=FSETS[b['feature_set']]
    m=make(**b['params']).fit(dev[cols],dev.Result); fitted[name]=(m,cols)
    out.append(row(name,m.predict_proba(X_t[cols]),dict(features=b['feature_set'],cv_logloss=b['cv_logloss'])))
# simple Elo-only logistic regression: the "is all this feature work worth it?" check
m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(dev[['Elo_diff']],dev.Result)
out.append(row('Elo-difference only (LogReg)',m.predict_proba(X_t[['Elo_diff']]),dict(features='Elo_diff')))
# simple average of the three tuned models
avg=np.mean([fitted[n][0].predict_proba(X_t[fitted[n][1]]) for n in models],axis=0); out.append(row('Ensemble (average of 3)',avg))
bk=X_t[['ImpHome','ImpDraw','ImpAway']].values; out.append(row('Bookmakers (reference)',bk))
res=pd.DataFrame(out).round(4).sort_values('test_logloss'); print('\n',res.to_string(index=False))
res.to_csv('results/model_comparison.csv',index=False)

# ---- save the best real model (lowest test... chosen by CV log loss, NOT by test) ----
champ=min(best,key=lambda n:best[n]['cv_logloss']); m,cols=fitted[champ]
# refit on dev + test for deployment (more data = better live predictions); the evaluation above used the dev-only fit
final=models[champ][0](**best[champ]['params']).fit(pd.concat([dev,test])[cols],pd.concat([dev,test]).Result)
joblib.dump(dict(model=final,features=cols,classes=['Home win','Draw','Away win']),'models/best_model.pkl')
joblib.dump(dict(model=m,features=cols,classes=['Home win','Draw','Away win']),'models/best_model_dev_only.pkl')
json.dump(dict(champion=champ,**best),open('results/best_params.json','w'),indent=2,default=float)
print('\nChampion (chosen by CV, before seeing test):',champ)
