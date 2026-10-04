"""Step 5: evaluation of the champion model on the untouched test seasons (2024-25, 2025-26). Run from project root."""
import pandas as pd, numpy as np, joblib, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
from sklearn.inspection import permutation_importance
F='results/figures'
INK,INK2,GRID,SURF='#0b0b0b','#52514e','#e6e5e0','#fcfcfb'
BLUE,ORANGE,AQUA,MUTED='#2a78d6','#eb6834','#1baf7a','#b9b8b1'
CLS=['Home win','Draw','Away win']; CCOL=[BLUE,ORANGE,AQUA]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':INK2,'xtick.color':INK2,
    'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.dpi':160})
def title(ax,t,sub=None):
    ax.set_title(t,loc='left',fontsize=13,color=INK,fontweight='bold',pad=(34 if sub and '\n' in sub else 22) if sub else 10)
    if sub: ax.text(0,1.02,sub,transform=ax.transAxes,color=INK2,fontsize=9.5)

f=pd.read_csv('data/processed/model_features.csv',parse_dates=['MatchDate'])
test=f[f.Season.isin(['2024-25','2025-26'])].reset_index(drop=True).copy()
b=joblib.load('models/best_model_dev_only.pkl'); m,cols=b['model'],b['features']
P=m.predict_proba(test[cols]); pred=P.argmax(1); y=test.Result.values
test['pred']=pred; test['conf']=P.max(1)
for i,c in enumerate(['pH','pD','pA']): test[c]=P[:,i]

# 1. model comparison
r=pd.read_csv('results/model_comparison.csv').sort_values('test_accuracy')
fig,ax=plt.subplots(figsize=(8,4.2))
col=[BLUE if 'Random Forest' in n else MUTED for n in r.model]
ax.barh(r.model,r.test_accuracy*100,color=col,height=0.6)
for i,v in enumerate(r.test_accuracy*100): ax.text(v+0.4,i,f'{v:.1f}%',va='center',color=INK,fontsize=9)
ax.set_xlim(0,60); ax.xaxis.grid(True,color=GRID); ax.set_axisbelow(True); ax.set_xlabel('Test accuracy (%) — 760 matches, 2024-25 & 2025-26')
title(ax,'All models beat the baseline','Champion (Random Forest) in blue; bookmakers shown for reference')
plt.tight_layout(); plt.savefig(f'{F}/1_model_comparison.png'); plt.close()

# 2. confusion matrix (row-normalised)
cm=confusion_matrix(y,pred,labels=[0,1,2]); cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(5.6,4.8))
ax.imshow(cmn,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b',['#f1f6fd',BLUE,'#0d3a73']),vmin=0,vmax=1)
for i in range(3):
    for j in range(3): ax.text(j,i,f'{cmn[i,j]*100:.0f}%\n({cm[i,j]})',ha='center',va='center',fontsize=10,color='white' if cmn[i,j]>0.5 else INK)
ax.set_xticks(range(3),CLS); ax.set_yticks(range(3),CLS); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
for s in ax.spines.values(): s.set_visible(False)
title(ax,'Confusion matrix (test seasons)','Row % = share of each actual outcome.\nThe model never predicts a draw.')
plt.tight_layout(); plt.savefig(f'{F}/2_confusion_matrix.png'); plt.close()

# 3. permutation importance (how much worse log loss gets when a feature is shuffled)
pi=permutation_importance(m,test[cols],y,scoring='neg_log_loss',n_repeats=20,random_state=42,n_jobs=2)
imp=pd.DataFrame({'feature':cols,'importance':pi.importances_mean,'std':pi.importances_std}).sort_values('importance',ascending=False)
imp.to_csv('results/feature_importance.csv',index=False)
top=imp.head(12).iloc[::-1]
fig,ax=plt.subplots(figsize=(8,4.8))
ax.barh(top.feature,top.importance*1000,xerr=top['std']*1000,color=BLUE,height=0.6,error_kw=dict(ecolor=INK2,lw=1,capsize=2))
ax.xaxis.grid(True,color=GRID); ax.set_axisbelow(True); ax.axvline(0,color=INK2,lw=1)
ax.set_xlabel('Increase in test log loss when the feature is shuffled (×1000)')
title(ax,'Elo difference matters most','Permutation importance, top 12 features, 20 shuffles each')
plt.tight_layout(); plt.savefig(f'{F}/3_feature_importance.png'); plt.close()

# 4. calibration: when the model says X%, does it happen X% of the time?
fig,ax=plt.subplots(figsize=(6,5))
ax.plot([0,1],[0,1],ls='--',color=MUTED,lw=1.5,label='Perfect calibration')
cal=[]
for k,(c,lab,colr) in enumerate(zip(['pH','pD','pA'],CLS,CCOL)):
    bins=pd.qcut(test[c],8,duplicates='drop'); g=test.groupby(bins,observed=True).agg(p=(c,'mean'),hit=('Result',lambda s:(s==k).mean()),n=(c,'size'))
    ax.plot(g.p,g.hit,marker='o',ms=6,lw=2,color=colr,label=lab,markeredgecolor=SURF,markeredgewidth=1.5)
    cal.append(g.assign(outcome=lab))
pd.concat(cal).to_csv('results/calibration_table.csv')
ax.set_xlim(0,0.9); ax.set_ylim(0,0.9); ax.set_xlabel('Predicted probability'); ax.set_ylabel('Actual frequency')
ax.grid(True,color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False,loc='upper left')
title(ax,'Calibration','Points near the dashed line = trustworthy probabilities')
plt.tight_layout(); plt.savefig(f'{F}/4_calibration.png'); plt.close()

# 5. accuracy vs confidence
test['band']=pd.cut(test.conf,[0,0.4,0.5,0.6,0.7,1],labels=['<40%','40-50%','50-60%','60-70%','70%+'])
cb=test.groupby('band',observed=True).apply(lambda d:pd.Series({'accuracy':(d.pred==d.Result).mean(),'matches':len(d),'avg_conf':d.conf.mean()}))
cb.to_csv('results/accuracy_by_confidence.csv')
fig,ax=plt.subplots(figsize=(7,4.2))
ax.bar(cb.index.astype(str),cb.accuracy*100,color=BLUE,width=0.6)
for i,(a,n) in enumerate(zip(cb.accuracy*100,cb.matches)): ax.text(i,a+1.5,f'{a:.0f}%\n{int(n)} matches',ha='center',fontsize=9,color=INK)
ax.set_ylim(0,100); ax.yaxis.grid(True,color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("Model's confidence in its pick"); ax.set_ylabel('Accuracy (%)')
title(ax,'The more confident the model, the more often it is right','Test seasons 2024-25 & 2025-26')
plt.tight_layout(); plt.savefig(f'{F}/5_accuracy_by_confidence.png'); plt.close()

# 6. EDA: outcome share per season
d=f[f.Season!='2026-27'].groupby('Season').Result.value_counts(normalize=True).unstack()*100
fig,ax=plt.subplots(figsize=(8,4.2))
for k,(lab,colr) in enumerate(zip(CLS,CCOL)):
    ax.plot(d.index,d[k],marker='o',ms=6,lw=2,color=colr,markeredgecolor=SURF,markeredgewidth=1.5)
    ax.text(len(d)-0.8,d[k].iloc[-1]+(-1.8 if k==1 else 1.8 if k==2 else 0),lab,color=INK,va='center',fontsize=9)
ax.set_ylim(0,60); ax.yaxis.grid(True,color=GRID); ax.set_axisbelow(True); ax.set_ylabel('% of matches'); plt.xticks(rotation=45)
ax.set_xlim(-0.5,len(d)+0.6)
title(ax,'Home advantage shrank in 2020-21 (no crowds)','Share of results per season, 2017-18 to 2025-26')
plt.tight_layout(); plt.savefig(f'{F}/6_results_by_season.png'); plt.close()

# text report
rep=classification_report(y,pred,labels=[0,1,2],target_names=CLS,zero_division=0)
s=f"""EVALUATION — champion: Random Forest (compact features), test = 2024-25 + 2025-26 ({len(test)} matches)
Accuracy {accuracy_score(y,pred):.3f} | Log loss {log_loss(y,P,labels=[0,1,2]):.4f}
Bookmakers: accuracy {(test[['ImpHome','ImpDraw','ImpAway']].values.argmax(1)==y).mean():.3f}, log loss {log_loss(y,test[['ImpHome','ImpDraw','ImpAway']].values):.4f}
Predicted outcome counts: {dict(zip(CLS,np.bincount(pred,minlength=3).tolist()))}  | Actual: {dict(zip(CLS,np.bincount(y,minlength=3).tolist()))}

{rep}
Accuracy by confidence band:
{cb.round(3).to_string()}

Top 10 features (permutation importance, log loss):
{imp.head(10).round(5).to_string(index=False)}
"""
open('results/evaluation_report.txt','w').write(s); print(s)
