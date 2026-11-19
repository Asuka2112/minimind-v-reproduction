"""Postprocessing only: visualize the frozen-protocol metrics without rescoring."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path('/workspace/projects/minimind-v-independent-eval-20261103')
x=json.load(open(r/'summary.json'))
jobs=x['metrics']['full']['jobs']
models=['initial','pretrain','sft','official']
labels=['Initial LM','Pretrain','Reproduced SFT','Official VLM']
colors=['#8393a6','#67a4bf','#1c7b59','#c58a47']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,metric,title in [(axes[0,0],'strict','Strict generated-answer accuracy'),(axes[0,1],'restricted','Restricted A-D logit diagnostic')]:
 vals=np.array([jobs[f'{m}_correct'][metric]['accuracy']*100 for m in models])
 ci=np.array([jobs[f'{m}_correct'][metric]['ci95_image_cluster'] for m in models])*100
 ax.bar(labels,vals,color=colors,width=.63)
 ax.errorbar(range(4),vals,yerr=np.maximum(0,np.array([vals-ci[:,0],ci[:,1]-vals])),fmt='none',color='#333333',capsize=4)
 for i,v in enumerate(vals):ax.annotate(f'{v:.2f}%',(i,v),xytext=(0,8),textcoords='offset points',ha='center')
 if metric=='restricted':
  ax.axhline(25,color='#888888',linestyle='--',linewidth=1,label='Uniform random: 25%')
  ax.axhline(29.8,color='#a15555',linestyle=':',linewidth=1.4,label='Always B: 29.80% (descriptive)')
  ax.legend(frameon=False,fontsize=8)
 ax.set_ylim(0,max(35,float(ci.max())+8));ax.set_ylabel('Accuracy (%)');ax.set_title(title,loc='left',weight='bold');ax.tick_params(axis='x',labelrotation=12)
ax=axes[1,0]
conditions=['correct','shuffled','none']
for offset,metric,color in [(-.2,'strict','#1c7b59'),(.2,'restricted','#67a4bf')]:
 vals=np.array([jobs[f'sft_{c}'][metric]['accuracy']*100 for c in conditions]);ci=np.array([jobs[f'sft_{c}'][metric]['ci95_image_cluster'] for c in conditions])*100
 positions=np.arange(3)+offset
 ax.bar(positions,vals,width=.36,color=color,label='Strict generation' if metric=='strict' else 'Restricted A-D')
 ax.errorbar(positions,vals,yerr=np.maximum(0,np.array([vals-ci[:,0],ci[:,1]-vals])),fmt='none',color='#333333',capsize=3)
ax.set_xticks(range(3),['Correct image','Shuffled image','No image']);ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,40);ax.set_title('Reproduced SFT: image dependence',loc='left',weight='bold');ax.legend(frameon=False)
ax=axes[1,1]
cats=list(jobs['sft_correct']['categories'])
short=['Coarse','Fine','Instance','Logical','Math','Science']
for m,label,color in zip(models,labels,colors):
 ax.plot(range(6),[jobs[f'{m}_correct']['categories'][c]['restricted']*100 for c in cats],marker='o',markersize=4,label=label,color=color)
ax.set_xticks(range(6),short);ax.set_ylim(0,60);ax.set_ylabel('Restricted A-D accuracy (%)');ax.axhline(25,color='#aaaaaa',linestyle='--',linewidth=.8);ax.set_title('Six categories: 250 questions each',loc='left',weight='bold');ax.legend(frameon=False,fontsize=8,ncol=2)
fig.suptitle('MiniMind-V independent MMStar evaluation | full 1,500 questions',fontsize=15,weight='bold')
fig.supxlabel('FP32, greedy16; protocol fixed before inference. Error bars: paired image-cluster bootstrap 95% CI.\nRestricted A-D scores are diagnostic and are not generated-answer or leaderboard scores.',fontsize=9)
(r/'figures').mkdir(exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(r/f'figures/evaluation.{suffix}',dpi=180)
print('FIGURES_WRITTEN')
