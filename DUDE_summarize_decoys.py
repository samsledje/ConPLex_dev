import sys
import glob
import pandas as pd

directory = sys.argv[1]

lines = []
for f in glob.glob(f'{directory}/*pval.txt'):
    with open(f,'r') as fi:
        target,tstat,pval = fi.readline().strip().split()
        model = '_'.join(f.split('_')[2:-1])
        lines.append((target,model,tstat,pval))
pvals = pd.DataFrame(lines,columns=['Target','Model','TStat','PVal'])
pvals['PVal'] = pvals.PVal.astype(float)

target_types = pd.read_csv('./dataset/DUDe/dude_subset_list.txt',names=['Target','Subset'])

mrg = pd.merge(pvals, target_types, left_on='Target', right_on='Target', how='left')
mrg = mrg.fillna('None')
mrg = mrg.sort_values(by=['Subset','Target','Model'])
with pd.option_context('display.max_rows',None, 'display.max_columns', None, 'display.width', None):
    print(mrg)

summary = pd.concat([
    mrg.groupby('Subset')['PVal'].count(),
    mrg.groupby('Subset')['PVal'].mean()
],axis=1)
summary.columns = ['Count','Mean PVal']
print(summary)
