from __future__ import annotations
import ast, math, json, re
from pathlib import Path
from typing import Iterable, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

DATA_PATH = Path('rezultati.csv')
OUT_DIR = Path('results')
OUT_DIR.mkdir(exist_ok=True)

Z = 1.96
THRESHOLDS = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0], dtype=float)
U = 3.0
SCALE_C = 0.05
MODEL_ORDER = ['gpt-4.1','gpt-4o','o3','o4-mini']
PROMPT_ORDER = ['C','M','N']


def parse_list(s):
    return np.asarray(ast.literal_eval(str(s).strip()), dtype=float)


def read_raw(path=DATA_PATH):
    df = pd.read_csv(path, sep=';', engine='python')
    df = df[['i','j','k','t','p','r','data']].copy()
    for c in ['i','j','k']:
        df[c] = df[c].astype(str).str.strip()
    df['values'] = [parse_list(x) for x in df['data']]
    df['n'] = [len(v) for v in df['values']]
    df = df.drop(columns=['data'])
    return df


def robust_parts(vals):
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    q25, q75 = np.quantile(vals, [0.25,0.75])
    iqr = float(q75-q25)
    return med, mad, iqr


def add_row_scores(df: pd.DataFrame, degenerate_policy: str = 'zero') -> pd.DataFrame:
    df = df.copy()
    meds, mads, iqrs = [], [], []
    for vals in df['values']:
        med, mad, iqr = robust_parts(vals)
        meds.append(med); mads.append(mad); iqrs.append(iqr)
    df['median'] = meds
    df['mad'] = mads
    df['iqr'] = iqrs
    df['degenerate'] = (df['mad'] == 0) & (df['iqr'] == 0)
    df['sigma_min'] = df.groupby('i')['mad'].transform(lambda x: SCALE_C * np.median(1.4826*np.asarray(x, dtype=float)))
    z_arrays = []
    out_counts = []
    nAURC_rows = []
    scales = []
    for vals, med, mad, iqr, floor, deg in zip(df['values'], df['median'], df['mad'], df['iqr'], df['sigma_min'], df['degenerate']):
        if deg and degenerate_policy == 'zero':
            z = np.zeros_like(vals, dtype=float)
            s = 0.0
        else:
            s = max(1.4826*mad, (iqr/1.349 if np.isfinite(iqr) else 0.0), floor)
            if not np.isfinite(s) or s <= 0:
                z = np.zeros_like(vals, dtype=float)
                s = 0.0
            else:
                z = np.abs(vals - med) / s
        counts = np.array([np.sum(z > th) for th in THRESHOLDS], dtype=int)
        z_arrays.append(z.astype(float))
        out_counts.append(int(np.sum(z > U)))
        nAURC_rows.append(100*np.trapezoid(counts/len(z), THRESHOLDS)/(THRESHOLDS[-1]-THRESHOLDS[0]))
        scales.append(s)
    df['scale'] = scales
    df['z_values'] = z_arrays
    df['out_count'] = out_counts
    df['row_nAURC'] = nAURC_rows
    return df


def wilson_ci(y, n, z=Z):
    y=float(y); n=float(n)
    if n <= 0: return (np.nan, np.nan)
    p=y/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return center-half, center+half


def nAURC_from_z(z):
    counts = np.array([np.sum(z > th) for th in THRESHOLDS], dtype=float)
    return 100*np.trapezoid(counts/len(z), THRESHOLDS)/(THRESHOLDS[-1]-THRESHOLDS[0])


def logit(x): return np.log(x/(1-x))
def invlogit(x): return 1/(1+np.exp(-x))


def logit_center_ci_percent(vals_percent):
    vals = np.asarray(vals_percent, dtype=float)
    vals = vals[np.isfinite(vals)]
    prop = np.clip(vals/100.0, 1e-8, 1-1e-8)
    y = logit(prop)
    center = 100*invlogit(float(np.mean(y)))
    if len(y) < 2:
        return center, np.nan, np.nan
    se = float(np.std(y, ddof=1)/math.sqrt(len(y)))
    return center, 100*invlogit(np.mean(y)-Z*se), 100*invlogit(np.mean(y)+Z*se)


def normal_ci(vals):
    arr=np.asarray(vals, dtype=float)
    arr=arr[np.isfinite(arr)]
    mean=float(np.mean(arr)) if len(arr) else np.nan
    if len(arr) < 2:
        return mean, np.nan, np.nan
    se=float(np.std(arr,ddof=1)/math.sqrt(len(arr)))
    return mean, mean-Z*se, mean+Z*se


def tail_metrics(z):
    z=np.asarray(z, dtype=float)
    exc = z[z > U] - U
    ae = float(np.mean(exc)) if len(exc) else 0.0
    maxex = float(np.max(exc)) if len(exc) else 0.0
    q90 = float(np.quantile(z,0.90))
    q99 = float(np.quantile(z,0.99))
    r = (q99/q90) if q90 > 0 else (1.0 if q99 == 0 else np.inf)
    q125,q25,q375,q625,q75,q875 = np.quantile(z,[0.125,0.25,0.375,0.625,0.75,0.875])
    moors = ((q875-q625)+(q375-q125))/(q75-q25) if (q75-q25) != 0 else np.nan
    return {'AE':ae, 'MAXEX':maxex, 'R9990':r, 'S99':q99, 'Moors':moors}


def concat_z(series):
    arrs = list(series)
    return np.concatenate(arrs) if arrs else np.array([], dtype=float)


def compute_problem_metrics(df):
    rows=[]; curves=[]; tail_rows=[]
    for (j,k,i), sub in df.groupby(['j','k','i'], sort=False):
        z_all = concat_z(sub['z_values'])
        n = len(z_all)
        counts = np.array([np.sum(z_all > th) for th in THRESHOLDS], dtype=int)
        rows.append({'j':j,'k':k,'i':i,'n':n,'out_count':int(np.sum(z_all>U)), 'p_out':float(np.mean(z_all>U)),
                     'nAURC':100*np.trapezoid(counts/n, THRESHOLDS)/(THRESHOLDS[-1]-THRESHOLDS[0])})
        for th,cnt in zip(THRESHOLDS, counts):
            curves.append({'j':j,'k':k,'i':i,'threshold':th,'count':int(cnt),'n':n,'p_exceed':float(cnt/n)})
        sub_non_deg = sub[~sub['degenerate']]
        z_tail = concat_z(sub_non_deg['z_values']) if len(sub_non_deg) else z_all
        metrics = tail_metrics(z_tail)
        tail_rows.append({'j':j,'k':k,'i':i,'n_tail':len(z_tail), **metrics})
    return pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(tail_rows)


def pooled_outlier_table(df, problem):
    rows=[]
    for (j,k), sub in df.groupby(['j','k'], sort=False):
        n=int(sub['n'].sum()); y=int(sub['out_count'].sum())
        lo,hi=wilson_ci(y,n)
        pm=problem[(problem.j==j)&(problem.k==k)]
        cen,nlo,nhi=logit_center_ci_percent(pm['nAURC'])
        rows.append({'j':j,'k':k,'n':n,'out_count':y,'p_out':y/n,'p_out_pct':100*y/n,
                     'p_out_lo':lo,'p_out_hi':hi,'p_out_pct_lo':100*lo,'p_out_pct_hi':100*hi,
                     'm':pm.i.nunique(),'nAURC':cen,'nAURC_lo':nlo,'nAURC_hi':nhi})
    return pd.DataFrame(rows)


def tail_tables(tail_problem):
    rows=[]; alpha_rows=[]
    for (j,k), sub in tail_problem.groupby(['j','k'], sort=False):
        row={'j':j,'k':k,'m':sub.i.nunique()}
        vals={
            'log10_1p_AE':np.log10(1+sub.AE.astype(float).to_numpy()),
            'log10_1p_MAXEX':np.log10(1+sub.MAXEX.astype(float).to_numpy()),
            'log10_R9990':np.log10(sub.R9990.astype(float).to_numpy()),
            'log10_1p_S99':np.log10(1+sub.S99.astype(float).to_numpy()),
            'Moors':sub.Moors.astype(float).to_numpy(),
        }
        for name, arr in vals.items():
            mean,lo,hi=normal_ci(arr)
            row[name]=mean; row[name+'_lo']=lo; row[name+'_hi']=hi
        rows.append(row)
        ae=sub.AE.astype(float).to_numpy()
        ae_bar=float(np.mean(ae))
        alpha = 1 + U/ae_bar if ae_bar > 0 else np.inf
        if len(ae)>1 and ae_bar>0:
            se_ae=float(np.std(ae,ddof=1)/math.sqrt(len(ae)))
            se_alpha=U/(ae_bar**2)*se_ae
            lo=alpha-Z*se_alpha; hi=alpha+Z*se_alpha
            if lo < 1 and alpha > 5:
                lo = 1.0
        else:
            lo=hi=np.nan
        alpha_rows.append({'j':j,'k':k,'m':sub.i.nunique(),'AE_bar':ae_bar,'alpha':alpha,'alpha_lo':lo,'alpha_hi':hi})
    return pd.DataFrame(rows), pd.DataFrame(alpha_rows)


def pooled_curve(df):
    rows=[]
    for (j,k), sub in df.groupby(['j','k'], sort=False):
        z=concat_z(sub['z_values']); n=len(z)
        for th in THRESHOLDS:
            cnt=int(np.sum(z>th)); rows.append({'j':j,'k':k,'threshold':th,'count':cnt,'n':n,'p_exceed':cnt/n})
    return pd.DataFrame(rows)


def dominance(df, n_boot=2000, seed=123):
    curve=pooled_curve(df)
    points=[]
    for j in MODEL_ORDER:
        for comp in ['C','M']:
            N=curve[(curve.j==j)&(curve.k=='N')].sort_values('threshold')['p_exceed'].to_numpy()
            C=curve[(curve.j==j)&(curve.k==comp)].sort_values('threshold')['p_exceed'].to_numpy()
            points.append({'j':j,'pair':f'N-{comp}','area':float(np.trapezoid(N-C, THRESHOLDS))})
    # row bootstrap
    rng=np.random.default_rng(seed)
    df2=df.reset_index(drop=True).copy()
    counts=np.array([[np.sum(z>th) for th in THRESHOLDS] for z in df2['z_values']], dtype=int)
    cis=[]
    for j in MODEL_ORDER:
        for comp in ['C','M']:
            idxN=df2[(df2.j==j)&(df2.k=='N')].index.to_numpy()
            idxC=df2[(df2.j==j)&(df2.k==comp)].index.to_numpy()
            areas=[]
            for _ in range(n_boot):
                sN=rng.choice(idxN, size=len(idxN), replace=True)
                sC=rng.choice(idxC, size=len(idxC), replace=True)
                pN=counts[sN].sum(axis=0)/df2.loc[sN,'n'].sum()
                pC=counts[sC].sum(axis=0)/df2.loc[sC,'n'].sum()
                areas.append(float(np.trapezoid(pN-pC, THRESHOLDS)))
            lo,hi=np.quantile(areas,[0.025,0.975])
            cis.append({'j':j,'pair':f'N-{comp}','area_lo':float(lo),'area_hi':float(hi)})
    return pd.DataFrame(points).merge(pd.DataFrame(cis), on=['j','pair'])


def dersimonian_laird(effects, variances):
    eff=np.asarray(effects,dtype=float); var=np.clip(np.asarray(variances,dtype=float),1e-12,np.inf)
    w=1/var
    fixed=np.sum(w*eff)/np.sum(w)
    Q=np.sum(w*(eff-fixed)**2); k=len(eff)
    C=np.sum(w)-np.sum(w*w)/np.sum(w)
    tau2=max(0,(Q-(k-1))/C) if C>0 and k>1 else 0.0
    wr=1/(var+tau2)
    mu=np.sum(wr*eff)/np.sum(wr)
    se=math.sqrt(1/np.sum(wr))
    return mu, mu-Z*se, mu+Z*se, tau2


def meta_analysis(problem):
    rows=[]
    for j in MODEL_ORDER:
        for comp in ['C','M']:
            effs=[]; vars=[]
            for i, sub in problem[problem.j==j].groupby('i'):
                a=sub[sub.k==comp].iloc[0]; b=sub[sub.k=='N'].iloc[0]
                p1=float(a.p_out); p0=float(b.p_out); n1=int(a.n); n0=int(b.n)
                effs.append(100*(p1-p0))
                vars.append(10000*(p1*(1-p1)/n1 + p0*(1-p0)/n0))
            _,_,_,tau2=dersimonian_laird(effs,vars)
            mu=float(np.mean(effs))
            se=math.sqrt(tau2/len(effs)) if len(effs)>0 else np.nan
            lo=mu-Z*se; hi=mu+Z*se
            rows.append({'j':j,'contrast':f'{comp}-N','pooled_diff_pp':mu,'ci_low':lo,'ci_high':hi,'tau2':tau2})
    return pd.DataFrame(rows)


def newcombe(y1,n1,y0,n0):
    p1=y1/n1; p0=y0/n0
    l1,u1=wilson_ci(y1,n1); l0,u0=wilson_ci(y0,n0)
    lo=(p1-p0)-math.sqrt((p1-l1)**2+(u0-p0)**2)
    hi=(p1-p0)+math.sqrt((u1-p1)**2+(p0-l0)**2)
    return lo,hi


def temperature_contrasts(df):
    temp=df[df.t.notna()].copy()
    temp['temp_group']=np.where(temp.t.isin([0.0,0.3]),'Low','High')
    rows=[]
    for (j,k), sub in temp.groupby(['j','k'], sort=False):
        if j not in ['gpt-4.1','gpt-4o']: continue
        H=sub[sub.temp_group=='High']; L=sub[sub.temp_group=='Low']
        yH=int(H.out_count.sum()); nH=int(H.n.sum()); yL=int(L.out_count.sum()); nL=int(L.n.sum())
        lo,hi=newcombe(yH,nH,yL,nL)
        rows.append({'j':j,'k':k,'diff_pp':100*(yH/nH-yL/nL),'ci_low_pp':100*lo,'ci_high_pp':100*hi,
                     'y_high':yH,'n_high':nH,'y_low':yL,'n_low':nL})
    ae_rows=[]
    for (j,k,i,g), sub in temp.groupby(['j','k','i','temp_group'], sort=False):
        if j not in ['gpt-4.1','gpt-4o']: continue
        z=concat_z(sub['z_values'])
        ae=tail_metrics(z)['AE']
        ae_rows.append({'j':j,'k':k,'i':i,'temp_group':g,'log10_1p_AE':math.log10(1+ae)})
    ae_df=pd.DataFrame(ae_rows)
    diff=[]
    for (j,k), sub in ae_df.groupby(['j','k'], sort=False):
        wide=sub.pivot(index='i',columns='temp_group',values='log10_1p_AE')
        d=(wide['High']-wide['Low']).to_numpy()
        mean,lo,hi=normal_ci(d)
        diff.append({'j':j,'k':k,'delta_log10_1p_AE':mean,'ci_low':lo,'ci_high':hi,'m':len(d)})
    return pd.DataFrame(rows), pd.DataFrame(diff)


def top_p_contrasts(df):
    top=df[df.p.notna()].copy()
    rows=[]
    for (j,k),sub in top.groupby(['j','k'],sort=False):
        if j not in ['gpt-4.1','gpt-4o']: continue
        A=sub[np.isclose(sub.p,0.90)]; B=sub[np.isclose(sub.p,1.0)]
        yA=int(A.out_count.sum()); nA=int(A.n.sum()); yB=int(B.out_count.sum()); nB=int(B.n.sum())
        lo,hi=newcombe(yA,nA,yB,nB)
        rows.append({'j':j,'k':k,'contrast':'0.90-1.00','diff_pp':100*(yA/nA-yB/nB),'ci_low_pp':100*lo,'ci_high_pp':100*hi})
    ae_rows=[]
    for (j,k,i,pv),sub in top.groupby(['j','k','i','p'],sort=False):
        if j not in ['gpt-4.1','gpt-4o'] or not (np.isclose(pv,0.90) or np.isclose(pv,1.0)): continue
        ae=tail_metrics(concat_z(sub['z_values']))['AE']
        ae_rows.append({'j':j,'k':k,'i':i,'p':float(pv),'log10_1p_AE':math.log10(1+ae)})
    ae_df=pd.DataFrame(ae_rows)
    diff=[]
    for (j,k), sub in ae_df.groupby(['j','k'], sort=False):
        wide=sub.pivot(index='i',columns='p',values='log10_1p_AE')
        c90=[c for c in wide.columns if np.isclose(c,0.90)][0]; c100=[c for c in wide.columns if np.isclose(c,1.0)][0]
        d=(wide[c90]-wide[c100]).to_numpy()
        mean,lo,hi=normal_ci(d)
        diff.append({'j':j,'k':k,'contrast':'0.90-1.00','delta_log10_1p_AE':mean,'ci_low':lo,'ci_high':hi,'m':len(d)})
    return pd.DataFrame(rows), pd.DataFrame(diff)


def common_reference(df):
    x=df.copy()
    x['_t']=x.t.astype('object').where(x.t.notna(),'NA')
    x['_p']=x.p.astype('object').where(x.p.notna(),'NA')
    rows=[]
    problem_floor=x.groupby('i')['sigma_min'].median().to_dict()
    for (i,j,t,p), g in x.groupby(['i','j','_t','_p'], sort=False):
        vals=concat_z(g['values']) 
        med=float(np.median(vals)); mad=float(np.median(np.abs(vals-med)))
        q25,q75=np.quantile(vals,[0.25,0.75]); iqr=float(q75-q25)
        s=max(1.4826*mad,iqr/1.349 if np.isfinite(iqr) else 0,problem_floor.get(i,0.0))
        for k, sg in g.groupby('k', sort=False):
            vv=concat_z(sg['values'])
            z=np.abs(vv-med)/s if s>0 and np.isfinite(s) else np.zeros_like(vv)
            n=len(z); counts=np.array([np.sum(z>th) for th in THRESHOLDS])
            rows.append({'i':i,'j':j,'k':k,'t':t,'p':p,'n':n,'out_count':int(np.sum(z>U)),
                         'p_out_pct':100*np.mean(z>U),
                         'nAURC':100*np.trapezoid(counts/n,THRESHOLDS)/(THRESHOLDS[-1]-THRESHOLDS[0])})
    cell=pd.DataFrame(rows)
    summ=[]
    for (j,k), sub in cell.groupby(['j','k'], sort=False):
        n=int(sub.n.sum()); y=int(sub.out_count.sum())
        summ.append({'j':j,'k':k,'n':n,'p_out_pct':100*y/n,'nAURC':float(sub.nAURC.mean())})
    return pd.DataFrame(summ), cell


def kendall_table(tail, pooled, alpha):
    cell=tail.merge(pooled[['j','k','p_out','nAURC']],on=['j','k']).merge(alpha[['j','k','alpha']],on=['j','k'])
    heavy=['log10_1p_AE','log10_1p_MAXEX','log10_R9990','log10_1p_S99','Moors']
    targets=['p_out','nAURC','alpha']
    rows=[]
    for h in heavy:
        for t in targets:
            tau,p=kendalltau(cell[h], cell[t], nan_policy='omit')
            rows.append({'heavy_metric':h,'target':t,'tau':float(tau),'p_value':float(p)})
    return pd.DataFrame(rows)


def icc(df):
    rl=df[['j','k','i','t','p','r','out_count','n','row_nAURC']].copy()
    rl['p_out']=rl.out_count/rl.n
    rl['_t']=rl.t.astype('object').where(rl.t.notna(),'NA')
    rl['_p']=rl.p.astype('object').where(rl.p.notna(),'NA')
    def one(outcome):
        groups=[g[outcome].to_numpy(float) for _,g in rl.groupby(['j','k','i','_t','_p'], sort=False)]
        G=len(groups); ns=np.array([len(g) for g in groups],float); ntot=int(ns.sum()); kbar=float(ns.mean())
        grand=np.concatenate(groups).mean()
        ssb=sum(len(g)*(g.mean()-grand)**2 for g in groups)
        ssw=sum(((g-g.mean())**2).sum() for g in groups)
        msb=ssb/(G-1); msw=ssw/(ntot-G)
        val=(msb-msw)/(msb+(kbar-1)*msw)
        return {'outcome':outcome,'ICC1':val,'G':G,'n_tot':ntot,'kbar':kbar,'MSB':msb,'MSW':msw}
    return pd.DataFrame([one('p_out'),one('row_nAURC')])


def write_all(prefix, df, n_boot=2000):
    problem, curves, tail_problem = compute_problem_metrics(df)
    pooled=pooled_outlier_table(df,problem)
    tail, alpha=tail_tables(tail_problem)
    dom=dominance(df, n_boot=n_boot, seed=123)
    meta=meta_analysis(problem)
    temp_p,temp_ae=temperature_contrasts(df)
    top_p,top_ae=top_p_contrasts(df)
    common, common_cell=common_reference(df)
    kend=kendall_table(tail,pooled,alpha)
    ic=icc(df)
    objs={'problem_level_metrics':problem,'problem_level_curves':curves,'problem_level_tail_metrics':tail_problem,
          'table_pooled_outlier_nAURC':pooled,'table_tail_diagnostics':tail,'table_tail_alpha':alpha,
          'table_dominance_curves':dom,'table_meta_pout':meta,'table_temperature_pout':temp_p,'table_temperature_AE':temp_ae,
          'table_top_p_pout':top_p,'table_top_p_AE':top_ae,'table_common_reference':common,'common_reference_cells':common_cell,
          'table_kendall':kend,'table_icc':ic}
    for name, obj in objs.items():
        obj.to_csv(OUT_DIR/f'{prefix}_{name}.csv',index=False)
    return objs


def main():
    raw=read_raw()
    print('Raw rows', len(raw), 'total entries', raw.n.sum())
    df_zero=add_row_scores(raw, 'zero')
    main_objs=write_all('manuscript_mode', df_zero)
    df_floor=add_row_scores(raw, 'floor')
    floor_objs=write_all('strict_floor_mode', df_floor)
    pd.set_option('display.max_rows', 100)
    print('\n pooled')
    print(main_objs['table_pooled_outlier_nAURC'][['j','k','n','p_out_pct','p_out_pct_lo','p_out_pct_hi','nAURC','nAURC_lo','nAURC_hi']].sort_values(['j','k']).to_string(index=False))
    print('\n tail')
    print(main_objs['table_tail_diagnostics'][['j','k','log10_1p_AE','log10_1p_MAXEX','log10_R9990','log10_1p_S99','Moors']].sort_values(['j','k']).to_string(index=False))
    print('\n alpha')
    print(main_objs['table_tail_alpha'][['j','k','alpha','alpha_lo','alpha_hi']].sort_values(['j','k']).to_string(index=False))
    print('\n dominance')
    print(main_objs['table_dominance_curves'].sort_values(['j','pair']).to_string(index=False))
    print('\n meta')
    print(main_objs['table_meta_pout'].sort_values(['j','contrast']).to_string(index=False))
    print('\n Mod temperature')
    print(main_objs['table_temperature_pout'][['j','k','diff_pp','ci_low_pp','ci_high_pp']].sort_values(['j','k']).to_string(index=False))
    print('\n Floor temperature')
    print(floor_objs['table_temperature_pout'][['j','k','diff_pp','ci_low_pp','ci_high_pp']].sort_values(['j','k']).to_string(index=False))
    print('\n Kendall means')
    print(main_objs['table_kendall'].groupby('target').tau.mean())
    print('Output:', OUT_DIR)

if __name__ == '__main__':
    main()
