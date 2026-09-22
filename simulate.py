"""Reproducible synthetic kinematic experiments, not hardware measurements.
Run: python simulate.py. Requires numpy and matplotlib. Outputs under results/.
All five methods and all declared cases are retained, including failures.
"""
from pathlib import Path
import csv, json, platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).parent/'results'; OUT.mkdir(exist_ok=True)
V=1.0; D=.12; R=.8; ALPHA=3.; TMAX=30.
METHODS=['CBF','RCBF','SCBF','SRCBF','CAP']
DTS=[.05,.2,.5,.8]

def control(p,c,goal,dt,method):
    r=p-c; dist=np.linalg.norm(r); n=r/dist; h=dist-R
    v=goal-p; v*=min(1.,V/max(np.linalg.norm(v),1e-15))
    a=min(ALPHA,1/dt) if method=='CAP' else (-np.expm1(-ALPHA*dt)/dt if method in ['SCBF','SRCBF'] else ALPHA)
    b=-a*h+(D if method in ['RCBF','SRCBF','CAP'] else 0.)
    if b>V+1e-10: return None,v
    if n@v >= b-1e-12: return v,v
    tangent=np.array([-n[1],n[0]])
    z=np.clip(tangent@v,-np.sqrt(max(0.,V*V-b*b)),np.sqrt(max(0.,V*V-b*b)))
    return b*n+z*tangent,v

def run(dt,method,offset,regime,seed,trace=False):
    rng=np.random.default_rng(seed)
    p=np.array([0.,rng.uniform(-.08,.08)])
    phi=rng.uniform(0,2*np.pi); bias=D*np.array([np.cos(phi),np.sin(phi)])
    c=np.array([3.,offset]); goal=np.array([6.,0.])
    t=0.; length=0.; effort=0.; clearance=float(np.linalg.norm(p-c)-R)
    status='timeout'; hist=[[t,*p,clearance]]; residual=1e9; speed=0.
    while t<TMAX-1e-10:
        step=min(dt,TMAX-t)
        u,nom=control(p,c,goal,step,method)
        if u is None: status='infeasible'; break
        n=(p-c)/np.linalg.norm(p-c)
        a=min(ALPHA,1/step) if method=='CAP' else (-np.expm1(-ALPHA*step)/step if method in ['SCBF','SRCBF'] else ALPHA)
        b=-a*(np.linalg.norm(p-c)-R)+(D if method in ['RCBF','SRCBF','CAP'] else 0.)
        residual=min(residual,float(n@u-b));speed=max(speed,float(np.linalg.norm(u)))
        disturbance=bias if regime=='bias' else -D*n
        delta=step*(u+disturbance)
        tau=np.clip(-((p-c)@delta)/max(delta@delta,1e-30),0.,1.)
        segclear=float(np.linalg.norm(p+tau*delta-c)-R)
        clearance=min(clearance,segclear)
        length+=float(np.linalg.norm(delta)); effort+=step*float((u-nom)@(u-nom))
        p=p+delta;t+=step;hist.append([t,*p,float(np.linalg.norm(p-c)-R)])
        if clearance < -1e-8: status='violation';break
        if np.linalg.norm(p-goal)<=.15: status='arrived';break
    row=dict(method=method,dt=dt,offset=offset,regime=regime,seed=seed,status=status,
             min_clearance=clearance,time=t,path_length=length,intervention=effort,
             min_residual=residual,max_speed=speed)
    return row,np.array(hist)

def main():
    rows=[];traces={}
    for offset in [.15,.45]:
        for regime in ['bias','inward']:
            for dt in DTS:
                for seed in range(20):
                    for method in METHODS:
                        row,hist=run(dt,method,offset,regime,seed)
                        rows.append(row)
                        if offset==.15 and regime=='inward' and seed==0 and dt==.8: traces[method]=hist
    with (OUT/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary=[]
    for dt in DTS:
        for method in METHODS:
            rr=[r for r in rows if r['dt']==dt and r['method']==method]
            arrived=[r for r in rr if r['status']=='arrived']
            summary.append(dict(dt=dt,method=method,n=len(rr),violations=sum(r['status']=='violation' for r in rr),
                arrived=len(arrived),timeouts=sum(r['status']=='timeout' for r in rr),
                infeasible=sum(r['status']=='infeasible' for r in rr),worst_clearance=min(r['min_clearance'] for r in rr),
                mean_arrival_time=float(np.mean([r['time'] for r in arrived])) if arrived else None,
                mean_arrival_path=float(np.mean([r['path_length'] for r in arrived])) if arrived else None))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),indent=2))
    assert min(r['min_residual'] for r in rows)>=-1e-9
    assert max(r['max_speed'] for r in rows)<=V+1e-9
    assert all(r['min_clearance']>=-1e-8 for r in rows if r['method']=='SRCBF')
    colors=['#D55E00','#0072B2','#CC79A7','#009E73','#8A6A00']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for method,color in zip(METHODS,colors):
        rr=[r for r in summary if r['method']==method]
        ax[0].plot(DTS,[100*r['violations']/r['n'] for r in rr],'o-',label=method,color=color)
        ax[1].plot(DTS,[r['worst_clearance'] for r in rr],'o-',label=method,color=color)
    ax[0].set(ylabel='Safety violations (%)',xlabel='Sampling period (s)',ylim=(-4,104))
    ax[1].set(ylabel='Worst signed clearance (m)',xlabel='Sampling period (s)')
    ax[1].axhline(0,color='black',lw=.7,ls='--')
    ax[0].legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(OUT/'sampling.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    ax[0].add_patch(plt.Circle((3,.15),R,color='#DDDDDD'))
    ax[0].add_patch(plt.Circle((3,.15),.5,color='#999999'))
    for method,color in zip(METHODS,colors):
        hist=traces[method];ax[0].plot(hist[:,1],hist[:,2],label=method,color=color)
        ax[0].plot(hist[-1,1],hist[-1,2],'x',color=color)
        ax[1].plot(hist[:,0],hist[:,3],label=method,color=color)
        np.savetxt(OUT/f'trace_{method}.csv',hist,delimiter=',',header='t,x,y,endpoint_clearance',comments='')
    ax[0].plot(6,0,'k*');ax[0].set(aspect='equal',xlabel='x (m)',ylabel='y (m)',xlim=(-.2,6.3),ylim=(-1.4,1.3))
    ax[1].set(xlabel='Time (s)',ylabel='Endpoint clearance (m)');ax[1].axhline(0,color='black',ls='--',lw=.7)
    ax[1].legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(OUT/'trajectories.png');plt.close(fig)
    print(json.dumps(summary,indent=2))
    print('Trials:',len(rows),'all constraint checks passed')

if __name__=='__main__':main()
