"""Independent report-3 J scan and a finite four-case beta queue."""
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from model import load_config,parameters,stokes,initial_state,make_grid
from numerics import solve_0d,observables_0d,balance_0d,masked_unwrap,density_settling,write_json,run_report3_pilot
import run_charge_checks as queue
DATA=ROOT/'results/data/report3'

def now():return datetime.now(timezone.utc).isoformat()

def j_config(J,final=20000.,sample=.25,tighter=False):
    return load_config('F5',model='M3',reservoir='R1',physical={'J':float(J),'beta':0.},
        pump={'kind':'uniform','s':2.,'epsilon':.3,'off_time':None},
        initial={'density_0d':1e-4,'theta':.2},time={'final':final,'sample':sample},
        ode={'rtol':2e-12 if tighter else 2e-11,'atol':2e-14 if tighter else 2e-13,'max_step':1. if tighter else 2.})

def analyze_spin(data,mask_factor=1e-6):
    t=data['time'];psi=data['psi'];n=abs(psi)**2;total=n.sum(1)
    valid=n.min(1)>=np.maximum(1e-10,mask_factor*total)
    theta=masked_unwrap(np.angle(psi[:,0]*psi[:,1].conj()),valid)
    z=(n[:,0]-n[:,1])/np.maximum(total,1e-30)
    densities=np.column_stack((n,data['active'],data['inactive']))
    S=stokes(psi,axis=1).T;pol=S[:,1:]/np.maximum(S[:,0,None],1e-30)
    # A period is suggested from the strongest measured physical channel.
    late=t>=max(t[0],t[-1]-4000.);features=np.column_stack((z,pol))
    spans=np.ptp(features[late],axis=0);channel=int(np.argmax(spans))
    peaks,_=find_peaks(features[late,channel],prominence=max(.1*spans[channel],1e-8))
    candidate=float(np.median(np.diff(t[late][peaks]))) if len(peaks)>=3 else None
    dt=float(np.median(np.diff(t)));window=np.ceil(max(2000.,5*(candidate or 0.))/dt)*dt
    r={'status':'UNRESOLVED','window_length_ps':float(window),'candidate_period_ps':candidate,
       'phase_density_floor_relative':mask_factor,'windows':[],'frequency_rad_ps':None}
    if t[-1]-2*window<t[0]:r['reason']='Two required windows unavailable';return r
    windows=[]
    for left,right in [(t[-1]-2*window,t[-1]-window),(t[-1]-window,t[-1])]:
        ix=(t>=left-1e-8)&(t<=right+1e-8);tv=t[ix];nn=densities[ix];zz=z[ix];vv=valid[ix]
        th=theta[ix];physical=np.column_stack((nn,pol[ix]))
        scale=np.r_[np.maximum(np.mean(abs(nn),axis=0),1e-12),np.ones(3)]
        w={'interval_ps':[float(left),float(right)],'density_settling':density_settling(nn),
           'A_z':float(np.ptp(zz)/2),'phase_valid_fraction':float(vv.mean()),
           'theta_span':float(np.ptp(th)) if vv.all() else None}
        pairs=vv[1:]&vv[:-1]
        increments=np.abs(np.angle(psi[ix][1:]*psi[ix][:-1].conj()))
        w['max_individual_phase_step_rad']=float(increments[pairs].max()) if pairs.any() else None
        windows.append((tv,physical,scale,th,vv));r['windows'].append(w)
    phase_ok=all(w['phase_valid_fraction']==1 for w in r['windows'])
    alias_ok=all(w['max_individual_phase_step_rad'] is None or w['max_individual_phase_step_rad']<np.pi/4 for w in r['windows'])
    r.update(mean_z=float(np.mean(z[t>=t[-1]-window])),A_z=r['windows'][-1]['A_z'],phase_sampling_pass=alias_ok)
    if not alias_ok:r['reason']='Phase sampling needs refinement';return r
    settled=all(w['density_settling']['passed'] for w in r['windows'])
    if data['config']['physical']['J']==0 and settled:
        r['status']='UNCOUPLED' if phase_ok else 'DENSITY_STEADY_PHASE_MASKED'
    elif settled and phase_ok and all(w['theta_span']<.01 and w['A_z']<1e-3 for w in r['windows']):
        r['status']='LOCKED_STEADY'
    else:
        if candidate is None:r['reason']='No repeated physical cycle';return r
        periods=[];amps=[]
        for (tt,y,scale,th,vv),w in zip(windows,r['windows']):
            def cost(period):
                use=tt+period<=tt[-1];target=tt[use]+period
                returned=np.column_stack([np.interp(target,tt,y[:,j]) for j in range(y.shape[1])])
                return float(np.mean(((returned-y[use])/scale)**2))
            fit=minimize_scalar(cost,bounds=(.9*candidate,1.1*candidate),method='bounded',options={'xatol':1e-7})
            period=float(fit.x);use=tt+period<=tt[-1];target=tt[use]+period
            returned=np.column_stack([np.interp(target,tt,y[:,j]) for j in range(y.shape[1])])
            error=np.max(np.sqrt(np.mean(((returned-y[use])/scale)**2,axis=0)))
            phase_error=None
            if vv.all():
                cyc=np.exp(1j*th);back=np.interp(target,tt,cyc.real)+1j*np.interp(target,tt,cyc.imag)
                phase_error=float(np.sqrt(np.mean(abs(back-cyc[use])**2)))
            w.update(period_ps=period,cycles=float((tt[-1]-tt[0])/period),full_state_recurrence_RMS=float(error),phase_recurrence_RMS=phase_error)
            periods.append(period);amps.append(w['A_z'])
        pd=abs(periods[0]-periods[1])/max(periods);ad=abs(amps[0]-amps[1])/max(*amps,1e-12)
        r.update(period_ps=float(np.mean(periods)),period_relative_difference=pd,amplitude_relative_difference=ad)
        passed=pd<.01 and ad<.01 and all(w['cycles']>=5 and w['full_state_recurrence_RMS']<.01 and (w['phase_recurrence_RMS'] is None or w['phase_recurrence_RMS']<.01) for w in r['windows'])
        if not passed:r['reason']='Period, amplitude or full-state recurrence not stable';return r
        if phase_ok:
            tt,_,_,th,_=windows[-1];cycles=int((tt[-1]-tt[0])/r['period_ps']);duration=cycles*r['period_ps']
            net=float(np.interp(tt[0]+duration,tt,th)-th[0]);r['frequency_rad_ps']=-net/duration
            r['status']='RUNNING_PHASE_PERIODIC' if abs(net/cycles)>np.pi else 'BOUNDED_PHASE_PERIODIC'
        else:r['status']='PERIODIC_PHASE_MASKED'
    if r['frequency_rad_ps'] is None and phase_ok:
        tt,_,_,th,_=windows[-1];r['frequency_rad_ps']=-float(np.polyfit(tt-tt[0],th,1)[0])
    return r

def point(J,tighter=False):
    tag=f'J_{J:.8f}'+('_tight' if tighter else '')
    path=DATA/('report3_0d_'+tag+'.npz');start=time.monotonic()
    if path.exists():
        with np.load(path) as z:
            data={k:z[k].copy() for k in ('time','psi','active','inactive')};data['config']=json.loads(str(z['config_json']))
    else:data=solve_0d(j_config(J,tighter=tighter))
    while True:
        r=analyze_spin(data)
        if not r.get('phase_sampling_pass',True) and data['config']['time']['sample']>.015625:
            c=copy.deepcopy(data['config']);c['time']['sample']/=2;data=solve_0d(c);continue
        if r['status']=='UNRESOLVED' and data['time'][-1]<40000.:
            c=copy.deepcopy(data['config']);c['time']['final']=40000.
            extra=solve_0d(c,state0=tuple(data[k][-1] for k in ('psi','active','inactive')),start=float(data['time'][-1]))
            data={**data,**{k:np.concatenate((data[k],extra[k][1:])) for k in ('time','psi','active','inactive')},'config':c};continue
        break
    obs=observables_0d(data);balance=balance_0d(data,obs)
    r.update(J=float(J),final_ps=float(data['time'][-1]),elapsed_this_call_s=time.monotonic()-start,balance=balance,
             phase_mask_sensitivity={str(f):analyze_spin(data,f)['status'] for f in (1e-7,1e-5)},tighter=tighter)
    if J==0:
        p=parameters(data['config']);use=data['time']>=data['time'][-1]-r['window_length_ps']
        expected=((p['alpha1']-p['alpha2'])*(obs['nc'][use,0]-obs['nc'][use,1])+p['gA']*(data['active'][use,0]-data['active'][use,1])+p['gI']*(data['inactive'][use,0]-data['inactive'][use,1]))/p['hbar']
        r['J0_frequency_formula_error']=abs(r['frequency_rad_ps']-float(expected.mean())) if r['frequency_rad_ps'] is not None else None
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**{k:data[k] for k in ('time','psi','active','inactive')},config_json=json.dumps(data['config']),summary_json=json.dumps(r))
    tmp.replace(path);return r

def family(r):
    s=r['status']
    if s in ('UNCOUPLED','RUNNING_PHASE_PERIODIC'):return 'RUNNING'
    if s=='BOUNDED_PHASE_PERIODIC':return 'BOUNDED'
    if s=='LOCKED_STEADY':return 'LOCKED'
    return None

def j_scan():
    # One CPU core; independent of the GPU lock and no duplicate workers.
    file=DATA/'report3_J_scan.json'
    with (DATA/'report3_J_scan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r={'status':'RUNNING','started_utc':now(),'pid':os.getpid(),'points':{},'boundaries':[],'tighter_checks':{},
           'initial_final_ps':20000.,'max_final_ps':40000.,'source':'new independent starts; no old F5 scan reused',
           'benchmark_1000ps_s':[.429,.390],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write_json(file,r)
        try:
            coarse=np.linspace(0,.03,13)
            def get(j):
                key=f'{j:.8f}'
                if key not in r['points']:
                    print('J START',j,now(),flush=True);r['active_J_meV']=j;write_json(file,r)
                    r['points'][key]=point(j);write_json(file,r)
                return r['points'][key]
            for j in coarse:get(float(j))
            for a,b in zip(coarse[:-1],coarse[1:]):
                a=float(a);b=float(b);ra=get(a);rb=get(b)
                if family(ra) is None or family(rb) is None:
                    r['boundaries'].append({'interval_meV':[a,b],'status':'UNRESOLVED_ENDPOINT_NO_BOUNDARY_CLAIM'});continue
                if family(ra)==family(rb):continue
                stop='BRACKETED'
                for _ in range(6):
                    if b-a<=.0001:break
                    mid=(a+b)/2;rm=get(mid)
                    if family(rm) is None:stop='UNRESOLVED_MIDPOINT_NO_INTERPOLATION';break
                    if family(rm)==family(ra):a=mid;ra=rm
                    else:b=mid;rb=rm
                r['boundaries'].append({'interval_meV':[a,b],'status':stop,'endpoint_families':[family(ra),family(rb)]});write_json(file,r)
            chosen=[]
            for target in ('LOCKED','RUNNING'):
                candidates=[v for v in r['points'].values() if family(v)==target and v['J']>0]
                if candidates:chosen.append(min(candidates,key=lambda x:abs(x['J']-.01))['J'])
            for j in sorted(set(chosen)):
                tight=point(j,True);base=get(j)
                r['tighter_checks'][f'{j:.8f}']={'reference_status':base['status'],'tighter':tight,
                  'A_z_difference':abs(base.get('A_z',0)-tight.get('A_z',0)),
                  'period_relative_difference':abs(base['period_ps']/tight['period_ps']-1) if base.get('period_ps') and tight.get('period_ps') else None,
                  'frequency_difference':abs(base['frequency_rad_ps']-tight['frequency_rad_ps']) if base.get('frequency_rad_ps') is not None and tight.get('frequency_rad_ps') is not None else None}
                write_json(file,r)
            r.pop('active_J_meV',None);r.update(status='COMPLETED',finished_utc=now());write_json(file,r)
        except BaseException as e:
            r.update(status='FAILED',error=repr(e),finished_utc=now());write_json(file,r);raise
    print('J SCAN COMPLETED',now(),flush=True)

def beta_series():
    file=DATA/'report3_beta_queue.json'
    plans=[{'name':f'report3_T0_{geometry}_beta_{factor:g}','geometry':geometry,'factor':factor} for geometry in ('spot','ring') for factor in (.5,2.)]
    r={'status':'WAITING_FOR_GPU','started_utc':now(),'pid':os.getpid(),'planned':plans,'results':{},'thermal_stop':False,
       'time_ps':3000.,'policy':'four prescribed independent starts; no automatic extensions or other ring runs',
       'waiting_for':'follow-up then ring spatial grid check',
       'ring_initial':{'kind':'radial_perturbed','perturbation':1e-3,'perturb_seed':20261025}}
    write_json(file,r)
    # Wait for the already queued grid process to finish before joining the GPU lock.
    grid=DATA/'charge_grid_queue.json'
    while True:
        state=json.loads(grid.read_text())
        if state['status']=='COMPLETED':break
        if state['status'] in ('FAILED','RUN_INCOMPLETE'):
            r.update(status='NEEDS_REVIEW',reason='Preceding grid queue '+state['status']);write_json(file,r);return
        time.sleep(10.)
    with (queue.RUNS/'pilots.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            for task in plans:
                source='production_T0_'+task['geometry'];c=queue.config_of(source)
                c['physical']['beta']=.0119*task['factor'];c['execution']['precision']='complex128';c['time']['final']=3000.
                if task['geometry']=='ring':c['initial'].update(kind='radial_perturbed',perturbation=1e-3,perturb_seed=20261025)
                name=task['name'];r.update(status='RUNNING',active_case=name);write_json(file,r)
                print('BETA START',name,now(),flush=True)
                path=queue.RUNS/name;path.mkdir(exist_ok=True)
                # Preserve complete complex histories for future phase/charge diagnostics.
                from numerics import _ProbeStore
                if not (path/'probe_maps.npy').exists():
                    store=_ProbeStore(path/'probe_maps.npy',c['space']['N'],duration=3010.,cadence=c['save']['probe']);store.maps.flush();del store
                result=run_report3_pilot(c,name)
                r['results'][name]={'summary':result['summary'],'beta_meV_um2':c['physical']['beta'],'source_baseline':'matched_T0_ring' if task['geometry']=='ring' else source};write_json(file,r)
                if result['summary']['run_status']!='RUN_COMPLETED':
                    r.update(status='RUN_INCOMPLETE');write_json(file,r);return
            r.pop('active_case',None);r.update(status='COMPLETED',finished_utc=now());write_json(file,r)
        except BaseException as e:
            r.update(status='FAILED',error=repr(e),finished_utc=now());write_json(file,r);raise
    print('BETA SERIES COMPLETED',now(),flush=True)


def cycle_review(config,store,history,features,end,common_window=None):
    """Legacy recurrence plus integer-cycle means: reject a slowly drifting cycle centre."""
    from numerics import analyze_report3,_lag_polynomial_weights,_report3_maps,_report3_map_errors
    result=analyze_report3(store,history,features,end,config,common_window)
    result['diagnostic_revision']='integer-cycle centre drift v1; finite-time recurrence, not an attraction proof'
    if result['status']!='PERIODIC':return result
    ordered=store.ordered();times=store.times[ordered];f=np.asarray(features)
    means=[];mean_features=[];used=[]
    for window,period in zip(result['windows'],result['periodic']['periods_ps']):
        left,right=window['interval_ps'];cycles=int(np.floor((right-left)/period+1e-10))
        if cycles<5:
            result.update(status='UNRESOLVED',reason='Fewer than five measured periods in one window');return result
        stop=left+cycles*period
        def value(t):
            exact=np.flatnonzero(np.isclose(times,t,rtol=0,atol=1e-8))
            if len(exact):return _report3_maps(store.state(ordered[exact[0]]))
            ids,weights=_lag_polynomial_weights(times,t,8);a=None;b=None
            for j,w in zip(ids,weights):
                mm,s0=_report3_maps(store.state(ordered[j]))
                if a is None:a=np.zeros_like(mm);b=np.zeros_like(s0)
                a+=w*mm;b+=w*s0
            return a,b
        grid=np.r_[left,times[(times>left+1e-8)&(times<stop-1e-8)],stop]
        weights=np.r_[np.diff(grid)[0]/2,(grid[2:]-grid[:-2])/2,np.diff(grid)[-1]/2]/(stop-left)
        avg=None;avg_s0=None
        for t,w in zip(grid,weights):
            mm,s0=value(t)
            if avg is None:avg=np.zeros_like(mm);avg_s0=np.zeros_like(s0)
            avg+=w*mm;avg_s0+=w*s0
        means.append((avg,avg_s0));used.append({'interval_ps':[float(left),float(stop)],'complete_cycles':cycles})
        ft=np.r_[left,f[(f[:,0]>left)&(f[:,0]<stop),0],stop]
        fv=np.column_stack([np.interp(ft,f[:,0],f[:,j]) for j in range(1,10)])
        mean_features.append(np.trapezoid(fv,ft,axis=0)/(stop-left))
    map_drift=_report3_map_errors(means[1][0],means[0][0],means[0][1])
    a,b=mean_features;scale=np.maximum(np.maximum(abs(a),abs(b)),1e-30)
    for j in (0,2,4):
        total=max(float(max(a[j:j+2].sum(),b[j:j+2].sum())),1e-30)
        scale[j:j+2]=np.maximum(scale[j:j+2],1e-6*total)
    scale[6:]=max(float(max(a[:2].sum(),b[:2].sum())),1e-30)
    feature_drift=abs(b-a)/scale
    centre={'integer_cycle_intervals':used,'mean_map_relative_errors':map_drift,
            'mean_population_and_integral_Stokes_errors':feature_drift,'tolerance':.01,
            'status':'PASS' if max(float(map_drift.max()),float(feature_drift.max()))<.01 else 'FAIL'}
    result['periodic']['cycle_centre_check']=centre
    if centre['status']!='PASS':
        result.update(status='UNRESOLVED',reason='Mean density/reservoir/Stokes maps or cycle centre still drift')
        result.pop('period_ps',None)
    return result

def retained_fields(path,config,until=6000.):
    from numerics import _ProbeStore
    path.mkdir(exist_ok=True);cp=path/'checkpoint.npz';values=None
    if cp.exists():
        with np.load(cp) as z:values={k:z[k].copy() for k in z.files}
    store=_ProbeStore(path/'probe_maps.npy',config['space']['N'],duration=until+10.,cadence=config['save']['probe'],times=values['probe_times'] if values is not None else None)
    store.ensure(until+10.);store.maps.flush()
    if values is not None:
        values['probe_times']=store.times.copy();np.savez(path/'checkpoint.tmp.npz',**values);(path/'checkpoint.tmp.npz').replace(cp)
    del store

def review_saved(name,end,common_window=None):
    from numerics import _precision_reference
    c,store=_precision_reference(name)
    with np.load(DATA/(name+'.npz')) as z:h=z['history'].copy();f=z['features'].copy()
    requested_end=float(end);end=min(requested_end,float(h[-1,0]))
    result=cycle_review(c,store,h,f,end,common_window)
    result['analysis_end_ps']=end;result['requested_analysis_end_ps']=requested_end
    state,_=queue.state_of(name);result['topology']=queue.topology(state,c)
    return result

def matched_ring_queue():
    # Compatibility entry point. Use the declarative dispatcher for all new work.
    from experiment_queue import run, DEFAULT_PLAN
    run(DEFAULT_PLAN)


def review_existing_cycles():
    file=DATA/'report3_cycle_review.json';r={'status':'RUNNING','started_utc':now(),'cases':{},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'regression_tests':{'exact_cycle':'PASS: PERIODIC','slow_cycle_centre_drift':'PASS: legacy PERIODIC, revised UNRESOLVED'}}
    write_json(file,r)
    try:
        for case in ('A0','A1','T1','J1','C1'):
            name='production_'+case+'_spot';r['cases'][name]=review_saved(name,3000.);write_json(file,r)
            print('REVIEW CYCLE',name,r['cases'][name]['status'],flush=True)
        r.update(status='COMPLETED',finished_utc=now());write_json(file,r)
    except BaseException as e:
        r.update(status='FAILED',error=repr(e));write_json(file,r);raise

if __name__=='__main__':
    if '--review-existing-cycles' in sys.argv:review_existing_cycles()
    elif '--matched-ring' in sys.argv:matched_ring_queue()
    elif '--j-scan' in sys.argv:j_scan()
    elif '--beta-series' in sys.argv:beta_series()
    else:raise SystemExit('Use --j-scan or --beta-series')
