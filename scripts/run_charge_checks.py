"""Finite two-hour J1 investigation; existing reference runs stay intact."""
from pathlib import Path
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from model import initial_state, make_grid, parameters, injection
from numerics import (TorchSpinStepper, split_step, relative_norm, write_json,
                      run_report3_pilot, _precision_metrics, vortex_diagnostics)

DATA = ROOT / 'results/data/report3'
RUNS = ROOT / 'results/runs/report3'
MANIFEST = DATA / 'charge_checks_queue.json'
SMALL = 'boundary_J1_ring_seeded_L256'
LARGE = 'boundary_J1_ring_seeded_L384'
SINGLE = 'charge_J1_seeded_c64_L256_t2000'
PLANS = [
    {'name': SINGLE, 'source': SMALL, 'precision': 'complex64', 'final': 2000., 'estimated_minutes': 12, 'purpose': 'T1 matched explicit seed c64/128'},
    {'name': 'charge_J1_seeded_c128_L256_t3000', 'source': SMALL, 'precision': 'complex128', 'final': 3000., 'fork': True, 'estimated_minutes': 14, 'purpose': 'T2 late state small domain'},
    {'name': 'charge_J1_seeded_c128_L384_t3000', 'source': LARGE, 'precision': 'complex128', 'final': 3000., 'fork': True, 'estimated_minutes': 24, 'purpose': 'T2/T4 late state large domain'},
    {'name': 'charge_J1_seeded_c64_L256_t3000', 'source': SINGLE, 'precision': 'complex64', 'final': 3000., 'fork': True, 'estimated_minutes': 7, 'purpose': 'T1/T2 matched precision at common late time'},
    {'name': 'charge_J1_seeded_c128_dt0125_t2000', 'source': SMALL, 'precision': 'complex128', 'dt': .0125, 'final': 2000., 'estimated_minutes': 54, 'purpose': 'T3 independent start with half time step'},
]

def stamp(): return datetime.now(timezone.utc).isoformat()
def config_of(name): return yaml.safe_load((RUNS/name/'resolved_config.yaml').read_text())
def state_of(name):
    with np.load(RUNS/name/'checkpoint.npz') as z:
        return tuple(z[k].copy() for k in ('psi','active','inactive')), float(z['step'])
def save(): write_json(MANIFEST, report)
def topology(state, config):
    g=make_grid(config); p=parameters(config)
    return [vortex_diagnostics(state[0][j],g,p,radii=(4.,6.,8.,10.,12.)) for j in (0,1)]
def mirror(state):
    return tuple(v[...,(-np.arange(v.shape[-1]))%v.shape[-1]].copy() for v in state)

def short_checks():
    c=config_of(SMALL); c['space']={'N':128,'L':256.}
    g=make_grid(c); p=parameters(c); common=initial_state(c,g)
    # Smooth nonzero reservoirs test the coupled local equations, not only the vacuum.
    pump=injection(c,g)
    common=(common[0], .2*pump.copy(), .3*pump.copy())
    reference=tuple(v.copy() for v in common)
    for _ in range(4): reference=split_step(reference,pump,p,c['time']['dt'],g,accelerated=False)
    checks={}
    for precision,tol in (('complex128',1e-10),('complex64',5e-5)):
        cc=copy.deepcopy(c); cc['execution']['precision']=precision
        compiled=TorchSpinStepper(cc,g,compiled=True)
        eager=TorchSpinStepper(cc,g,compiled=False)
        def advance(backend,state):
            backend.staggered=False; v=backend.to_device(state)
            for _ in range(4):v,_,_=backend.step(v,close=True)
            backend.synchronize();return backend.to_numpy(v)
        a=advance(compiled,common); b=advance(eager,common)
        reflected=advance(compiled,mirror(common))
        parity=[relative_norm(v,r) for v,r in zip(a,reference)]
        fusion=[relative_norm(v,r) for v,r in zip(a,b)]
        equiv=[relative_norm(v,r) for v,r in zip(reflected,mirror(a))]
        checks[precision]={'numpy_error':parity,'compiled_eager_error':fusion,
                           'reflection_equivariance_error':equiv,'tolerance':tol,
                           'status':'PASS' if max(parity+fusion+equiv)<tol else 'FAIL'}
        write_json(DATA/'charge_short_checks.json',checks)
    if any(v['status']!='PASS' for v in checks.values()):
        raise RuntimeError('Short implementation checks failed; long queue stopped')
    return checks

def fork_run(source,target):
    dst=RUNS/target
    if dst.exists():return
    temp=RUNS/(target+'.copying')
    if temp.exists():raise RuntimeError('Unfinished copy requires inspection: '+str(temp))
    temp.mkdir()
    for filename in ('checkpoint.npz','probe_maps.npy','angular.npz','animation_frames.npz'):
        src=RUNS/source/filename
        if src.exists():subprocess.run(['cp','--reflink=auto',str(src),str(temp/filename)],check=True)
    temp.replace(dst)

def initial_queue():
    global report
    report={'status':'STARTING','started_utc':stamp(),'pid':os.getpid(),
            'budget_hours':2.,'thermal_stop':False,'planned':PLANS,'results':{},
            'policy':'one GPU process; shared absolute deadline; checkpoints and incomplete status at budget; references preserved',
            'saves':'complex full fields every 5 ps; populations every 0.25 ps; checkpoint and permanent animation frames every 200 ps'}
    save()
    deadline=time.monotonic()+7200.-120. # reserve final checkpoint, classification and export
    try:
        with (RUNS/'pilots.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            report['status']='SHORT_CHECKS';save()
            report['short_checks']=short_checks();save()
            for task in PLANS:
                if time.monotonic()>=deadline:
                    report['status']='BUDGET_REACHED';break
                name=task['name'];c=config_of(task['source'])
                c['time']['final']=task['final'];c['execution']['precision']=task['precision']
                if 'dt' in task:c['time']['dt']=task['dt']
                if task.get('fork'):fork_run(task['source'],name)
                report.update(status='RUNNING',active_case=name);save()
                print('START',name,stamp(),flush=True)
                result=run_report3_pilot(c,name,common_window=1000.,deadline=deadline)
                state,_=state_of(name)
                record={'summary':result['summary'],'topology':topology(state,c),
                        'source':task['source'],'completed_utc':stamp()}
                if result['summary']['run_status']=='RUN_COMPLETED' and not task.get('fork'):
                    ref,_=state_of(task['source'])
                    record['comparison_at_2000_ps']=_precision_metrics(state,ref,make_grid(c))
                    record['reference_topology']=topology(ref,c)
                report['results'][name]=record;save()
                if result['summary']['run_status']!='RUN_COMPLETED':
                    report['status']='BUDGET_REACHED';break
            else:report['status']='COMPLETED'
            # Compare late states only when both reached exactly the common target.
            pairs=[('charge_J1_seeded_c64_L256_t3000','charge_J1_seeded_c128_L256_t3000'),
                   ('charge_J1_seeded_c128_L384_t3000','charge_J1_seeded_c128_L256_t3000')]
            report['late_comparisons']={}
            for a,b in pairs:
                if all(n in report['results'] and report['results'][n]['summary']['run_status']=='RUN_COMPLETED' for n in (a,b)):
                    sa,_=state_of(a); sb,_=state_of(b);cb=config_of(b)
                    if sa[0].shape!=sb[0].shape:
                        offset=(sa[0].shape[-1]-sb[0].shape[-1])//2;n=sb[0].shape[-1]
                        sa=tuple(v[:,offset:offset+n,offset:offset+n] for v in sa)
                    report['late_comparisons'][a+' vs '+b]=_precision_metrics(sa,sb,make_grid(cb))
            report.pop('active_case',None);report['finished_utc']=stamp();save()
    except BaseException as e:
        report.update(status='FAILED',error=repr(e),finished_utc=stamp());save();raise
    print('QUEUE',report['status'],stamp(),flush=True)

# Follow-up uses the same lock as the first queue; no competing GPU producer.
FOLLOWUP = DATA / 'charge_followup_queue.json'
FOLLOWUP_PLANS = [
    {'name':'charge_J1_seeded_c128_dt0125_t2000','source':SMALL,'precision':'complex128','final':2000.,'dt':.0125,'minutes':[0,12],'kind':'finish_current'},
    {'name':'charge_J1_seeded_c64_L256_t3600','source':'charge_J1_seeded_c64_L256_t3000','final':3600.,'fork':True,'minutes':[4,6],'kind':'ring_late'},
    {'name':'charge_J1_seeded_c128_L256_t3600','source':'charge_J1_seeded_c128_L256_t3000','final':3600.,'fork':True,'minutes':[7,10],'kind':'ring_late'},
    {'name':'charge_J1_seeded_c128_dt0125_t3600','source':'charge_J1_seeded_c128_dt0125_t2000','final':3600.,'fork':True,'minutes':[40,55],'kind':'ring_time'},
    {'name':'charge_J1_seeded_c128_L384_t3600','source':'charge_J1_seeded_c128_L384_t3000','final':3600.,'fork':True,'minutes':[13,18],'kind':'ring_domain'},
    {'name':'charge_J1_radial_early_c128','source':'production_J1_ring','final':800.,'precision':'complex128','minutes':[8,12],'kind':'early_radial'},
    {'name':'charge_J1_radial_early_c64','source':'production_J1_ring','final':800.,'precision':'complex64','minutes':[4,6],'kind':'early_radial'},
    {'name':'check_C1_spot_domain_L384','source':'production_C1_spot','final':3000.,'space':{'N':768,'L':384.},'minutes':[70,90],'kind':'spot_domain'},
    {'name':'check_C1_spot_grid_N256','source':'production_C1_spot','final':3000.,'space':{'N':256,'L':256.},'minutes':[7,15],'kind':'spot_grid'},
    {'name':'check_C1_spot_late_dt0125','source':'production_C1_spot','final':3000.,'dt':.0125,'restart_time':2450.,'minutes':[16,22],'kind':'spot_time_late'},
]

def followup_save(value):write_json(FOLLOWUP,value)

def configuration(task):
    c=config_of(task['source']);c['time']['final']=task['final']
    if 'precision' in task:c['execution']['precision']=task['precision']
    if 'dt' in task:c['time']['dt']=task['dt']
    if 'space' in task:c['space']=task['space'].copy()
    return c

def late_restart(task,c):
    """A physical saved probe with matching accumulated balance and the new dt."""
    import hashlib
    from numerics import _precision_reference
    name=task['name'];dst=RUNS/name
    if (dst/'checkpoint.npz').exists():return
    source=RUNS/task['source'];old,store=_precision_reference(task['source'])
    t=task['restart_time'];ix=store.ordered(t,t)
    if len(ix)!=1:raise ValueError('Exact physical probe unavailable at restart time')
    state=store.state(ix[0]);newstep=round(t/c['time']['dt'])
    if not np.isclose(newstep*c['time']['dt'],t):raise ValueError('Restart time/dt mismatch')
    with np.load(source/'checkpoint.npz') as z:
        values={k:z[k].copy() for k in z.files}
    h=values['history'];selected=h[np.isclose(h[:,0],t,rtol=0,atol=1e-8)]
    if len(selected)!=1:raise ValueError('Missing exact balance row at physical restart')
    row=selected[0]
    # Copy before changing any checkpoint; prior results remain independent.
    fork_run(task['source'],name)
    signature=copy.deepcopy(c);signature['time'].pop('final')
    values.update(signature=np.array(hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()),
                  step=np.array(newstep),psi=state[0],active=state[1],inactive=state[2],elapsed=np.array(0.),
                  integral=np.array(row[7]),scale=np.array(row[8]),cond_integral=np.array(row[9]),cond_scale=np.array(row[10]),
                  history=h[h[:,0]<=t+1e-8],features=values['features'][values['features'][:,0]<=t+1e-8])
    values['probe_times'][values['probe_times']>t+1e-8]=-np.inf
    values['spatial_json']=np.array(json.dumps([v for v in json.loads(str(values['spatial_json'])) if v['time']<=t+1e-8]))
    np.savez(dst/'checkpoint.tmp.npz',**values);(dst/'checkpoint.tmp.npz').replace(dst/'checkpoint.npz')
    write_json(dst/'restart_provenance.json',{'source':task['source'],'time_ps':t,'new_dt_ps':c['time']['dt'],
        'state_from_exact_physical_probe':True,'balance_history_reused_through_restart':True,
        'scope':'late cycle accuracy only, not early mode selection',
        'state_sha256':[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in state]})

def early_diagnostics(name):
    from scipy.ndimage import map_coordinates
    from numerics import _precision_reference
    c,store=_precision_reference(name);g=make_grid(c)
    angles=2*np.pi*np.arange(1024)/1024;radius=8.
    coords=np.array([(radius*np.sin(angles)-g.y[0])/g.dx,(radius*np.cos(angles)-g.x[0])/g.dx])
    modes=np.arange(-8,9);basis=np.exp(-1j*modes[:,None]*angles)
    records=[]
    for i in store.ordered(0.,800.):
        if not np.isclose(store.times[i]/10,round(store.times[i]/10)):continue
        state=store.state(i);z=state[0];mirror_z=mirror((z,))[0]
        weights=[]
        for field in z:
            v=map_coordinates(field.real,coords,order=1)+1j*map_coordinates(field.imag,coords,order=1)
            coeff=np.mean(basis*v,axis=1)
            weights.append(abs(coeff)**2/max(float(np.mean(abs(v)**2)),1e-300))
        records.append({'time_ps':float(store.times[i]),'reflection_relative_L2':relative_norm(z,mirror_z),
                        'angular_field_power_fraction':np.asarray(weights)})
    result={'source':name,'radius_um':radius,'modes':modes,'records':records,
            'scope':'relative angular field powers on one fixed circle; not density harmonics or a complete mode basis',
            'replay_caveat':'not guaranteed bit-identical to the original compiled GPU trajectory'}
    write_json(DATA/(name+'_early_modes.json'),result)
    return {'file':name+'_early_modes.json','samples':len(records)}

def spatial_pair(a,b,ga,gb):
    # Compare on candidate coordinates when coarser, otherwise on reference coordinates.
    if np.isclose(ga.dx,gb.dx):
        if len(ga.x)>=len(gb.x):
            k=(len(ga.x)-len(gb.x))//2;n=len(gb.x)
            return tuple(v[:,k:k+n,k:k+n] for v in a),b,gb
    if np.isclose(ga.dx/gb.dx,round(ga.dx/gb.dx)) and ga.dx>=gb.dx and len(ga.x)*ga.dx==len(gb.x)*gb.dx:
        stride=round(ga.dx/gb.dx)
        if not np.allclose(ga.x,gb.x[::stride],rtol=0,atol=1e-12):raise ValueError('Nonmatching spatial nodes')
        return a,tuple(v[:,::stride,::stride] for v in b),ga
    if ga.dx<gb.dx and np.isclose(len(ga.x)*ga.dx,len(gb.x)*gb.dx):
        # Periodic Fourier projection onto the reference grid, including 768 -> 512.
        # Coordinates have the same physical origin and domain length.
        from scipy.signal import resample
        if not np.isclose(ga.x[0],gb.x[0]):raise ValueError('Different spatial origins')
        projected=tuple(resample(resample(v,len(gb.x),axis=-1),len(gb.y),axis=-2) for v in a)
        return projected,b,gb
    if a[0].shape==b[0].shape and np.isclose(ga.dx,gb.dx):return a,b,ga
    raise ValueError('Unsupported grid comparison')

def compare_late(candidate,reference):
    from scipy.optimize import minimize_scalar
    from numerics import _precision_reference,analyze_report3,_lag_polynomial_weights,_report3_maps,_report3_map_errors
    ca,sa=_precision_reference(candidate);cb,sb=_precision_reference(reference)
    ga=make_grid(ca);gb=make_grid(cb)
    with np.load(DATA/(candidate+'.npz')) as z:ha=z['history'].copy();fa=z['features'].copy();a=tuple(z[k].copy() for k in ('psi','active','inactive'))
    with np.load(DATA/(reference+'.npz')) as z:hb=z['history'].copy();fb=z['features'].copy();b=tuple(z[k].copy() for k in ('psi','active','inactive'))
    end=min(float(ha[-1,0]),float(hb[-1,0]))
    aa=analyze_report3(sa,ha,fa,end,ca);ab=analyze_report3(sb,hb,fb,end,cb)
    phase=0.;period_a=aa.get('period_ps');period_b=ab.get('period_ps')
    if period_a and period_b:
        left=max(end-2*max(aa['window_length_ps'],ab['window_length_ps']),float(fa[0,0]),float(fb[0,0]))
        use=(fa[:,0]>=left+period_b)&(fa[:,0]<=end-period_b)
        tt=fa[use,0];vv=fa[use,1:10];sc=np.maximum(np.mean(abs(fb[:,1:10]),axis=0),1.)
        if len(tt)>20:
            def cost(lag):
                target=np.column_stack([np.interp(tt+lag,fb[:,0],fb[:,j]) for j in range(1,10)])
                return float(np.mean(((vv-target)/sc)**2))
            trials=np.linspace(-period_b/2,period_b/2,81);j=int(np.argmin([cost(v) for v in trials]))
            fit=minimize_scalar(cost,bounds=(trials[max(0,j-1)],trials[min(80,j+1)]),method='bounded');phase=float(fit.x)
    af,bf,g=spatial_pair(a,b,ga,gb)
    result={'candidate':candidate,'reference':reference,'analysis_candidate':aa,'analysis_reference':ab,
            'final_same_time':_precision_metrics(af,bf,g),'single_cycle_shift_ps':phase,
            'period_relative_difference':abs(period_a/period_b-1) if period_a and period_b else None,
            'scope':'spatial comparisons use common coordinates; no independent component-phase alignment'}
    common_left=max(end-2*max(aa['window_length_ps'],ab['window_length_ps']),float(sa.times[sa.ordered()[0]]),float(sb.times[sb.ordered()[0]]))
    candidates=sa.ordered(common_left,end)
    candidates=candidates[np.unique(np.rint(np.linspace(0,len(candidates)-1,min(13,len(candidates)))).astype(int))]
    ts=sb.times[sb.ordered()];indices=sb.ordered();rows=[];max_error=np.zeros(9);max_population=np.zeros(2);max_stokes=np.zeros(3)
    for i in candidates:
        now=float(sa.times[i]);target=now+phase
        if target<ts[0] or target>ts[-1]:continue
        ix,w=_lag_polynomial_weights(ts,target,8)
        refmap=None;s0=None
        for k,weight in zip(ix,w):
            ac,bc,cg=spatial_pair(sa.state(i),sb.state(indices[k]),ga,gb)
            maps,S0=_report3_maps(bc)
            if refmap is None:refmap=np.zeros_like(maps);s0=np.zeros_like(S0)
            refmap+=weight*maps;s0+=weight*S0
        current,current_s0=_report3_maps(ac);err=_report3_map_errors(current,refmap,s0)
        a_pop=current[:2].sum((-2,-1))*cg.da;b_pop=refmap[:2].sum((-2,-1))*cg.da
        pop_error=abs(a_pop-b_pop)/np.maximum(abs(b_pop),max(float(abs(b_pop).sum())*1e-6,1e-30))
        mask=(s0>1e-3*s0.max())&(current_s0>1e-3*current_s0.max())
        if mask.any():
            delta=current[6:,mask]/current_s0[mask]-refmap[6:,mask]/s0[mask]
            stokes_error=np.sqrt(np.sum(delta**2*s0[mask],axis=1)/s0[mask].sum())
        else:stokes_error=np.full(3,np.inf)
        max_population=np.maximum(max_population,pop_error);max_stokes=np.maximum(max_stokes,stokes_error)
        rows.append({'time_ps':now,'cycle_aligned_map_errors':err,
                     'population_relative':pop_error,'normalized_stokes_weighted_RMS':stokes_error})
        max_error=np.maximum(max_error,err)
    result.update(shared_interval_ps=[common_left,end],map_samples=rows,max_cycle_aligned_map_errors=max_error,
                  max_cycle_aligned_population_errors=max_population,
                  max_cycle_aligned_normalized_stokes_RMS=max_stokes,review_required=True)
    return result

def followup_queue():
    previous=json.loads(FOLLOWUP.read_text()) if FOLLOWUP.exists() else {}
    if previous.get('status')=='COMPLETED':
        print('Follow-up already completed',flush=True);return
    r={'status':'WAITING_FOR_GPU','started_utc':previous.get('started_utc',stamp()),'pid':os.getpid(),
       'planned':FOLLOWUP_PLANS,'results':previous.get('results',{}),'thermal_stop':False,
       'policy':'finite list only; one producer; preserve sources; no automatic further experiments; scientific WARN retained'}
    followup_save(r)
    with (RUNS/'pilots.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Re-check the preceding queue only after its lock has been released.
        prior=json.loads(MANIFEST.read_text())
        if prior['status']=='FAILED':raise RuntimeError('Previous queue failed; inspect before follow-up')
        try:
            for task in FOLLOWUP_PLANS:
                name=task['name'];c=configuration(task)
                if name in r['results'] and r['results'][name]['summary']['run_status']=='RUN_COMPLETED':continue
                progress=RUNS/name/'summary.json'
                done=json.loads(progress.read_text()) if progress.exists() else {}
                already_done=done.get('run_status')=='RUN_COMPLETED' and done.get('time_ps')==task['final']
                r.update(status='RUNNING',active_case=name);followup_save(r)
                print('START',name,stamp(),flush=True);start=time.monotonic()
                if task.get('restart_time') is not None:late_restart(task,c)
                elif task.get('fork'):fork_run(task['source'],name)
                if already_done:result={'summary':done}
                else:result=run_report3_pilot(c,name,common_window=400. if task['kind']=='early_radial' else None)
                state,_=state_of(name)
                record={'summary':result['summary'],'topology':topology(state,c),
                        'additional_wall_s':time.monotonic()-start,'completed_utc':stamp(),'kind':task['kind']}
                if task['kind']=='early_radial':record['early_modes']=early_diagnostics(name)
                r['results'][name]=record;followup_save(r)
                if result['summary']['run_status']!='RUN_COMPLETED':
                    r.update(status='RUN_INCOMPLETE',finished_utc=stamp());followup_save(r);return
            r['status']='POSTPROCESSING';r.pop('active_case',None);followup_save(r)
            pairs=[('charge_J1_seeded_c64_L256_t3600','charge_J1_seeded_c128_L256_t3600'),
                   ('charge_J1_seeded_c128_dt0125_t3600','charge_J1_seeded_c128_L256_t3600'),
                   ('charge_J1_seeded_c128_L384_t3600','charge_J1_seeded_c128_L256_t3600'),
                   ('check_C1_spot_domain_L384','production_C1_spot'),
                   ('check_C1_spot_grid_N256','production_C1_spot'),
                   ('check_C1_spot_late_dt0125','production_C1_spot')]
            r['comparisons']={}
            for a,b in pairs:
                print('COMPARE',a,b,stamp(),flush=True)
                r['comparisons'][a+' vs '+b]=compare_late(a,b);followup_save(r)
            # Controlled seed histories are already retained in the original 2000 ps runs.
            r['controlled_early_modes']={n:early_diagnostics(n) for n in (SMALL,SINGLE)}
            r.update(status='COMPLETED',finished_utc=stamp());followup_save(r)
        except BaseException as e:
            r.update(status='FAILED',error=repr(e),finished_utc=stamp());followup_save(r);raise
    print('FOLLOWUP',r['status'],stamp(),flush=True)

def ring_grid_queue():
    from numerics import _ProbeStore
    manifest=DATA/'charge_grid_queue.json'
    reference='charge_J1_seeded_c128_L256_t3600'
    task={'name':'check_J1_ring_grid_N768_L256','source':reference,'precision':'complex128',
          'final':3600.,'space':{'N':768,'L':256.},'kind':'ring_grid','estimated_minutes':[90,120]}
    r={'status':'WAITING_FOR_GPU','pid':os.getpid(),'started_utc':stamp(),'planned':[task],
       'thermal_stop':False,'policy':'wait for current follow-up; one independent start; no automatic extensions',
       'comparison':'same L=256, dt=.025, explicit continuous seed; dx=.5 -> 1/3',
       'field_comparison':'periodic Fourier projection 768 -> 512; native populations checked separately',
       'full_complex_probe_cadence_ps':5.}
    write_json(manifest,r)
    try:
        with (RUNS/'pilots.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            prior=json.loads(FOLLOWUP.read_text())
            if prior['status']!='COMPLETED':raise RuntimeError('Preceding follow-up did not complete: '+prior['status'])
            c=configuration(task);name=task['name'];path=RUNS/name;path.mkdir(exist_ok=True)
            # Retain onset and late complex fields without forcing long analysis windows.
            if not (path/'probe_maps.npy').exists():
                store=_ProbeStore(path/'probe_maps.npy',c['space']['N'],duration=c['time']['final']+10.,cadence=c['save']['probe'])
                store.maps.flush();del store
            r.update(status='RUNNING',active_case=name,compute_started_utc=stamp());write_json(manifest,r)
            print('START',name,stamp(),flush=True)
            result=run_report3_pilot(c,name)
            state,_=state_of(name);ref,_=state_of(reference);g=make_grid(c);gr=make_grid(config_of(reference))
            r['result']={'summary':result['summary'],'topology':topology(state,c)}
            if result['summary']['run_status']!='RUN_COMPLETED':
                r.update(status='RUN_INCOMPLETE',finished_utc=stamp());write_json(manifest,r);return
            r.update(status='POSTPROCESSING');r.pop('active_case',None);write_json(manifest,r)
            r['comparison']=compare_late(name,reference)
            native=(abs(state[0])**2).sum(axis=(-2,-1),dtype=np.float64)*g.da
            refpop=(abs(ref[0])**2).sum(axis=(-2,-1),dtype=np.float64)*gr.da
            projected,_,_=spatial_pair(state,ref,g,gr)
            projected_pop=(abs(projected[0])**2).sum(axis=(-2,-1),dtype=np.float64)*gr.da
            r['native_populations']={'candidate':native,'reference':refpop,'relative_difference':abs(native/refpop-1),
                'projection_population_loss_relative':(native-projected_pop)/native}
            r.update(status='COMPLETED',finished_utc=stamp(),scientific_assessment='REVIEW_REQUIRED')
            write_json(manifest,r)
    except BaseException as e:
        r.update(status='FAILED',error=repr(e),finished_utc=stamp());write_json(manifest,r);raise
    print('RING GRID',r['status'],stamp(),flush=True)

if __name__=='__main__':
    if '--ring-grid' in sys.argv:ring_grid_queue()
    elif '--followup' in sys.argv:followup_queue()
    else:initial_queue()
