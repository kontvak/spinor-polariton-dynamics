"""Time integrators, diagnostics, resumable runs and F1–F5 research workflows."""
from __future__ import annotations
from pathlib import Path
import copy
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy import fft
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import yaml
from model import (ROOT, parameters, make_grid, injection, initial_state, seed_waves,
                   local_rhs, scalar_stationary, spin_stationary, stokes, report3_config)


def write_json(path, values):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2, default=convert, allow_nan=False)+'\n')
    temporary.replace(path)


def pack(state, p):
    psi, active, inactive = state
    return np.concatenate([psi.real.ravel(), psi.imag.ravel(), active.ravel()]
                          + ([inactive.ravel()] if p['model'] != 'M0' else []))


def unpack(values, p):
    n = p['spins']
    psi = values[:n]+1j*values[n:2*n]
    active = values[2*n:3*n]
    inactive = values[3*n:4*n] if p['model'] != 'M0' else np.zeros(n)
    return psi, active, inactive


def local_rk4(state, pump, p, dt):
    a = local_rhs(state, pump, p)
    b = local_rhs(tuple(v+dt*k/2 for v,k in zip(state,a)), pump, p)
    c = local_rhs(tuple(v+dt*k/2 for v,k in zip(state,b)), pump, p)
    d = local_rhs(tuple(v+dt*k for v,k in zip(state,c)), pump, p)
    return tuple(v+dt*(aa+2*bb+2*cc+dd)/6 for v,aa,bb,cc,dd in zip(state,a,b,c,d))


def solve_0d(config, *, method='DOP853', state0=None, start=0.0):
    """Integrate real components; stop and restart exactly at a pump switch."""
    p = parameters(config)
    state0 = initial_state(config) if state0 is None else state0
    final, sample = config['time']['final'], config['time']['sample']
    times = np.arange(round((final-start)/sample)+1)*sample+start
    if abs(times[-1]-final)>1e-9: raise ValueError('Sample step must divide duration')
    switch = config['pump']['off_time']
    ends = [start]+([switch] if switch is not None and start<switch<final else [])+[final]
    current = tuple(np.array(v, copy=True) for v in state0)
    output = np.empty((len(times), len(pack(current,p))))
    output[0] = pack(current,p)
    base_pump = injection(config)
    for left,right in zip(ends[:-1], ends[1:]):
        pump = base_pump if switch is None or left<switch else np.zeros_like(base_pump)
        indices = np.flatnonzero((times>left+1e-10)&(times<=right+1e-10))
        requested = times[indices]
        if method == 'DOP853':
            def rhs(t, y): return pack(local_rhs(unpack(y,p), pump, p),p)
            result = solve_ivp(rhs,(left,right),pack(current,p),dense_output=True,
                               method='DOP853',**config['ode'])
            if not result.success: raise RuntimeError(result.message)
            output[indices] = result.sol(requested).T
            current = unpack(result.y[:,-1],p)
        elif method == 'RK4':
            dt = config['time']['dt']
            count = round((right-left)/dt)
            if abs(count*dt-(right-left))>1e-9: raise ValueError('Step crosses pump switch')
            steps = {round((t-left)/dt):idx for t,idx in zip(requested,indices)}
            if any(abs(round((t-left)/dt)*dt-(t-left))>1e-8 for t in requested):
                raise ValueError('Sampling must align with fixed steps')
            for step in range(1,count+1):
                current = local_rk4(current,pump,p,dt)
                if step in steps: output[steps[step]]=pack(current,p)
        else: raise ValueError(method)
    n=p['spins']
    return {'time':times,'psi':output[:,:n]+1j*output[:,n:2*n],
            'active':output[:,2*n:3*n],
            'inactive':output[:,3*n:4*n] if p['model']!='M0' else np.zeros((len(times),n)),
            'config':config,'method':method}


def masked_unwrap(phase, valid):
    result = np.full(phase.shape,np.nan)
    indices = np.flatnonzero(valid)
    for part in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
        if part.size: result[part]=np.unwrap(phase[part])
    return result


def observables_0d(data):
    p=parameters(data['config']);psi=data['psi'];nc=abs(psi)**2
    active,inactive=data['active'],data['inactive']
    pump=np.broadcast_to(injection(data['config']),nc.shape).copy()
    switch=data['config']['pump']['off_time']
    if switch is not None:pump[data['time']>=switch]=0
    if p['spins']==1:
        energy=p['gC']*nc+p['gR']*(active+p['eta']*pump) if p['model']=='M0' else p['gC']*nc+p['gA']*active+p['gI']*inactive
    else:energy=p['alpha1']*nc+p['alpha2']*nc[:,::-1]+p['gA']*active+p['gI']*inactive
    reservoir_energy=energy-(p['gC']*nc if p['spins']==1 else p['alpha1']*nc+p['alpha2']*nc[:,::-1])
    feed=pump if p['model']=='M0' else p['W']*inactive
    external=pump if p['model']=='M0' else p['injection_ratio']*pump
    total=nc.sum(axis=1)+active.sum(axis=1)+(inactive.sum(axis=1) if p['model']!='M0' else 0)
    total_source=external.sum(axis=1)-p['gammaC']*nc.sum(axis=1)-p['gammaA']*active.sum(axis=1)-(p['gammaI']*inactive.sum(axis=1) if p['model']!='M0' else 0)
    scale=external.sum(axis=1)+p['gammaC']*nc.sum(axis=1)+p['gammaA']*active.sum(axis=1)+(p['gammaI']*inactive.sum(axis=1) if p['model']!='M0' else 0)
    result={'nc':nc,'active':active,'inactive':inactive,'feed':feed,'reservoir_energy':reservoir_energy,
            'total':total,'total_source':total_source,'balance_scale':scale,'energy':energy}
    if p['spins']==2:
        n=nc.sum(axis=1);valid=(n>1e-8)&(np.min(nc,axis=1)>1e-4*n)
        coherence=np.conj(psi[:,1])*psi[:,0]
        z=np.divide(nc[:,0]-nc[:,1],n,out=np.full_like(n,np.nan),where=n>1e-8)
        theta=masked_unwrap(np.angle(coherence),valid)
        IJ=2*p['J']/p['hbar']*coherence.imag
        result.update(dict(zip(('S0','S1','S2','S3'),stokes(psi,axis=1))))
        result.update(z=z,theta=theta,phase_valid=valid,IJ=IJ,
                      legacy_s1=np.divide(2*coherence.real,n,out=np.full_like(n,np.nan),where=n>1e-8),
                      legacy_s2=np.divide(2*coherence.imag,n,out=np.full_like(n,np.nan),where=n>1e-8),
                      SA=np.divide(active[:,0]-active[:,1],active.sum(axis=1),out=np.full_like(n,np.nan),where=active.sum(axis=1)>1e-8),
                      SI=np.divide(inactive[:,0]-inactive[:,1],inactive.sum(axis=1),out=np.full_like(n,np.nan),where=inactive.sum(axis=1)>1e-8),
                      plus_source=(p['R']*active[:,0]-p['gammaC'])*nc[:,0]+IJ)
    return result


def density_settling(values):
    values=np.asarray(values)
    if values.ndim==1: values=values[:,None]
    scale=np.maximum(np.mean(abs(values),axis=0),1e-8)
    span=np.ptp(values,axis=0)/scale
    half=len(values)//2
    drift=abs(values[:half].mean(axis=0)-values[half:].mean(axis=0))/scale
    return {'span_max':float(span.max()),'half_mean_drift_max':float(drift.max()),
            'passed':bool(span.max()<.005 and drift.max()<.005)}


def balance_0d(data, obs):
    """Integral balance with one-sided pump values at exact switching times."""
    t=data['time'];p=parameters(data['config'])
    switch=data['config']['pump']['off_time']
    borders=[t[0]]+([switch] if switch is not None and t[0]<switch<t[-1] else [])+[t[-1]]
    integral=0.;denominator=0.
    for left,right in zip(borders[:-1],borders[1:]):
        idx=(t>=left-1e-9)&(t<=right+1e-9)
        source=obs['total_source'][idx].copy();scale=obs['balance_scale'][idx].copy()
        if switch is not None and right==switch:
            amount=float(injection(data['config']).sum())*(1 if p['model']=='M0' else p['injection_ratio'])
            source[-1]+=amount;scale[-1]+=amount
        integral+=np.trapezoid(source,t[idx]);denominator+=np.trapezoid(scale,t[idx])
    delta=float(obs['total'][-1]-obs['total'][0]);error=abs(delta-integral)
    result={'absolute':float(error),'denominator':float(denominator),
            'relative':float(error/max(denominator,1e-30)),'population_change':delta}
    if p['spins']==2:
        src=obs['plus_source'];delta_plus=obs['nc'][-1,0]-obs['nc'][0,0]
        scale=np.trapezoid(abs((p['R']*obs['active'][:,0]-p['gammaC'])*obs['nc'][:,0])+abs(obs['IJ']),t)
        err=abs(delta_plus-np.trapezoid(src,t))
        result['plus']={'absolute':float(err),'denominator':float(scale),'relative':float(err/max(scale,1e-30))}
    result['status']='PASS' if result['relative']<1e-3 and result.get('plus',{}).get('relative',0)<1e-3 else 'FAIL'
    return result


def classify_spin(data, window=2000.):
    obs=observables_0d(data);t=data['time'];idx=t>=t[-1]-window-1e-9
    tt=t[idx];z=obs['z'][idx];theta=obs['theta'][idx]
    result={'status':'UNRESOLVED','window_ps':[float(tt[0]),float(tt[-1])]}
    if len(tt)<10 or not np.all(np.isfinite(z)):
        result['reason']='Incomplete density data';return result
    phase_valid=bool(np.all(obs['phase_valid'][idx]))
    n=np.column_stack((obs['nc'][idx],obs['active'][idx],obs['inactive'][idx]))
    result.update(mean_z=float(np.mean(z)),A_z=float(np.ptp(z)/2),
                  phase_slope=float(np.polyfit(tt,theta,1)[0]) if phase_valid else None,
                  phase_span=float(np.ptp(theta)) if phase_valid else None,
                  density_settling=density_settling(n))
    p=parameters(data['config'])
    if p['J']==0:
        result['status']='UNCOUPLED' if phase_valid and result['density_settling']['passed'] else 'UNRESOLVED'
        return result
    if (phase_valid and result['density_settling']['passed'] and abs(result['phase_slope'])<1e-4
            and result['phase_span']<.02 and result['A_z']<1e-3):
        result['status']='LOCKED_STEADY';return result
    peaks,_=find_peaks(z,prominence=max(result['A_z']*.25,1e-4))
    half=len(tt)//2
    if len(peaks)<3:
        result['reason']='No repeated peaks';return result
    candidate=float(np.mean(np.diff(tt[peaks])))
    periods=[]
    # A half-window can contain >2 periods but only two peak maxima.
    # Refine a return-time candidate using the observed z values, without detrending.
    for times,values in ((tt[:half],z[:half]),(tt[half:],z[half:])):
        def cost(period):
            select=times+period<=times[-1]
            residual=np.interp(times[select]+period,times,values)-values[select]
            return np.mean(residual**2)
        fit=minimize_scalar(cost,bounds=(.9*candidate,1.1*candidate),method='bounded',options={'xatol':1e-7})
        periods.append(float(fit.x))
    period=float(np.mean(periods))
    if (tt[-1]-tt[0])/period<5 or min(tt[half-1]-tt[0],tt[-1]-tt[half])/period<2:
        result['reason']='Fewer than five cycles or fewer than two per half';return result
    amplitudes=[float(np.ptp(part)/2) for part in (z[:half],z[half:])]
    good=tt+period<=tt[-1];base=tt[good];shifted=base+period
    original=n[good];returned=np.column_stack([np.interp(shifted,tt,column) for column in n.T])
    errors=np.linalg.norm(returned-original,axis=0)/np.maximum(np.linalg.norm(original,axis=0),1e-12)
    phase_error=None
    if phase_valid:
        phase=np.exp(1j*theta)
        returned_phase=np.interp(shifted,tt,phase.real)+1j*np.interp(shifted,tt,phase.imag)
        phase_error=float(np.sqrt(np.mean(abs(returned_phase-phase[good])**2)))
    stokes=np.column_stack((obs['legacy_s1'][idx],obs['legacy_s2'][idx],z))
    returned_stokes=np.column_stack([np.interp(shifted,tt,column) for column in stokes.T])
    stokes_error=float(np.sqrt(np.mean(np.sum((returned_stokes-stokes[good])**2,axis=1))))
    period_error=abs(periods[0]-periods[1])/period
    amplitude_error=abs(amplitudes[0]-amplitudes[1])/max(amplitudes)
    result.update(period_ps=period,cycles=float((tt[-1]-tt[0])/period),
                  half_periods_ps=periods,half_amplitudes=amplitudes,
                  period_half_difference=period_error,amplitude_half_difference=amplitude_error,
                  density_return_error=float(errors.max()),phase_return_error=phase_error,
                  stokes_return_error=stokes_error)
    if (period_error<.02 and amplitude_error<.05 and errors.max()<.01
            and (phase_error<.01 if phase_valid else stokes_error<.01) and result['A_z']>=1e-3):
        if phase_valid:
            result['status']='RUNNING_PHASE_PERIODIC' if abs(result['phase_slope'])*period>np.pi else 'BOUNDED_PHASE_PERIODIC'
        else:
            result['status']='PERIODIC_PHASE_MASKED'
            result['reason']='Density and Stokes cycle confirmed; relative phase masked below component-density cutoff'
    else:result['reason']='Cycle return or half-window consistency failed'
    if result['status'] in ('RUNNING_PHASE_PERIODIC','BOUNDED_PHASE_PERIODIC','PERIODIC_PHASE_MASKED'):
        previous_end=float(tt[-1]-10000.)
        previous=(t>=previous_end-window)&(t<=previous_end)
        if t[0]<=previous_end-window and np.count_nonzero(previous)>=10:
            old_amplitude=float(np.ptp(obs['z'][previous])/2)
            difference=abs(result['A_z']-old_amplitude)/max(result['A_z'],old_amplitude,1e-12)
            result.update(long_lag_ps=10000.,long_lag_amplitude=old_amplitude,long_lag_amplitude_difference=difference)
            if difference>.05:
                result['status']='UNRESOLVED'
                result['reason']='Amplitude still changes by more than 5% over a 10000 ps lag'
    return result


def case_available(path):
    path=Path(path)
    if path.exists() or (ROOT/'results/runs/0d'/path.stem/'observables.npz').exists():return True
    archive=ROOT/'results/data/report_data.npz'
    if archive.exists():
        with np.load(archive,allow_pickle=False) as z:return path.stem+'_time' in z
    return False


def save_0d(path, data, summary):
    """Detailed local run; the public portable archive is exported separately."""
    path=Path(path)
    if path.parent==ROOT/'results/data':
        path=ROOT/'results/runs/0d'/path.stem/'observables.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    obs=observables_0d(data);p=parameters(data['config'])
    np.savez_compressed(path,time=data['time'],psi=data['psi'],active=data['active'],inactive=data['inactive'],
                        config_json=json.dumps(data['config']),method=np.array(data['method']),
                        **{key:value for key,value in obs.items() if key not in ('active','inactive')})
    write_json(path.parent/'summary.json',summary)
    metadata={'units':{'length':'um','time':'ps','energy':'meV','density':'um^-2'},
              'method':data['method'],'derived':p,'PA':injection(data['config']).tolist(),
              'Pin':(injection(data['config'])*(1 if p['model']=='M0' else p['injection_ratio'])).tolist(),
              'versions':{n:importlib.metadata.version(n) for n in ('numpy','scipy','numba')},
              'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('model.py','numerics.py')},
              'metadata_note':'Resolved on save/export; numerical tolerances and states are from the stored trajectory.'}
    (path.parent/'resolved_config.yaml').write_text(yaml.safe_dump({**data['config'],'resolved':metadata},sort_keys=False))
    np.savez_compressed(path.parent/'checkpoint.npz',time=data['time'][-1],
                        **{k:data[k][-1] for k in ('psi','active','inactive')},
                        config_json=json.dumps(data['config']),method=data['method'])


def load_0d(path):
    path=Path(path)
    local=ROOT/'results/runs/0d'/path.stem/'observables.npz'
    source=path if path.exists() else local
    if source.exists():
        with np.load(source,allow_pickle=False) as z:
            return {**{k:z[k].copy() for k in ('time','psi','active','inactive')},
                    'config':json.loads(str(z['config_json'])),'method':str(z['method'])}
    prefix=path.stem+'_'
    with np.load(ROOT/'results/data/report_data.npz',allow_pickle=False) as z:
        return {**{k:z[prefix+k].copy() for k in ('time','psi','active','inactive')},
                'config':json.loads(str(z[prefix+'config'])),'method':str(z[prefix+'method'])}


def run_0d_series(*, final_spin=6000., force=False):
    """All prescribed independent 0D starts; retain every scan point."""
    from model import load_config
    directory=ROOT/'results/data';directory.mkdir(parents=True,exist_ok=True)
    for name,model,reservoir in [('M0','M0','R1'),('R1','M1','R1'),('R2','M1','R2')]:
        c=load_config('F1',model=model,reservoir=reservoir);file=directory/f'F1_{name}.npz'
        if not case_available(file) or force:
            data=solve_0d(c);obs=observables_0d(data)
            before=(data['time']>=900)&(data['time']<1000)
            summary={'model':model,'reservoir':reservoir,'balance':balance_0d(data,obs),
                     'before_switch':density_settling(np.column_stack((obs['nc'][before],obs['active'][before])))}
            save_0d(file,data,summary);print('Saved',file.name,flush=True)
    records=[]
    for epsilon in np.linspace(-1,1,21):
        c=load_config('F3',pump={'epsilon':float(epsilon)})
        file=directory/f'F3_eps_{epsilon:+.1f}.npz'
        if case_available(file) and not force:data=load_0d(file)
        else:data=solve_0d(c)
        obs=observables_0d(data);last=data['time']>=data['time'][-1]-500
        stationary=density_settling(np.column_stack((obs['nc'][last],obs['active'][last],obs['inactive'][last])))
        if not stationary['passed'] and data['time'][-1]<4000:
            cc=copy.deepcopy(c);cc['time']['final']=4000
            extra=solve_0d(cc,state0=tuple(data[k][-1] for k in ('psi','active','inactive')),start=float(data['time'][-1]))
            data={**data,**{k:np.concatenate((data[k],extra[k][1:])) for k in ('time','psi','active','inactive')},'config':cc}
            obs=observables_0d(data);last=data['time']>=3500
            stationary=density_settling(np.column_stack((obs['nc'][last],obs['active'][last],obs['inactive'][last])))
        ref=spin_stationary(2,float(epsilon),parameters(c))
        mean=obs['nc'][last].mean(axis=0)
        slopes=[float(np.polyfit(data['time'][last],np.unwrap(np.angle(data['psi'][last,j])),1)[0]) for j in range(2)]
        summary={'epsilon':float(epsilon),'status':'PASS' if stationary['passed'] else 'UNRESOLVED',
                 'density':mean,'z':float(obs['z'][last].mean()),'SA':float(obs['SA'][last].mean()),
                 'SI':float(obs['SI'][last].mean()),'energy_difference':-parameters(c)['hbar']*(slopes[0]-slopes[1]),
                 'density_error':float(np.max(abs(mean-ref[0])/ref[0])),
                 'analytic_density':ref[0],'analytic_energy_difference':float(ref[3][0]-ref[3][1]),
                 'settling':stationary,'balance':balance_0d(data,obs)}
        save_0d(file,data,summary);records.append(summary)
    write_json(directory/'F3_scan.json',records)
    run_spin_scan(np.linspace(0,.020,21),force=force,initial_final=final_spin)
    return run_spin_refinement(force=force)


def spin_scan_values():
    raw=yaml.safe_load((ROOT/'config.yaml').read_text())
    values=[*np.linspace(0,.020,21),*raw.get('scan_J_additional',[])]
    for left,right,step in raw.get('scan_J_refined_ranges_meV',[]):
        count=round((right-left)/step)
        if step<=0 or not np.isclose(left+count*step,right):raise ValueError('Invalid J refinement range')
        values.extend(left+np.arange(count+1)*step)
    values=sorted({round(float(v),8) for v in values})
    if len({f'{v:.5f}' for v in values})!=len(values):raise ValueError('J values collide in saved filenames')
    return values


def spin_summary(data,*,prefer_primary=True,analysis_end=None):
    """Analyze a preferred common window, with a late-window fallback if unresolved."""
    end=6000. if analysis_end is None else float(analysis_end)
    if analysis_end is not None:prefer_primary=True
    primary={**data,**{k:data[k][data['time']<=end] for k in ('time','psi','active','inactive')}}
    result=classify_spin(primary if prefer_primary else data)
    if prefer_primary and result['status']=='UNRESOLVED' and data['time'][-1]>end:result=classify_spin(data)
    obs=observables_0d(data);left,right=result['window_ps']
    mask=(data['time']>=left)&(data['time']<=right)
    z=obs['z'][mask];finite=z[np.isfinite(z)]
    result.setdefault('mean_z',float(finite.mean()) if finite.size else None)
    result.setdefault('A_z',float(np.ptp(finite)/2) if finite.size else None)
    result.setdefault('phase_slope',None);result.setdefault('phase_span',None)
    result.setdefault('density_settling',density_settling(np.column_stack((obs['nc'][mask],obs['active'][mask],obs['inactive'][mask]))))
    result['phase_valid_fraction']=float(np.mean(obs['phase_valid'][mask]))
    result.update(J=parameters(data['config'])['J'],final_ps=right,stored_final_ps=float(data['time'][-1]),
                  z_min=float(np.min(obs['z'][mask])),z_max=float(np.max(obs['z'][mask])),
                  density_mean=obs['nc'][mask].mean(axis=0),balance=balance_0d(data,obs),
                  minimum_active=float(data['active'].min()),minimum_inactive=float(data['inactive'].min()))
    result['plot_mean_z']=result['mean_z'];result['mean_phase_drift']=result['phase_slope']
    result['mean_method']='full observation window'
    if result['status'].endswith('PERIODIC') or result['status']=='PERIODIC_PHASE_MASKED':
        period=result['period_ps'];cycles=int(np.floor((right-left)/period))
        begin=right-cycles*period;t=data['time']
        selected=t[(t>begin)&(t<right)];sample=np.r_[begin,selected,right]
        result['plot_mean_z']=float(np.trapezoid(np.interp(sample,t,obs['z']),sample)/(right-begin))
        result['mean_window_ps']=[float(begin),right];result['mean_cycles']=cycles
        result['mean_method']='integer number of final cycles, trapezoid with interpolated endpoints'
        if result['phase_valid_fraction']==1.:
            tt=t[mask];theta=obs['theta'][mask];valid=tt+period<=right
            drift=(np.interp(tt[valid]+period,tt,theta)-theta[valid])/period
            result['mean_phase_drift']=float(np.mean(drift))
    return result


def run_spin_scan(J_values=None,*,force=False,initial_final=6000.,max_final=12000.,prefer_primary=True,analysis_end=None):
    """Independent F5 starts, optional exact-state extensions, merge every point."""
    from model import load_config
    directory=ROOT/'results/data';file=directory/'F5_scan.json'
    if not 6000<=initial_final<=max_final:raise ValueError('Require 6000 <= initial_final <= max_final')
    records={round(r['J'],8):r for r in json.loads(file.read_text())} if file.exists() else {}
    values=spin_scan_values() if J_values is None else sorted({round(float(v),8) for v in J_values})
    for J in values:
        start=time.perf_counter();case=directory/f'F5_J_{J:.5f}.npz'
        config=load_config('F5',physical={'J':J},time={'final':initial_final})
        cached=case_available(case) and not force
        data=load_0d(case) if cached else solve_0d(config)
        old=records.get(J,{}) if cached else {};attempts=list(old.get('analysis_attempts',[]))
        while True:
            result=spin_summary(data,prefer_primary=prefer_primary,analysis_end=analysis_end)
            attempt={k:result[k] for k in ('status','window_ps','mean_z','A_z','density_settling')}
            attempt.update(reason=result.get('reason'),long_lag_amplitude_difference=result.get('long_lag_amplitude_difference'))
            if not attempts or attempts[-1]!=attempt:attempts.append(attempt)
            result.update(analysis_attempts=attempts,
                          elapsed_wall_s=old.get('elapsed_wall_s',0.)+time.perf_counter()-start,
                          prefer_primary=prefer_primary,preferred_analysis_end_ps=analysis_end,
                          supplemental=round(J,8) not in {round(float(v),8) for v in np.linspace(0,.020,21)})
            save_0d(case,data,result)
            if data['time'][-1]>=initial_final and (result['status']!='UNRESOLVED' or data['time'][-1]>=max_final):break
            config=copy.deepcopy(data['config']);config['time']['final']=min(max_final,initial_final if data['time'][-1]<initial_final else 2*float(data['time'][-1]))
            extra=solve_0d(config,state0=tuple(data[k][-1] for k in ('psi','active','inactive')),start=float(data['time'][-1]))
            data={**data,**{k:np.concatenate((data[k],extra[k][1:])) for k in ('time','psi','active','inactive')},'config':config}
        records[J]=result
        write_json(file,[records[k] for k in sorted(records)])
        print(f'J={J:.5f}: {result["status"]}, z={result["mean_z"]:+.6f}, Az={result["A_z"]:.6g}, window={result["window_ps"]}',flush=True)
    # Extrema for all old and new points use the exact stored analysis window.
    for J,result in records.items():
        data=load_0d(directory/f'F5_J_{J:.5f}.npz');obs=observables_0d(data)
        left,right=result['window_ps'];mask=(data['time']>=left)&(data['time']<=right)
        result.update(z_min=float(obs['z'][mask].min()),z_max=float(obs['z'][mask].max()))
    output=[records[k] for k in sorted(records)];write_json(file,output)
    return output


def run_spin_refinement(*,force=False):
    """Dense independent starts with a common late window; reuse saved trajectories."""
    raw=yaml.safe_load((ROOT/'config.yaml').read_text());settings=raw.get('scan_refinement',{})
    rows=run_spin_scan(spin_scan_values(),force=force,prefer_primary=False,
                       analysis_end=settings.get('initial_final',12000.),
                       initial_final=settings.get('initial_final',12000.),
                       max_final=settings.get('max_final',18000.))
    if raw.get('scan_long_J'):
        rows=run_spin_scan(raw['scan_long_J'],prefer_primary=False,analysis_end=raw['scan_long_final'],
                           initial_final=raw['scan_long_final'],max_final=settings['max_final'])
    return rows


def run_refinement_checks(J_values=None):
    """Independent starts with twice tighter DOP853 tolerances and max_step=1 ps."""
    directory=ROOT/'results/data';results={}
    raw=yaml.safe_load((ROOT/'config.yaml').read_text())
    values=raw.get('scan_J_check',raw.get('scan_J_additional',[])) if J_values is None else J_values
    for J in values:
        start=time.perf_counter();ref=load_0d(directory/f'F5_J_{J:.5f}.npz')
        config=copy.deepcopy(ref['config']);config['ode']={'rtol':1e-11,'atol':1e-13,'max_step':1.}
        data=solve_0d(config)
        records=json.loads((directory/'F5_scan.json').read_text())
        row=next(r for r in records if np.isclose(r['J'],J,rtol=0,atol=1e-10))
        end=row['window_ps'][-1]
        baseline=spin_summary(ref,prefer_primary=False,analysis_end=end);other=spin_summary(data,prefer_primary=False,analysis_end=end)
        left,right=baseline['window_ps'];mask=(ref['time']>=left)&(ref['time']<=right)
        nr=abs(ref['psi'][mask])**2;nn=abs(data['psi'][mask])**2
        density=float(np.max(abs(nn.mean(axis=0)-nr.mean(axis=0))/nr.mean(axis=0)))
        short=ref['time']<=200
        field=float(np.max(abs(data['psi'][short]-ref['psi'][short]))/np.max(abs(ref['psi'][short])))
        metrics={'status':'PASS','classification_reference':baseline['status'],'classification_tighter':other['status'],
                 'mean_density_relative':density,'short_field_relative':field,'window_ps':baseline['window_ps'],
                 'elapsed_wall_s':time.perf_counter()-start,'ode_tighter':config['ode']}
        passed=other['status']==baseline['status'] and baseline['status']!='UNRESOLVED' and density<.005 and field<1e-7
        if baseline.get('period_ps') and other.get('period_ps'):
            metrics['amplitude_relative']=abs(other['A_z']-baseline['A_z'])/baseline['A_z']
            metrics['period_relative']=abs(other['period_ps']-baseline['period_ps'])/baseline['period_ps']
            passed=passed and max(metrics['amplitude_relative'],metrics['period_relative'])<.01
        elif baseline['status']=='LOCKED_STEADY':
            a=np.angle(ref['psi'][-1,0]*ref['psi'][-1,1].conj());b=np.angle(data['psi'][-1,0]*data['psi'][-1,1].conj())
            metrics['relative_phase_difference']=float(abs(np.angle(np.exp(1j*(a-b)))))
            passed=passed and metrics['relative_phase_difference']<.01
        metrics['status']='PASS' if passed else 'FAIL';results[f'J={J:.5f}']=metrics
        save_0d(directory/f'F5_check_J_{J:.5f}.npz',data,{'classification':other,'comparison':metrics})
        file=directory/'reference_checks.json';checks=json.loads(file.read_text()) if file.exists() else {}
        checks.setdefault('F5_refinement',{}).update(results);write_json(file,checks)
        print('Tighter F5 check',J,metrics['status'],flush=True)
    return results


def run_phase_sampling_checks():
    """Resolve fast phase changes near a vanishing component using 0.05 ps output."""
    directory=ROOT/'results/data';rows=json.loads((directory/'F5_scan.json').read_text());checks={}
    for J in (.00636,.00638,.00640,.00642):
        ref=load_0d(directory/f'F5_J_{J:.5f}.npz')
        row=next(r for r in rows if np.isclose(r['J'],J,rtol=0,atol=1e-10))
        left,right=row['window_ps'];index=int(np.searchsorted(ref['time'],left))
        if ref['time'][index]!=left:raise ValueError('Missing exact start of phase-check window')
        config=copy.deepcopy(ref['config']);config['time'].update(final=right,sample=.05)
        config['ode']={'rtol':1e-11,'atol':1e-13,'max_step':1.}
        fine=solve_0d(config,state0=tuple(ref[k][index] for k in ('psi','active','inactive')),start=left)
        result=spin_summary(fine,prefer_primary=False);obs=observables_0d(fine)
        take=(ref['time']>=left)&(ref['time']<=right);base=abs(ref['psi'][take])**2
        samples=np.rint((ref['time'][take]-left)/.05).astype(int);density=abs(fine['psi'][samples])**2
        err=float(np.linalg.norm(density-base)/np.linalg.norm(base))
        n=obs['nc'];fraction=n.min(axis=1)/n.sum(axis=1)
        phase=np.angle(fine['psi'][:,0]*fine['psi'][:,1].conj())
        step=float(np.max(abs(np.angle(np.exp(1j*np.diff(phase))))))
        period=abs(result['period_ps']-row['period_ps'])/row['period_ps']
        metrics={'status':'PASS' if result['status']==row['status'] and err<1e-7 and period<.01 else 'FAIL',
                 'classification':result['status'],'density_relative':err,'period_relative':period,
                 'sample_ps':.05,'window_ps':[left,right],'minimum_component_fraction':float(fraction.min()),
                 'max_phase_increment_rad':step,'phase_valid_fraction':result['phase_valid_fraction']}
        save_0d(directory/f'F5_sampling_J_{J:.5f}.npz',fine,{'classification':result,'comparison':metrics})
        checks[f'J={J:.5f}']=metrics;print('Phase sampling',J,metrics['status'],flush=True)
    file=directory/'reference_checks.json';report=json.loads(file.read_text());report['F5_phase_sampling']=checks
    write_json(file,report);return checks


def run_polarization_controls():
    """Test the largest sampled sign jump using state continuation and two phases."""
    from model import load_config
    directory=ROOT/'results/data';rows=json.loads((directory/'F5_scan.json').read_text())
    pairs=[(a,b) for a,b in zip(rows[:-1],rows[1:])
           if a['status']!='UNRESOLVED' and b['status']!='UNRESOLVED' and a['mean_z']*b['mean_z']<0]
    if not pairs:raise ValueError('No sampled stationary polarization reversal')
    left,right=max(pairs,key=lambda ab:abs(ab[0]['mean_z']-ab[1]['mean_z']))
    settings=yaml.safe_load((ROOT/'config.yaml').read_text()).get('scan_refinement',{})
    final=settings.get('initial_final',18000.);max_final=settings.get('max_final',36000.)
    controls=[];phases=[]
    phase_file=directory/'phase_seed_checks.json'
    phase_rows=json.loads(phase_file.read_text()) if phase_file.exists() else []
    def integrate(config,state0=None):
        data=solve_0d(config,state0=state0)
        while True:
            result=spin_summary(data,prefer_primary=False,analysis_end=final)
            if result['status']!='UNRESOLVED' or data['time'][-1]>=max_final:return data,result
            config=copy.deepcopy(data['config']);config['time']['final']=min(max_final,2*float(data['time'][-1]))
            extra=solve_0d(config,state0=tuple(data[k][-1] for k in ('psi','active','inactive')),start=float(data['time'][-1]))
            data={**data,**{k:np.concatenate((data[k],extra[k][1:])) for k in ('time','psi','active','inactive')},'config':config}
    for source,target in ((left,right),(right,left)):
        J=target['J'];original=load_0d(directory/f"F5_J_{source['J']:.5f}.npz")
        state=tuple(original[k][-1].copy() for k in ('psi','active','inactive'))
        config=load_config('F5',physical={'J':J},time={'final':final})
        data,result=integrate(config,state0=state)
        name=f"F5_continue_J_{J:.5f}_from_{source['J']:.5f}"
        result.update(case=name,source_J=source['J'],source_z=source['mean_z'],
                      opposite_sign_to_direct=bool(result['mean_z']*target['mean_z']<0))
        save_0d(directory/(name+'.npz'),data,result);controls.append(result)
        print('Polarization continuation',source['J'],'->',J,result['status'],result['mean_z'],flush=True)
    for target in (left,right):
        J=target['J']
        for theta in (1.,2.):
            config=load_config('F5',physical={'J':J},initial={'theta':theta},time={'final':final})
            data,result=integrate(config)
            result.update(initial_theta=theta,protocol='independent start near polarization reversal')
            name=f'phase_J_{J:.5f}_theta_{theta:g}'
            save_0d(directory/(name+'.npz'),data,result);phases.append(result)
            phase_rows=[r for r in phase_rows if not (np.isclose(r['J'],J,rtol=0,atol=1e-10) and r['initial_theta']==theta)]
            phase_rows.append(result);write_json(phase_file,phase_rows)
            print('Polarization initial phase',J,theta,result['status'],result['mean_z'],flush=True)
    report={'jump_interval_meV':[left['J'],right['J']],'continuations':controls,'initial_phases':phases,
            'coexistence_both_sides':all(r['status']!='UNRESOLVED' and r['opposite_sign_to_direct'] for r in controls)}
    file=directory/'reference_checks.json';checks=json.loads(file.read_text())
    checks['F5_polarization_controls']=report;write_json(file,checks)
    return report


# This compiled scalar implementation is cross-checked against local_rk4.
# fastmath is deliberately disabled: no reassociation or precision reduction.
from numba import njit, prange, set_num_threads


@njit(cache=True, inline="always")
def _point_rhs(z,a,b,pump,coeff,m0):
    h,gc,gr,ga,gi,eta,r,lc,la,li,w,ratio=coeff
    n=z.real*z.real+z.imag*z.imag
    energy=gc*n+(gr*(a+eta*pump) if m0 else ga*a+gi*b)
    frequency=energy/h
    gain=(r*a-lc)/2
    dz=complex(gain*z.real+frequency*z.imag,gain*z.imag-frequency*z.real)
    da=pump-(la+r*n)*a if m0 else w*b-(la+r*n)*a
    db=0. if m0 else ratio*pump-(li+w)*b
    source=(pump if m0 else ratio*pump)-lc*n-la*a-(0. if m0 else li*b)
    scale=(pump if m0 else ratio*pump)+lc*n+la*a+(0. if m0 else li*b)
    return dz,da,db,source,scale


@njit(cache=True, parallel=True, nogil=True)
def _scalar_kernel(z,a,b,pump,coeff,dt,m0,sources,scales):
    half=dt*.5;sixth=dt/6
    for j in prange(z.size):
        zz,aa,bb,pp=z[j],a[j],b[j],pump[j]
        k1=_point_rhs(zz,aa,bb,pp,coeff,m0)
        k2=_point_rhs(zz+half*k1[0],aa+half*k1[1],bb+half*k1[2],pp,coeff,m0)
        k3=_point_rhs(zz+half*k2[0],aa+half*k2[1],bb+half*k2[2],pp,coeff,m0)
        k4=_point_rhs(zz+dt*k3[0],aa+dt*k3[1],bb+dt*k3[2],pp,coeff,m0)
        z[j]=zz+sixth*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        a[j]=aa+sixth*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        b[j]=bb+sixth*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        sources[j]=dt*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6
        scales[j]=dt*(k1[4]+2*k2[4]+2*k3[4]+k4[4])/6


def _scalar_local_inplace(z,a,b,pump,coeff,dt,m0):
    # Keep reductions outside the point kernel so independent updates can be
    # vectorized without reassociating floating-point expressions (no fastmath).
    sources=np.empty(z.size);scales=np.empty(z.size)
    _scalar_kernel(z,a,b,pump,coeff,dt,m0,sources,scales)
    return float(sources.sum()),float(scales.sum()),float(a.min()),float(b.min())


def scalar_coefficients(p):
    return tuple(float(p[k]) for k in ('hbar','gC','gR','gA','gI','eta','R',
                                        'gammaC','gammaA','gammaI','W','injection_ratio'))



def spin_coefficients(p):
    return tuple(float(p[k]) for k in ('hbar','alpha1','alpha2','gA','gI','R',
                 'gammaC','gammaA','gammaI','W','injection_ratio','GammaA','GammaI','J'))


@njit(cache=True, inline="always")
def _spin_point_rhs(z0,z1,a0,a1,b0,b1,p0,p1,c):
    h,u,v,ga,gi,r,lc,la,li,w,ratio,sa,si,j=c
    n0=z0.real*z0.real+z0.imag*z0.imag
    n1=z1.real*z1.real+z1.imag*z1.imag
    e0=(u*n0+v*n1+ga*a0+gi*b0)/h
    e1=(u*n1+v*n0+ga*a1+gi*b1)/h
    d0=((r*a0-lc)/2-1j*e0)*z0+1j*j/h*z1
    d1=((r*a1-lc)/2-1j*e1)*z1+1j*j/h*z0
    da0=w*b0-(la+r*n0)*a0+sa*(a1-a0)
    da1=w*b1-(la+r*n1)*a1+sa*(a0-a1)
    db0=ratio*p0-(li+w)*b0+si*(b1-b0)
    db1=ratio*p1-(li+w)*b1+si*(b0-b1)
    incoming=ratio*(p0+p1)
    outgoing=lc*(n0+n1)+la*(a0+a1)+li*(b0+b1)
    return d0,d1,da0,da1,db0,db1,incoming-outgoing,incoming+outgoing


@njit(cache=True, parallel=True, nogil=True)
def _spin_kernel(z,a,b,pump,c,dt,sources,scales):
    half=dt/2;sixth=dt/6
    for k in prange(z.shape[1]):
        z0,z1,a0,a1,b0,b1=z[0,k],z[1,k],a[0,k],a[1,k],b[0,k],b[1,k]
        p0,p1=pump[0,k],pump[1,k]
        q=_spin_point_rhs(z0,z1,a0,a1,b0,b1,p0,p1,c)
        r=_spin_point_rhs(z0+half*q[0],z1+half*q[1],a0+half*q[2],a1+half*q[3],
                          b0+half*q[4],b1+half*q[5],p0,p1,c)
        t=_spin_point_rhs(z0+half*r[0],z1+half*r[1],a0+half*r[2],a1+half*r[3],
                          b0+half*r[4],b1+half*r[5],p0,p1,c)
        u=_spin_point_rhs(z0+dt*t[0],z1+dt*t[1],a0+dt*t[2],a1+dt*t[3],
                          b0+dt*t[4],b1+dt*t[5],p0,p1,c)
        z[0,k]=z0+sixth*(q[0]+2*r[0]+2*t[0]+u[0])
        z[1,k]=z1+sixth*(q[1]+2*r[1]+2*t[1]+u[1])
        a[0,k]=a0+sixth*(q[2]+2*r[2]+2*t[2]+u[2])
        a[1,k]=a1+sixth*(q[3]+2*r[3]+2*t[3]+u[3])
        b[0,k]=b0+sixth*(q[4]+2*r[4]+2*t[4]+u[4])
        b[1,k]=b1+sixth*(q[5]+2*r[5]+2*t[5]+u[5])
        sources[k]=sixth*(q[6]+2*r[6]+2*t[6]+u[6])
        scales[k]=sixth*(q[7]+2*r[7]+2*t[7]+u[7])
        if not (np.isfinite(z[0,k].real) and np.isfinite(z[0,k].imag)
                and np.isfinite(z[1,k].real) and np.isfinite(z[1,k].imag)):
            sources[k]=np.nan


def _spin_local_inplace(z,a,b,pump,coeff,dt,workspace=None):
    sources,scales=(np.empty(z.size//2),np.empty(z.size//2)) if workspace is None else workspace
    _spin_kernel(z.reshape(2,-1),a.reshape(2,-1),b.reshape(2,-1),
                 pump.reshape(2,-1),coeff,dt,sources,scales)
    return float(sources.sum()),float(scales.sum()),float(a.min()),float(b.min())

def kinetic_factor(grid, p, tau):
    """Exact exp(-i H_k tau/hbar); tuple entries are U00, U01, U10.

    H_k=K*k^2 I + beta [[0,(kx-i*ky)^2],[(kx+i*ky)^2,0]].
    np.sinc uses sin(pi*x)/(pi*x); no division by k^2 is used.
    """
    phase=np.exp(-1j*p['K']*grid.k2*tau/p['hbar'])
    beta=p.get('beta',0.)
    if beta==0.:return phase
    if p['spins']!=2:raise ValueError('TE-TM requires two components')
    u=beta*grid.k2*tau/p['hbar']
    off=-1j*tau/p['hbar']*np.sinc(u/np.pi)*beta
    xy=grid.kx[None,:]-1j*grid.ky[:,None]
    return phase*np.cos(u),phase*off*xy**2,phase*off*xy.conj()**2


def kinetic_step(psi,factor,workers=1):
    transformed=fft.fft2(psi,axes=(-2,-1),workers=workers)
    if isinstance(factor,tuple):
        diagonal,upper,lower=factor
        plus,minus=transformed
        transformed=np.stack((diagonal*plus+upper*minus,
                              lower*plus+diagonal*minus))
    else:transformed*=factor
    return fft.ifft2(transformed,axes=(-2,-1),workers=workers,overwrite_x=True)


def split_step(state,pump,p,dt,grid,*,workers=1,accelerated=True):
    factor=kinetic_factor(grid,p,dt/2)
    state=(kinetic_step(state[0],factor,workers),state[1].copy(),state[2].copy())
    if accelerated and p['spins']==1:
        z,a,b=(np.ascontiguousarray(v) for v in state)
        _scalar_local_inplace(z.ravel(),a.ravel(),b.ravel(),pump.ravel(),
                              scalar_coefficients(p),dt,p['model']=='M0')
        state=z,a,b
    elif accelerated:
        state=tuple(np.ascontiguousarray(v) for v in state)
        _spin_local_inplace(*state,pump,spin_coefficients(p),dt)
    else:state=local_rk4(state,pump,p,dt)
    return kinetic_step(state[0],factor,workers),state[1],state[2]


def relative_norm(a,b):
    return float(np.sqrt(np.sum(abs(a-b)**2)/max(float(np.sum(abs(b)**2)),1e-30)))


def aligned_field_error(field,reference):
    overlap=np.sum(np.conj(reference)*field)
    return relative_norm(field*np.exp(-1j*np.angle(overlap)),reference),float(np.angle(overlap))


def field_gradients(psi,grid):
    transformed=fft.fft2(psi,axes=(-2,-1))
    kx,ky=grid.kx.copy(),grid.ky.copy();kx[len(kx)//2]=0;ky[len(ky)//2]=0
    return fft.ifft2(1j*kx*transformed),fft.ifft2(1j*ky[:,None]*transformed)


def vortex_diagnostics(psi,grid,p,radii=(4.,6.,7.5,9.,12.),n_angles=1024):
    from scipy.ndimage import map_coordinates
    if psi.ndim==3:psi=psi[0]
    n=abs(psi)**2;total=n.sum()*grid.da
    dx,dy=field_gradients(psi,grid)
    moment=np.sum(np.conj(psi)*(-1j)*(grid.x[None,:]*dy-grid.y[:,None]*dx))*grid.da
    contours=[];angles=np.linspace(0,2*np.pi,n_angles,endpoint=False)
    for radius in radii:
        xx=radius*np.cos(angles);yy=radius*np.sin(angles)
        if min(xx.min(),yy.min())<grid.x[0] or max(xx.max(),yy.max())>grid.x[-1]:continue
        coords=np.array([(yy-grid.y[0])/grid.dx,(xx-grid.x[0])/grid.dx])
        values=(map_coordinates(psi.real,coords,order=1,mode='nearest')
                +1j*map_coordinates(psi.imag,coords,order=1,mode='nearest'))
        increments=np.angle(np.roll(values,-1)*values.conj());raw=float(increments.sum()/(2*np.pi))
        density=float(np.min(abs(values)**2)/max(float(n.max()),1e-30))
        valid=density>=.05 and np.max(abs(increments))<np.pi/2 and abs(raw-round(raw))<1e-6
        contours.append({'radius_um':radius,'raw':raw,'density_min_rel':density,
                         'q':int(round(raw)) if valid else None,'status':'VALID' if valid else 'UNDEFINED'})
    charges=[c['q'] for c in contours if c['status']=='VALID']
    q=charges[0] if len(charges)>=2 and len(set(charges))==1 else None
    return {'q':q,'status':'DEFINED' if q is not None else 'UNDEFINED','contours':contours,
            'Lz_over_hbar_NC':float(moment.real/max(total,1e-30)),
            'Lz_imag_over_hbar_NC':float(moment.imag/max(total,1e-30))}


def populations(state,p,grid):
    return np.array([np.sum(abs(state[0])**2),np.sum(state[1]),
                     np.sum(state[2]) if p['model']!='M0' else 0.])*grid.da



def component_populations(state,grid):
    return np.array([np.sum(v,axis=(-2,-1))*grid.da
                     for v in (abs(state[0])**2,state[1],state[2])])


def _history_row(t,state,p,grid,integral,scale):
    row=[t,*populations(state,p,grid),integral,scale]
    return row+(component_populations(state,grid).ravel().tolist() if p['spins']==2 else [])


def _spin_stationarity(frames,history,p):
    rows=np.asarray(history);last=rows[rows[:,0]>=rows[-1,0]-100-1e-8]
    if last.shape[1]!=12:raise ValueError('Spin stationarity requires individual component histories')
    reference=frames[0][1];components=[]
    for sigma,label in enumerate(('+','-')):
        values=last[:,6+np.array([sigma,2+sigma,4+sigma])]
        if values[:,0].max()<1e-3:
            components.append({'component':label,'status':'LOW_SIGNAL'});continue
        settling=density_settling(values)
        maps=[abs(reference[0][sigma])**2,reference[1][sigma],reference[2][sigma]]
        errors=np.zeros(3)
        for _,state in frames[1:]:
            current=[abs(state[0][sigma])**2,state[1][sigma],state[2][sigma]]
            errors=np.maximum(errors,[relative_norm(a,b) for a,b in zip(current,maps)])
        passed=settling['passed'] and float(errors.max())<.01
        components.append({'component':label,'status':'PASS' if passed else 'UNRESOLVED',
                           'populations':settling,'density_map':float(errors[0]),
                           'active_map':float(errors[1]),'inactive_map':float(errors[2])})
    phase_required=p['J']!=0. or p.get('beta',0.)!=0.
    field_error=max(aligned_field_error(state[0],reference[0])[0] for _,state in frames[1:]) if phase_required else None
    passed=all(c['status']=='PASS' for c in components) and (not phase_required or field_error<.01)
    return {'status':'PASS' if passed else 'UNRESOLVED',
            'components':components,'total_populations':density_settling(last[:,1:4]),
            'phase_required':phase_required,'aligned_field':field_error,
            'window_ps':[float(last[0,0]),float(last[-1,0])]}

def spatial_checks(state,grid,p):
    densities=abs(state[0])**2;density=densities.sum(axis=0);total=float(density.sum())
    length=grid.dx*len(grid.x);edge=(abs(grid.x[None,:])>=.4*length)|(abs(grid.y[:,None])>=.4*length)
    transformed=fft.fft2(state[0],axes=(-2,-1));power=abs(transformed)**2
    high=(abs(grid.kx[None,:])>.8*np.pi/grid.dx)|(abs(grid.ky[:,None])>.8*np.pi/grid.dx)
    result={'edge_mass':float(density[edge].sum()/max(total,1e-30)),
            'high_k':float(power[:,high].sum()/max(float(power.sum()),1e-30)),
            'minimum_active':float(state[1].min()),'minimum_inactive':float(state[2].min()),
            'peak_density':float(density.max())}
    if p['spins']==2:
        result['components']=[{'edge_mass':float(d[edge].sum()/max(float(d.sum()),1e-30)),
                               'high_k':float(q[high].sum()/max(float(q.sum()),1e-30))}
                              for d,q in zip(densities,power)]
    return result


def _stationarity(frames,history,p):
    if len(frames)<2 or frames[-1][0]-frames[0][0]<100-1e-8:
        return {'status':'NOT_TESTED'}
    if p['spins']==2:return _spin_stationarity(frames,history,p)
    rows=np.asarray(history);last=rows[rows[:,0]>=rows[-1,0]-100-1e-8]
    # Densities and populations must be significant before normalization.
    if last[:,1].max()<1e-3:return {'status':'LOW_SIGNAL'}
    quantities=density_settling(last[:,1:3+(p['model']!='M0')])
    reference=frames[0][1]
    nref=np.sum(abs(reference[0])**2,axis=0);nref=nref/max(nref.sum(),1e-30)
    density_error=0.;field_error=0.;active_error=0.;inactive_error=0.
    for _,state in frames[1:]:
        n=np.sum(abs(state[0])**2,axis=0);n=n/max(n.sum(),1e-30)
        density_error=max(density_error,relative_norm(n,nref))
        field_error=max(field_error,aligned_field_error(state[0],reference[0])[0])
        active_error=max(active_error,relative_norm(state[1],reference[1]))
        if p['model']!='M0':inactive_error=max(inactive_error,relative_norm(state[2],reference[2]))
    passed=quantities['passed'] and max(density_error,field_error,active_error,inactive_error)<.01
    return {'status':'PASS' if passed else 'UNRESOLVED','populations':quantities,
            'density_shape':density_error,'aligned_field':field_error,
            'active':active_error,'inactive':inactive_error}


def run_2d(config,name,*,resume=True,max_steps=None,initial=None):
    """Deterministic Strang/RK4 run with full-precision restart and dense scalars."""
    path=ROOT/'results/runs'/name;path.mkdir(parents=True,exist_ok=True)
    p=parameters(config);grid=make_grid(config);pump=injection(config,grid)
    dt=config['time']['dt'];total_steps=round(config['time']['final']/dt)
    if abs(total_steps*dt-config['time']['final'])>1e-8:raise ValueError('Invalid final time')
    signature=copy.deepcopy(config);signature['time'].pop('final')
    signature=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    checkpoint=path/'checkpoint.npz';history=[];diagnostics=[];frames=[];windows=[]
    cores=[];core_times=[];start_step=0;elapsed_previous=0.;integral=0.;scale=0.
    minimum_a=0.;minimum_i=0.;stage='RUNNING'
    state=initial_state(config,grid) if initial is None else tuple(np.array(v,copy=True) for v in initial)
    if any(v.shape!=(p["spins"],len(grid.y),len(grid.x)) for v in state):raise ValueError("Initial state shape mismatch")
    initial_total=float(populations(state,p,grid).sum())
    if checkpoint.exists() and resume:
        old=json.loads((path/'run_metadata.json').read_text())
        if old['signature']!=signature:raise ValueError('Checkpoint configuration changed')
        with np.load(checkpoint,allow_pickle=False) as z:
            state=tuple(z[k].copy() for k in ('psi','active','inactive'))
            start_step=int(z['step']);integral=float(z['integral']);scale=float(z['scale'])
            initial_total=float(z['initial_total']);elapsed_previous=float(z['elapsed'])
            history=z['history'].tolist();diagnostics=json.loads(str(z['diagnostics_json']))
            windows=json.loads(str(z['windows_json']))
            minimum_a=float(z['minimum_active']);minimum_i=float(z['minimum_inactive'])
            if 'frame_times' in z:
                for j,t in enumerate(z['frame_times']):frames.append((float(t),tuple(z[k][j].copy() for k in ('frame_psi','frame_active','frame_inactive'))))
        core_file=path/'core_density.npz'
        if core_file.exists():
            with np.load(core_file) as z:
                valid=z['time']<=start_step*dt+1e-9
                core_times=z['time'][valid].tolist();cores=list(z['density'][valid])
    if start_step>total_steps:raise ValueError('Requested time precedes checkpoint')
    def snapshot(step):
        folder=path/'snapshots';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f't_{step*dt:07.1f}.npz',time=step*dt,
                            psi=state[0],active=state[1],inactive=state[2],x=grid.x,y=grid.y)
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','numba')}
    metadata={'signature':signature,'versions':versions,'python':platform.python_version(),
              'units':{'length':'um','time':'ps','energy':'meV','density':'um^-2'},
              'method':'Strang kinetic half / joint RK4 / kinetic half',
              'dtype':'complex128 / float64','seed_waves':seed_waves(config).tolist(),
              'initial_kind':'continuous_seed' if initial is None else 'provided_full_state',
              'derived':p,'PA_peak':np.max(pump,axis=(-2,-1)).tolist(),
              'Pin_peak':(np.max(pump,axis=(-2,-1))*(1 if p['model']=='M0' else p['injection_ratio'])).tolist()}
    write_json(path/'run_metadata.json',metadata)
    (path/'resolved_config.yaml').write_text(yaml.safe_dump({**config,'resolved':metadata},sort_keys=False))
    if start_step==0:
        snapshot(0);history=[_history_row(0.,state,p,grid,0.,0.)]
        frames=[(0.,tuple(v.copy() for v in state))]
    factor=kinetic_factor(grid,p,dt/2)
    workers=config['execution']['fft_workers'];coeff=scalar_coefficients(p) if p['spins']==1 else spin_coefficients(p)
    workspace=(np.empty(grid.k2.size),np.empty(grid.k2.size)) if p['spins']==2 else None
    set_num_threads(config['execution'].get('local_workers',4))
    ix=np.flatnonzero(abs(grid.x)<=40);iy=ix.copy()
    sample_steps=round(config['save']['observable']/dt)
    core_steps=round(config['save']['core']/dt)
    probe_steps=round(5./dt)
    diag_steps=round(config['save']['diagnostic']/dt)
    checkpoint_steps=round(config['save']['checkpoint']/dt)
    full_times={0.,200.,500.,1000.,2000.,4000.,config['time']['final']}
    start=time.perf_counter();step=start_step
    import signal, threading
    interrupted=False;previous_handler=None
    def stop_at_boundary(signum, frame):
        nonlocal interrupted
        interrupted=True
    if threading.current_thread() is threading.main_thread():
        previous_handler=signal.signal(signal.SIGINT,stop_at_boundary)
    def save_checkpoint():
        elapsed=elapsed_previous+time.perf_counter()-start
        data={'step':step,'psi':state[0],'active':state[1],'inactive':state[2],
              'integral':integral,'scale':scale,'initial_total':initial_total,'elapsed':elapsed,
              'history':np.asarray(history),'diagnostics_json':json.dumps(diagnostics),
              'windows_json':json.dumps(windows),'minimum_active':minimum_a,'minimum_inactive':minimum_i,
              'frame_times':np.array([f[0] for f in frames])}
        for j,k in enumerate(('frame_psi','frame_active','frame_inactive')):
            data[k]=np.array([f[1][j] for f in frames])
        tmp=path/'checkpoint.tmp.npz';np.savez_compressed(tmp,**data);tmp.replace(checkpoint)
        np.savez_compressed(path/'observables.npz',history=np.asarray(history),
                            columns=np.array(['time','NC','NA','NI','balance_integral','balance_scale']
                                +(['NC_plus','NC_minus','NA_plus','NA_minus','NI_plus','NI_minus'] if p['spins']==2 else [])))
        if cores:np.savez_compressed(path/'core_density.npz',time=np.array(core_times),density=np.array(cores),x=grid.x[ix],y=grid.y[iy])
    if start_step==0:
        core_times=[0.];cores=[np.sum(abs(state[0][:,iy[:,None],ix])**2,axis=0).astype(np.float32)]
    stop=total_steps if max_steps is None else min(total_steps,start_step+max_steps)
    try:
        for step in range(start_step+1,stop+1):
            field=kinetic_step(state[0],factor,workers)
            if p['spins']==1:
                field=np.ascontiguousarray(field)
                change,normalizer,ma,mi=_scalar_local_inplace(
                    field.ravel(),state[1].ravel(),state[2].ravel(),pump.ravel(),coeff,dt,p['model']=='M0')
                minimum_a=min(minimum_a,ma);minimum_i=min(minimum_i,mi)
                state=(field,state[1],state[2])
            else:
                field=np.ascontiguousarray(field)
                change,normalizer,ma,mi=_spin_local_inplace(
                    field,state[1],state[2],pump,coeff,dt,workspace)
                minimum_a=min(minimum_a,ma);minimum_i=min(minimum_i,mi)
                state=(field,state[1],state[2])
            if not np.isfinite(change+normalizer) or min(minimum_a,minimum_i)<-1e-10:
                raise FloatingPointError('Nonfinite state or negative reservoir; no clipping is applied')
            integral+=change*grid.da;scale+=normalizer*grid.da
            state=(kinetic_step(state[0],factor,workers),state[1],state[2])
            t=step*dt
            if step%sample_steps==0:
                history.append(_history_row(t,state,p,grid,integral,scale))
            if step%probe_steps==0:
                frames.append((t,tuple(v.copy() for v in state)))
                frames=[row for row in frames if row[0]>=t-100-1e-8]
            if step%core_steps==0:
                core_times.append(t);cores.append(np.sum(abs(state[0][:,iy[:,None],ix])**2,axis=0).astype(np.float32))
            if step%diag_steps==0:
                diagnostics.append({'time':t,**spatial_checks(state,grid,p)})
            if abs(t/100-round(t/100))<1e-9:
                windows.append({'time':t,**_stationarity(frames,history,p)})
            if any(abs(t-value)<1e-9 for value in full_times):snapshot(step)
            if step%checkpoint_steps==0:
                save_checkpoint()
                print(name,f'{t:g}/{config["time"]["final"]:g} ps',
                      f'{elapsed_previous+time.perf_counter()-start:.1f} s',flush=True)
            if interrupted:break
        stage='RUN_COMPLETED' if step==total_steps else 'RUN_INCOMPLETE'
    except KeyboardInterrupt:
        stage='RUN_INCOMPLETE'
    except Exception as error:
        stage='RUN_FAILED';write_json(path/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        save_checkpoint()
        pop=populations(state,p,grid)
        balance_error=abs(float(pop.sum())-initial_total-integral)
        latest=_stationarity(frames,history,p)
        successful=len(windows)>=2 and all(w['status']=='PASS' for w in windows[-2:])
        late=[d for d in diagnostics if d['time']>=step*dt-100]
        summary={'run_status':stage,'time_ps':step*dt,'elapsed_wall_s':elapsed_previous+time.perf_counter()-start,
                 'NC':pop[0],'NA':pop[1],'NI':pop[2],
                 'balance':{'absolute':balance_error,'denominator':scale,'relative':balance_error/max(scale,1e-30)},
                 'minimum_active':minimum_a,'minimum_inactive':minimum_i,
                 'stationarity':latest,'two_windows_pass':successful,
                 'edge_mass_max_last_window':max((d['edge_mass'] for d in late),default=None),
                 'high_k_max_last_window':max((d['high_k'] for d in late),default=None),
                 'vortex':vortex_diagnostics(state[0],grid,p) if pop[0]>1e-8 else {'q':None,'status':'UNDEFINED'}}
        summary['checks']='PASS' if (successful and summary['balance']['relative']<1e-3
            and summary['edge_mass_max_last_window'] is not None and summary['edge_mass_max_last_window']<1e-4
            and summary['high_k_max_last_window']<1e-6) else 'UNRESOLVED'
        if p['spins']==2:
            individual=component_populations(state,grid)
            summary['component_populations']={'NC':individual[0],'NA':individual[1],'NI':individual[2]}
            summary['Z']=float((individual[0,0]-individual[0,1])/max(individual[0].sum(),1e-30))
            summary['symmetry_density_relative']=relative_norm(abs(state[0][0])**2,abs(state[0][1])**2)
            summary.pop('vortex',None)
            summary['spatial_components']=[{key:max((d['components'][k][key] for d in late),default=None)
                                            for key in ('edge_mass','high_k')} for k in range(2)]
            if any(v['edge_mass'] is None or v['edge_mass']>=1e-4 or v['high_k']>=1e-6
                   for v in summary['spatial_components']):summary['checks']='UNRESOLVED'
        write_json(path/'summary.json',summary);write_json(path/'diagnostics.json',diagnostics)
        write_json(path/'settling_windows.json',windows)
        if previous_handler is not None:signal.signal(signal.SIGINT,previous_handler)
    return {'path':path,'summary':summary}


def benchmark_2d(config,steps=120):
    p=parameters(config);grid=make_grid(config);pump=injection(config,grid)
    set_num_threads(config['execution'].get('local_workers',4))
    state=initial_state(config,grid);dt=config['time']['dt']
    factor=kinetic_factor(grid,p,dt/2)
    coeff=scalar_coefficients(p) if p['spins']==1 else spin_coefficients(p)
    workspace=(np.empty(grid.k2.size),np.empty(grid.k2.size))
    def local(state):
        if p['spins']==1:
            _scalar_local_inplace(*(v.ravel() for v in state),pump.ravel(),coeff,dt,p['model']=='M0')
        else:_spin_local_inplace(*state,pump,coeff,dt,workspace)
    local(tuple(v.copy() for v in state))
    start=time.perf_counter()
    for _ in range(steps):
        field=kinetic_step(state[0],factor,config['execution']['fft_workers'])
        state=(np.ascontiguousarray(field),state[1],state[2]);local(state)
        state=(kinetic_step(state[0],factor,config['execution']['fft_workers']),state[1],state[2])
    seconds=(time.perf_counter()-start)/steps
    return {'steps':steps,'step_ms':1000*seconds,'N':config['space']['N'],
            'estimated_integration_s':seconds*round(config['time']['final']/dt),
            'fft_workers':config['execution']['fft_workers'],'local_workers':config['execution']['local_workers']}


def compare_2d(reference,candidate):
    def final(path):
        with np.load(Path(path)/'checkpoint.npz') as z:
            return tuple(z[k].copy() for k in ('psi','active','inactive')),z['history'].copy()
    a,ha=final(reference);b,hb=final(candidate)
    if abs(ha[-1,0]-hb[-1,0])>1e-8:raise ValueError('Different comparison times')
    factor=a[0].shape[-1]//b[0].shape[-1]
    if factor>=1 and a[0].shape[-1]%b[0].shape[-1]==0:
        aa=tuple(v[:,::factor,::factor] for v in a);bb=b
    else:
        factor=b[0].shape[-1]//a[0].shape[-1]
        if factor<1 or b[0].shape[-1]%a[0].shape[-1]:raise ValueError('Non-nested grids')
        aa=a;bb=tuple(v[:,::factor,::factor] for v in b)
    nc=abs(aa[0])**2;nc2=abs(bb[0])**2
    errors={'NC_relative':float(abs(ha[-1,1]-hb[-1,1])/ha[-1,1]),
            'NA_relative':float(abs(ha[-1,2]-hb[-1,2])/ha[-1,2]),
            'density_relative':relative_norm(nc2,nc)}
    time_common=ha[:,0];candidate_nc=np.interp(time_common,hb[:,0],hb[:,1])
    errors['transient_NC_RMSE_relative']=float(np.sqrt(np.mean((candidate_nc-ha[:,1])**2))/max(float(np.mean(ha[:,1])),1e-30))
    errors['status']='PASS' if errors['NC_relative']<.005 and errors['NA_relative']<.005 and errors['density_relative']<.01 else 'FAIL'
    return errors


def run_ring_suite(max_workers=2):
    """F2: three physical runs and the prescribed R2 grid/time checks."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    from model import load_config
    jobs=[('F2_M0',load_config('F2',model='M0')),
          ('F2_R1',load_config('F2',reservoir='R1')),
          ('F2_R2',load_config('F2',reservoir='R2')),
          ('F2_R2_grid256',load_config('F2',reservoir='R2',space={'N':256})),
          ('F2_R2_dt_half',load_config('F2',reservoir='R2',time={'dt':.0125}))]
    report={'status':'RUNNING','cases':{},'comparisons':{},'protocol':'F2, independent starts; no copied previous trajectories'}
    output=ROOT/'results/data/F2_summary.json'
    def execute(items):
        if max_workers==1:
            for name,c in items:
                result=run_2d(c,name);report['cases'][name]=result
                write_json(output,report)
        else:
            with ProcessPoolExecutor(max_workers=max_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(run_2d,c,name):name for name,c in items}
                for future in as_completed(futures):
                    name=futures[future];report['cases'][name]=future.result()
                    write_json(output,report)
    execute(jobs)
    main=('F2_M0','F2_R1','F2_R2')
    if any(not report['cases'][name]['summary']['two_windows_pass'] for name in main):
        print('Extend all F2 cases to a common 4000 ps',flush=True)
        for _,c in jobs:c['time']['final']=4000.
        execute(jobs)
    reference=ROOT/'results/runs/F2_R2'
    report['comparisons']['grid256']=compare_2d(reference,ROOT/'results/runs/F2_R2_grid256')
    report['comparisons']['dt_half']=compare_2d(reference,ROOT/'results/runs/F2_R2_dt_half')
    if report['comparisons']['grid256']['status']=='FAIL':
        final=report['cases']['F2_R2']['summary']['time_ps']
        config=load_config('F2',reservoir='R2',space={'N':1024},time={'final':final})
        execute([('F2_R2_grid1024',config)])
        report['comparisons']['grid1024']=compare_2d(reference,ROOT/'results/runs/F2_R2_grid1024')
    report['status']='COMPLETED'
    spatial=report['comparisons'].get('grid1024',report['comparisons']['grid256'])['status']
    passed=spatial=='PASS' and report['comparisons']['dt_half']['status']=='PASS'
    passed=passed and all(report['cases'][name]['summary']['checks']=='PASS' for name in main)
    report['assessment']='PASS' if passed else 'NEEDS_REVIEW'
    if len({report['cases'][name]['summary']['vortex']['q'] for name in main})>1:
        report['stationary_control']='results/data/stationary_controls.json'
        run_stationary_controls(max_workers=max_workers)
    write_json(output,report)
    return report


def run_0d_checks():
    """Measured reference residuals and the prescribed fixed-step comparisons."""
    from model import load_config
    directory=ROOT/'results/data';checks={};arrays={}
    # Reclassify the common primary window first; extensions remain fully saved.
    records=[]
    previous={round(r['J'],8):r for r in json.loads((directory/'F5_scan.json').read_text())}
    for J in spin_scan_values():
        file=directory/f'F5_J_{J:.5f}.npz';data=load_0d(file)
        result={**previous.get(J,{}),**spin_summary(data,prefer_primary=previous.get(J,{}).get('prefer_primary',True))}
        write_json(ROOT/'results/runs/0d'/file.stem/'summary.json',result);records.append(result)
    write_json(directory/'F5_scan.json',records)
    for name,model,variant in [('M0','M0','R1'),('R1','M1','R1'),('R2','M1','R2')]:
        c=load_config('F1',model=model,reservoir=variant)
        fixed=solve_0d(c,method='RK4');ref=load_0d(directory/f'F1_{name}.npz')
        error={key:float(np.max(abs(fixed[key]-ref[key]))/max(float(np.max(abs(ref[key]))),1e-12))
               for key in ('psi','active','inactive')}
        checks['F1_RK4_'+name]={'errors':error,'status':'PASS' if max(error.values())<1e-7 else 'FAIL'}
    fixed_results={}
    for J in (.005,.010):
        file=directory/f'F5_J_{J:.5f}.npz';reference=load_0d(file)
        c=copy.deepcopy(reference['config']);c['time']['final']=6000.
        reference={**reference,**{k:reference[k][reference['time']<=6000] for k in ('time','psi','active','inactive')}}
        refobs=observables_0d(reference);refclass=classify_spin(reference)
        for dt in (.025,.0125):
            config=copy.deepcopy(c);config['time']['dt']=dt
            fixed=solve_0d(config,method='RK4');obs=observables_0d(fixed);result=classify_spin(fixed)
            last=fixed['time']>=4000;short=fixed['time']<=200
            density_error=float(np.max(abs(obs['nc'][last].mean(axis=0)-refobs['nc'][last].mean(axis=0))/refobs['nc'][last].mean(axis=0)))
            short_error=float(np.max(abs(fixed['psi'][short]-reference['psi'][short]))/np.max(abs(reference['psi'][short])))
            phase_difference=float(np.angle(np.exp(1j*(obs['theta'][-1]-refobs['theta'][-1]))))
            result.update(mean_density_relative=density_error,short_field_relative=short_error,
                          phase_final_difference=phase_difference)
            if refclass['status'].endswith('PERIODIC'):
                result['amplitude_relative']=abs(result['A_z']-refclass['A_z'])/refclass['A_z']
                result['period_relative']=abs(result.get('period_ps',0)-refclass['period_ps'])/refclass['period_ps']
                passed=density_error<.005 and result['amplitude_relative']<.01 and result['period_relative']<.01
            else:passed=density_error<.005 and abs(phase_difference)<.01
            result['check']='PASS' if passed else 'FAIL'
            fixed_results[f'J={J:g},dt={dt:g}']=result
            print('0D convergence',J,dt,result['check'],flush=True)
    checks['F4_fixed_steps']=fixed_results
    # Independent formulation using active feeding Q_A, with no n_I equation.
    for variant in ('R1','R2'):
        c=load_config('F1',reservoir=variant);p=parameters(c);ref=load_0d(directory/f'F1_{variant}.npz')
        values=np.empty((len(ref['time']),4));current=np.array([.01,0.,0.,0.]);values[0]=current
        for left,right,PA in ((0.,1000.,1.5*p['Pth']),(1000.,1200.,0.)):
            def rhs(t,y):
                psi=y[0]+1j*y[1];nc=abs(psi)**2
                dz=((p['R']*y[2]-p['gammaC'])/2-1j*p['gR']*(y[2]+p['eta']*y[3])/p['hbar']-1j*p['gC']*nc/p['hbar'])*psi
                return [dz.real,dz.imag,y[3]-(p['gammaA']+p['R']*nc)*y[2],(PA-y[3])/p['tauI']]
            result=solve_ivp(rhs,(left,right),current,method='DOP853',dense_output=True,**c['ode'])
            mask=(ref['time']>left)&(ref['time']<=right);values[mask]=result.sol(ref['time'][mask]).T
            current=result.y[:,-1]
        error=float(np.max(abs(values[:,2]-ref['active'][:,0]))/np.max(ref['active']))
        feed_error=float(np.max(abs(values[:,3]-p['W']*ref['inactive'][:,0]))/max(1.5*p['Pth'],1e-30))
        checks['Q_equivalence_'+variant]={'active_error':error,'feed_error':feed_error,'status':'PASS' if max(error,feed_error)<1e-8 else 'FAIL'}
        arrays['Q_time']=ref['time'];arrays['Q_error_'+variant]=values[:,2]-ref['active'][:,0]
    # A second, independent density/relative-phase formulation for F4.
    c=load_config('F4',physical={'J':.005});p=parameters(c);PA=injection(c)
    def density_phase_rhs(t,y):
        n=y[:2];a=y[3:5];b=y[5:7];theta=y[2]
        exchange=2*p['J']/p['hbar']*np.sqrt(n[0]*n[1])*np.sin(theta)
        dn=(p['R']*a-p['gammaC'])*n+np.array([exchange,-exchange])
        V=p['alpha1']*n+p['alpha2']*n[::-1]+p['gA']*a+p['gI']*b
        phase=-(V[0]-V[1])/p['hbar']-p['J']/p['hbar']*(n[0]-n[1])/np.sqrt(n[0]*n[1])*np.cos(theta)
        da=p['W']*b-(p['gammaA']+p['R']*n)*a+p['GammaA']*(a[::-1]-a)
        db=p['injection_ratio']*PA-(p['gammaI']+p['W'])*b+p['GammaI']*(b[::-1]-b)
        return np.r_[dn,phase,da,db]
    ref=load_0d(directory/'F5_J_0.00500.npz');mask=ref['time']<=1200
    res=solve_ivp(density_phase_rhs,(0.,1200.),[5e-5,5e-5,.2,0.,0.,0.,0.],
                  method='DOP853',t_eval=ref['time'][mask],**c['ode'])
    density_error=float(np.max(abs(res.y[:2].T-abs(ref['psi'][mask])**2))/np.max(abs(ref['psi'][mask])**2))
    checks['independent_density_phase']={'density_relative':density_error,'status':'PASS' if density_error<1e-8 else 'FAIL'}
    arrays['F3_epsilon']=np.array([r['epsilon'] for r in json.loads((directory/'F3_scan.json').read_text())])
    arrays['F3_error']=np.array([r['density_error'] for r in json.loads((directory/'F3_scan.json').read_text())])
    np.savez_compressed(directory/'reference_checks.npz',**arrays)
    write_json(directory/'reference_checks.json',checks)
    checks['F5_refinement']=run_refinement_checks()
    return checks


def run_phase_seed_checks():
    """Two additional phases on either side of the observed sampled switch."""
    from model import load_config
    directory=ROOT/'results/data';phase_file=directory/'phase_seed_checks.json'
    rows=json.loads(phase_file.read_text()) if phase_file.exists() else []
    for J in (.006,.007):
        for theta in (1.,2.):
            config=load_config('F5',physical={'J':J},initial={'theta':theta})
            data=solve_0d(config);result=classify_spin(data)
            if result['status']=='UNRESOLVED':
                config['time']['final']=12000.
                extra=solve_0d(config,state0=tuple(data[k][-1] for k in ('psi','active','inactive')),start=6000.)
                data={**data,**{k:np.concatenate((data[k],extra[k][1:])) for k in ('time','psi','active','inactive')},'config':config}
                result=classify_spin(data)
            result.update(J=J,initial_theta=theta)
            save_0d(directory/f'phase_J_{J:.5f}_theta_{theta:g}.npz',data,result)
            rows=[r for r in rows if not (np.isclose(r['J'],J,rtol=0,atol=1e-10) and r['initial_theta']==theta)]
            rows.append(result);print('Additional phase',J,theta,result['status'],flush=True)
    write_json(directory/'phase_seed_checks.json',rows)
    return rows


def run_rabi_check():
    from model import load_config
    config=load_config('F4');p=parameters(config)
    for key in ('gC','gR','gA','gI','alpha1','alpha2','gammaC'):p[key]=0.
    dt=.125;times=np.arange(4001)*dt
    state=(np.array([1.+0j,0j]),np.zeros(2),np.zeros(2));fields=[state[0].copy()]
    for _ in range(4000):
        state=local_rk4(state,np.zeros(2),p,dt);fields.append(state[0].copy())
    fields=np.array(fields);exact=np.column_stack((np.cos(p['J']*times/p['hbar']),1j*np.sin(p['J']*times/p['hbar'])))
    file=ROOT/'results/data/reference_checks.npz'
    with np.load(file) as z:arrays={k:z[k].copy() for k in z.files}
    error=np.max(abs(fields-exact),axis=1)
    arrays.update(rabi_time=times,rabi_density=abs(fields[:,0])**2,rabi_expected=abs(exact[:,0])**2,rabi_error=error)
    np.savez_compressed(file,**arrays)
    metadata=json.loads(file.with_suffix('.json').read_text())
    metadata['Rabi']={'dt_ps':dt,'duration_ps':500.,'max_abs_field_error':float(error.max()),
                      'population_period_ps':float(np.pi*p['hbar']/p['J']),'status':'PASS' if error.max()<1e-10 else 'FAIL'}
    write_json(file.with_suffix('.json'),metadata)
    return metadata['Rabi']


def _stationary_control_case(model,dt):
    from model import load_config
    source=ROOT/'results/runs/F2_M0/checkpoint.npz'
    digest=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
    config=load_config('F2',model=model,reservoir='R2',time={'final':200.,'dt':dt},
                       initial={'source':'F2_M0 at 2000 ps','source_sha256':digest})
    if source.exists():
        with np.load(source) as z:field=z['psi'].copy();active=z['active'].copy()
    else:
        with np.load(ROOT/'results/data/report_data.npz') as z:
            field=z['F2_M0_psi'].copy();active=z['F2_M0_active'].copy()
    p=parameters(config);grid=make_grid(config)
    inactive=injection(config,grid)/p['W'] if model=='M1' else np.zeros_like(active)
    name=f'F2_control_{model}_{dt:g}'
    result=run_2d(config,name,initial=(field,active,inactive))
    errors=[]
    for snapshot in sorted((result['path']/'snapshots').glob('*.npz')):
        with np.load(snapshot) as z:
            aligned,phase=aligned_field_error(z['psi'],field)
            density=relative_norm(abs(z['psi'])**2,abs(field)**2)
            errors.append({'time':float(z['time']),'field_error':aligned,'density_error':density,'removed_phase':phase})
    result['errors']=errors
    result['status']='PASS' if max(e['field_error'] for e in errors)<.01 and max(e['density_error'] for e in errors)<.01 and result['summary']['vortex']['q']==-3 else 'FAIL'
    return result


def run_stationary_controls(max_workers=2):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    jobs=[(model,dt) for dt in (.025,.0125) for model in ('M0','M1')]
    records={}
    if max_workers==1:
        for model,dt in jobs:
            records[f'{model},dt={dt:g}']=_stationary_control_case(model,dt)
            write_json(ROOT/'results/data/stationary_controls.json',records)
    else:
        with ProcessPoolExecutor(max_workers=max_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(_stationary_control_case,model,dt):(model,dt) for model,dt in jobs}
            for future in as_completed(futures):
                model,dt=futures[future];records[f'{model},dt={dt:g}']=future.result()
                write_json(ROOT/'results/data/stationary_controls.json',records)
    for dt in (.025,.0125):
        a=records[f'M0,dt={dt:g}'];b=records[f'M1,dt={dt:g}']
        a['matched_model_comparison']=compare_2d(a['path'],b['path'])
    for model in ('M0','M1'):
        records[f'{model},dt=0.025']['time_refinement']=compare_2d(records[f'{model},dt=0.0125']['path'],records[f'{model},dt=0.025']['path'])
    write_json(ROOT/'results/data/stationary_controls.json',records)
    return records


def export_report_data():
    """One portable array archive; detailed checkpoints stay in ignored runs/."""
    from scipy.integrate import cumulative_trapezoid
    directory=ROOT/'results/data';arrays={};manifest={}
    names=([f'F1_{v}' for v in ('M0','R1','R2')]
           +[f'F3_eps_{v:+.1f}' for v in np.linspace(-1,1,21)]
           +[f'F5_J_{v:.5f}' for v in spin_scan_values()])
    phase_rows=json.loads((directory/'phase_seed_checks.json').read_text())
    names.extend(f"phase_J_{r['J']:.5f}_theta_{r['initial_theta']:g}" for r in phase_rows)
    checks=json.loads((directory/'reference_checks.json').read_text())
    names.extend(r['case'] for r in checks.get('F5_polarization_controls',{}).get('continuations',[]))
    for name in names:
        file=directory/(name+'.npz');data=load_0d(file)
        selected=np.ones(len(data['time']),dtype=bool);sampling={'history_ps':data['config']['time']['sample']}
        if name.startswith('F5_J_') and name not in {f'F5_J_{v:.5f}' for v in (0.,.005,.0065,.010)}:
            step=data['config']['time']['sample'];stride=max(1,round(20./step))
            selected=np.isclose(data['time']/(stride*step),np.rint(data['time']/(stride*step)),rtol=0,atol=1e-8)
            selected[-1]=True;selected|=data['time']<=200
            record=next(r for r in json.loads((directory/'F5_scan.json').read_text())
                        if np.isclose(r['J'],parameters(data['config'])['J'],rtol=0,atol=1e-10))
            windows=[record['window_ps']]
            windows+= [[left-10000.,right-10000.] for left,right in windows if left-10000.>=data['time'][0]]
            windows=sorted({tuple(w) for w in windows})
            for left,right in windows:selected|=(data['time']>=left)&(data['time']<=right)
            sampling={'history_ps':stride*step,'analysis_ps':step,'full_resolution_windows_ps':windows,
                      'note':'Exact saved samples only; detailed local trajectories retain all time samples.'}
        for key in ('time','psi','active','inactive'):arrays[name+'_'+key]=data[key][selected]
        arrays[name+'_config']=json.dumps(data['config']);arrays[name+'_method']=data['method']
        arrays[name+'_archive_sampling']=json.dumps(sampling)
        old_summary=file.with_suffix('.json');local=ROOT/'results/runs/0d'/name
        summary_path=old_summary if old_summary.exists() else local/'summary.json'
        summary=json.loads(summary_path.read_text())
        if file.exists():
            save_0d(file,data,summary)
            file.unlink();old_summary.unlink(missing_ok=True)
        manifest[name]={'final_ps':float(data['time'][-1]),'method':data['method'],'archive_sampling':sampling}
    report=json.loads((directory/'F2_summary.json').read_text())
    report.setdefault('diagnostic_series',{});report.setdefault('component_balance',{})
    for name in report['cases']:
        path=ROOT/'results/runs'/name
        if not (path/'checkpoint.npz').exists():
            with np.load(directory/'report_data.npz',allow_pickle=False) as old:
                for key in ('config','psi','active','inactive','history'):arrays[name+'_'+key]=old[name+'_'+key].copy()
            continue
        config=yaml.safe_load((path/'resolved_config.yaml').read_text());p=parameters(config);g=make_grid(config)
        arrays[name+'_config']=json.dumps(config)
        with np.load(path/'checkpoint.npz',allow_pickle=False) as z:
            for key in ('psi','active','inactive','history'):arrays[name+'_'+key]=z[key].copy()
            if name in ('F2_M0','F2_R1','F2_R2'):
                t=z['frame_times'];nc=abs(z['frame_psi'])**2;a=z['frame_active'];b=z['frame_inactive']
                values=np.column_stack([v.sum(axis=(1,2,3))*g.da for v in (nc,a,b)])
                feed=np.broadcast_to(injection(config,g),a.shape) if p['model']=='M0' else p['W']*b
                sources=np.column_stack([v.sum(axis=(1,2,3))*g.da for v in
                    ((p['R']*a-p['gammaC'])*nc,feed-(p['gammaA']+p['R']*nc)*a)])
                scales=np.column_stack([v.sum(axis=(1,2,3))*g.da for v in
                    ((p['R']*a+p['gammaC'])*nc,feed+(p['gammaA']+p['R']*nc)*a)])
                integral=cumulative_trapezoid(sources,t,axis=0,initial=0)
                denominator=cumulative_trapezoid(scales,t,axis=0,initial=0)
                absolute=abs(values[:,:2]-values[0,:2]-integral)
                relative=absolute/np.maximum(denominator,1e-30)
                report['component_balance'][name]={'time_ps':t.tolist(),'relative':relative.tolist(),
                    'final_absolute':absolute[-1].tolist(),'final_denominator':denominator[-1].tolist(),
                    'components':['condensate','active reservoir'],
                    'method':'Trapezoid of full-field sources sampled every 5 ps in final 100 ps; no local kinetic source under periodic boundaries.'}
        report['diagnostic_series'][name]=json.loads((path/'diagnostics.json').read_text())
    controls_path=directory/'stationary_controls.json'
    if controls_path.exists():
        controls=json.loads(controls_path.read_text())
        for dt in (.025,.0125):
            a=ROOT/'results/runs'/f'F2_control_M0_{dt:g}'/'checkpoint.npz'
            b=ROOT/'results/runs'/f'F2_control_M1_{dt:g}'/'checkpoint.npz'
            if a.exists() and b.exists():
                with np.load(a) as aa,np.load(b) as bb:
                    error,phase=aligned_field_error(bb['psi'],aa['psi'])
                controls[f'M0,dt={dt:g}']['matched_model_comparison'].update(aligned_field_relative=error,removed_phase=phase)
        write_json(controls_path,controls)
    arrays['manifest_json']=json.dumps(manifest)
    temporary=directory/'report_data.tmp.npz';np.savez_compressed(temporary,**arrays)
    temporary.replace(directory/'report_data.npz')
    report['stationary_control']='results/data/stationary_controls.json'
    write_json(directory/'F2_summary.json',report)
    # Public metadata must not depend on the author's absolute checkout path.
    def relative_paths(value):
        if isinstance(value,dict):return {k:relative_paths(v) for k,v in value.items()}
        if isinstance(value,list):return [relative_paths(v) for v in value]
        if isinstance(value,str):return value.replace(str(ROOT)+'/', '')
        return value
    for file in directory.glob('*.json'):
        original=json.loads(file.read_text());portable=relative_paths(original)
        if portable!=original:write_json(file,portable)
    return directory/'report_data.npz'


def spatial_spin_reference_checks():
    """Small uniform 2D/DOP853 comparison and symmetric-preparation control."""
    from model import load_config
    set_num_threads(1)
    errors={}
    for epsilon in (0.,.3):
        c=load_config('N3',pump={'kind':'uniform','s':2.,'epsilon':epsilon},
                      space={'N':8,'L':8.},time={'final':200.,'sample':.5})
        p=parameters(c);g=make_grid(c);pump=injection(c,g)
        state=tuple(np.broadcast_to(v[:,None,None],(2,8,8)).copy() for v in initial_state(c))
        maximum_symmetry=0.
        for _ in range(round(c['time']['final']/c['time']['dt'])):
            state=split_step(state,pump,p,c['time']['dt'],g)
            if epsilon==0.:maximum_symmetry=max(maximum_symmetry,float(np.max(abs(state[0][0]-state[0][1]))))
        ref=solve_0d(c)
        residuals={k:relative_norm(v[:,0,0],ref[k][-1]) for k,v in zip(('psi','active','inactive'),state)}
        errors[str(epsilon)]={'relative_errors':residuals,'symmetry_max_absolute':maximum_symmetry if epsilon==0. else None,
                             'status':'PASS' if max(residuals.values())<1e-7 and maximum_symmetry<1e-12 else 'FAIL'}
    coarse=load_config('N3',space={'N':256});fine=load_config('N3')
    a=initial_state(coarse,make_grid(coarse))[0];b=initial_state(fine,make_grid(fine))[0]
    errors['continuous_seed']={'nested_grid_equal':bool(np.array_equal(a,b[:,::2,::2])),
                               'equal_initial_components':bool(np.array_equal(b[0],b[1]))}
    if not all(v['status']=='PASS' for k,v in errors.items() if k!='continuous_seed') or not all(errors['continuous_seed'].values()):
        raise RuntimeError('Spatial spin reference check failed: '+str(errors))
    return errors


def compare_spatial_spin(reference,candidate):
    """Absolute component densities on nested nodes; no fitted amplitude or rotation."""
    def read(name):
        path=ROOT/'results/runs'/name
        c=yaml.safe_load((path/'resolved_config.yaml').read_text());g=make_grid(c)
        with np.load(path/'checkpoint.npz',allow_pickle=False) as z:
            state=tuple(z[k].copy() for k in ('psi','active','inactive'));t=float(z['step'])*c['time']['dt']
        return state,g,t
    a,ga,ta=read(reference);b,gb,tb=read(candidate)
    if abs(ta-tb)>1e-8:raise ValueError('Spatial comparisons require equal final times')
    pa=component_populations(a,ga);pb=component_populations(b,gb)
    if a[0].shape[-1]<b[0].shape[-1]:raise ValueError('Reference must have the finer grid')
    ratio=a[0].shape[-1]//b[0].shape[-1]
    if ratio*b[0].shape[-1]!=a[0].shape[-1] or not np.allclose(ga.x[::ratio],gb.x):raise ValueError('Non-nested physical grids')
    aa=tuple(v[:,::ratio,::ratio] for v in a)
    density=[relative_norm(abs(b[0][k])**2,abs(aa[0][k])**2) for k in range(2)]
    population=np.abs(pb-pa)/np.maximum(np.abs(pa),1e-30)
    passed=bool(max(density)<.01 and population[0].max()<.01)
    return {'status':'PASS' if passed else 'FAIL','time_ps':ta,
            'NC_relative':population[0],'NA_relative':population[1],'NI_relative':population[2],
            'density_L2_relative':density,'tolerance':.01,
            'map_comparison':'Fine-grid density sampled at the identical coarse-grid nodes; absolute densities, no fit.'}


def export_spatial_spin_data(report):
    """Portable final fields, reservoir fields, histories and resolved configurations."""
    arrays={};manifest={}
    for name,record in report['cases'].items():
        path=ROOT/'results/runs'/name
        c=yaml.safe_load((path/'resolved_config.yaml').read_text())
        arrays[name+'_config']=json.dumps(c)
        with np.load(path/'checkpoint.npz',allow_pickle=False) as z:
            for key in ('psi','active','inactive','history'):arrays[name+'_'+key]=z[key].copy()
        manifest[name]={'time_ps':record['time_ps'],'N':c['space']['N'],'dt_ps':c['time']['dt']}
    arrays['manifest_json']=json.dumps(manifest)
    directory=ROOT/'results/data';directory.mkdir(parents=True,exist_ok=True);tmp=directory/'spatial_spin.tmp.npz'
    np.savez_compressed(tmp,**arrays);tmp.replace(directory/'spatial_spin.npz')
    return directory/'spatial_spin.npz'


def run_spatial_spin_suite(max_workers=4,*,extension_ps=100.,max_final=4000.):
    """N3: resume independent starts; test common times in short settling extensions."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    from model import load_config
    jobs=[('N3_eps_0',load_config('N3',pump={'epsilon':0.})),
          ('N3_eps_03',load_config('N3',pump={'epsilon':.3})),
          ('N3_eps_03_dt_half',load_config('N3',pump={'epsilon':.3},time={'dt':.0125})),
          ('N3_eps_03_grid256',load_config('N3',pump={'epsilon':.3},space={'N':256}))]
    if extension_ps<=0 or not np.isclose(extension_ps/100,round(extension_ps/100)):
        raise ValueError('Extension must be a positive multiple of the 100 ps settling window')
    if max_final<2000 or not np.isclose(max_final/100,round(max_final/100)):
        raise ValueError('Maximum final time must be >=2000 ps and align with settling windows')
    latest=[c['time']['final'] for _,c in jobs]
    for name,c in jobs:
        checkpoint=ROOT/'results/runs'/name/'checkpoint.npz'
        if checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as z:latest.append(float(z['step'])*c['time']['dt'])
    target=100.*np.ceil((max(latest)-1e-8)/100.)
    if target>max_final:raise ValueError('Saved state is beyond the requested maximum final time')
    for _,c in jobs:c['time']['final']=target
    report={'status':'RUNNING','assessment':'INCOMPLETE','cases':{},'comparisons':{},
            'reference_checks':spatial_spin_reference_checks(),
            'extension_policy':{'step_ps':extension_ps,'max_final_ps':max_final,'initial_common_target_ps':target},
            'protocol':'Independent continuous equal-component seed; initially empty reservoirs; no coupling.'}
    print(f'N3: common target {target:g} ps; subsequent extensions {extension_ps:g} ps only if needed.',flush=True)
    output=ROOT/'results/data/N3_summary.json';write_json(output,report)
    def execute(items):
        if max_workers==1:
            for name,c in items:
                report['cases'][name]=run_2d(c,name)['summary'];write_json(output,report)
        else:
            with ProcessPoolExecutor(max_workers=max_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(run_2d,c,name):name for name,c in items}
                for future in as_completed(futures):
                    report['cases'][futures[future]]=future.result()['summary'];write_json(output,report)
    execute(jobs)
    while True:
        if any(r['run_status']!='RUN_COMPLETED' for r in report['cases'].values()):
            report['status']='RUN_INCOMPLETE';write_json(output,report)
            return report
        if all(r['two_windows_pass'] for r in report['cases'].values()) or target>=max_final:break
        target=min(max_final,target+extension_ps)
        print(f'N3: extend all four runs to common {target:g} ps, then recheck both components.',flush=True)
        for _,c in jobs:c['time']['final']=target
        execute(jobs)
    report['comparisons']['grid256']=compare_spatial_spin('N3_eps_03','N3_eps_03_grid256')
    report['comparisons']['dt_half']=compare_spatial_spin('N3_eps_03','N3_eps_03_dt_half')
    symmetry=report['cases']['N3_eps_0']['symmetry_density_relative']
    report['symmetric_preparation']={'relative_density_difference':symmetry,'status':'PASS' if symmetry<1e-10 else 'FAIL'}
    passed=(all(r['checks']=='PASS' for r in report['cases'].values())
            and all(r['status']=='PASS' for r in report['comparisons'].values())
            and report['symmetric_preparation']['status']=='PASS')
    report.update(status='COMPLETED',assessment='PASS' if passed else 'NEEDS_REVIEW')
    export_spatial_spin_data(report);write_json(output,report)
    return report


class TorchSpinStepper:
    """Resident spinor step on Intel Arc (or CPU for tests).

    Configurable field precision; float64 reductions. No filter or absorber. J remains local.
    Optional torch.compile fuses *real* RK4 arithmetic; FFT stays torch.fft.
    """
    def __init__(self, config, grid=None, device='xpu', compiled=False):
        import torch
        self.torch=torch;self.p=parameters(config);self.grid=grid or make_grid(config)
        self.precision=config.get('execution',{}).get('precision','complex128')
        if self.precision not in ('complex64','complex128'):raise ValueError('Unsupported field precision')
        self.complex_dtype=torch.complex64 if self.precision=='complex64' else torch.complex128
        self.real_dtype=torch.float32 if self.precision=='complex64' else torch.float64
        if self.p['spins']!=2:raise ValueError('Spinor backend requires two fields')
        if device=='xpu' and not torch.xpu.is_available():
            raise RuntimeError('Intel Arc unavailable; no silent CPU fallback')
        self.device=torch.device(device);self.dt=config['time']['dt']
        self.pump=self.tensor(injection(config,self.grid))
        factor=kinetic_factor(self.grid,self.p,self.dt/2)
        self.factor=tuple(self.tensor(v) for v in factor) if isinstance(factor,tuple) else self.tensor(factor)
        self.local=self._local
        if compiled:self.local=torch.compile(self._local,fullgraph=True)
        self.compiled=compiled
        full=kinetic_factor(self.grid,self.p,self.dt)
        self.full_factor=tuple(self.tensor(v) for v in full) if isinstance(full,tuple) else self.tensor(full)
        self.staggered=False

    def tensor(self, value):
        t=self.torch
        return t.as_tensor(value,device=self.device,
                           dtype=self.complex_dtype if np.iscomplexobj(value) else self.real_dtype)

    def to_device(self,state):return tuple(self.tensor(v) for v in state)

    def to_numpy(self,state):return tuple(v.detach().cpu().numpy().copy() for v in state)

    def synchronize(self):
        if self.device.type=='xpu':self.torch.xpu.synchronize()

    def kinetic(self,z,full=False):
        t=self.torch;f=t.fft.fft2(z);factor=self.full_factor if full else self.factor
        if isinstance(factor,tuple):
            a,b,c=factor
            # Both outputs use the unchanged input spinor.
            f=t.stack((a*f[0]+b*f[1],c*f[0]+a*f[1]))
        else:f=f*factor
        return t.fft.ifft2(f)

    def _rhs_real(self,x,y,a,b):
        p=self.p;t=self.torch;n=x*x+y*y
        energy=(p['alpha1']*n+p['alpha2']*n.flip(0)+p['gA']*a+p['gI']*b)/p['hbar']
        gain=(p['R']*a-p['gammaC'])/2;j=p['J']/p['hbar']
        dx=gain*x+energy*y-j*y.flip(0)
        dy=gain*y-energy*x+j*x.flip(0)
        da=p['W']*b-(p['gammaA']+p['R']*n)*a+p['GammaA']*(a.flip(0)-a)
        db=p['injection_ratio']*self.pump-(p['gammaI']+p['W'])*b+p['GammaI']*(b.flip(0)-b)
        incoming=p['injection_ratio']*self.pump.sum(0)
        outgoing=(p['gammaC']*n+p['gammaA']*a+p['gammaI']*b).sum(0)
        cg=(p['R']*a*n).sum(0);cl=p['gammaC']*n.sum(0)
        return dx,dy,da,db,incoming-outgoing,incoming+outgoing,cg-cl,cg+cl

    def _local(self,x,y,a,b):
        dt=self.dt;v=(x,y,a,b)
        q=self._rhs_real(*v)
        r=self._rhs_real(*(v[i]+dt*q[i]/2 for i in range(4)))
        s=self._rhs_real(*(v[i]+dt*r[i]/2 for i in range(4)))
        u=self._rhs_real(*(v[i]+dt*s[i] for i in range(4)))
        out=tuple(v[i]+dt/6*(q[i]+2*r[i]+2*s[i]+u[i]) for i in range(4))
        change=dt/6*(q[4]+2*r[4]+2*s[4]+u[4])
        scale=dt/6*(q[5]+2*r[5]+2*s[5]+u[5])
        cond=dt/6*(q[6]+2*r[6]+2*s[6]+u[6])
        cond_scale=dt/6*(q[7]+2*r[7]+2*s[7]+u[7])
        return (*out,change.sum(dtype=self.torch.float64),scale.sum(dtype=self.torch.float64),cond.sum(dtype=self.torch.float64),cond_scale.sum(dtype=self.torch.float64))

    def step(self,state,close=True):
        # Consecutive identical U(dt/2) factors combine exactly into U(dt).
        # Every observable/checkpoint boundary still contains the physical state.
        t=self.torch;z=state[0] if self.staggered else self.kinetic(state[0])
        x,y,a,b,source,scale,cond,cond_scale=self.local(z.real,z.imag,state[1],state[2])
        self.last_condensate=(cond*self.grid.da,cond_scale*self.grid.da)
        z=self.kinetic(t.complex(x,y),full=not close)
        self.staggered=not close
        return (z,a,b),source*self.grid.da,scale*self.grid.da

    def physical_state(self,state):
        if not self.staggered:return state
        inverse=kinetic_factor(self.grid,self.p,-self.dt/2)
        saved=self.factor
        self.factor=tuple(self.tensor(v) for v in inverse) if isinstance(inverse,tuple) else self.tensor(inverse)
        z=self.kinetic(state[0]);self.factor=saved;self.staggered=False
        return z,state[1],state[2]


def tetm_reference_checks(device=None, compiled=False):
    """Independent analytic limits; returned residuals are saved by the pilot runner."""
    from scipy.linalg import expm
    c=report3_config('T0',space={'N':32,'L':32.})
    g=make_grid(c);p=parameters(c);tau=3.7
    rows={}
    def record(label,value,tolerance):
        rows[label]={'error':float(value),'tolerance':tolerance,
                     'status':'PASS' if value<tolerance else 'FAIL'}
    factor=kinetic_factor(g,p,tau)
    matrix_error=0.
    for iy,ix in ((0,0),(0,3),(2,0),(2,3),(5,29)):
        kx,ky=g.kx[ix],g.ky[iy]
        H=p['K']*(kx*kx+ky*ky)*np.eye(2,dtype=complex)
        H[0,1]=p['beta']*(kx-1j*ky)**2;H[1,0]=H[0,1].conj()
        actual=np.array([[factor[0][iy,ix],factor[1][iy,ix]],
                         [factor[2][iy,ix],factor[0][iy,ix]]])
        matrix_error=max(matrix_error,float(abs(actual-expm(-1j*H*tau/p['hbar'])).max()))
    record('matrix_exponential_axes_off_axes_k0',matrix_error,2e-13)
    wave=np.exp(1j*(g.kx[3]*g.x[None,:]+g.ky[2]*g.y[:,None]))
    k2=g.kx[3]**2+g.ky[2]**2;phi=np.arctan2(g.ky[2],g.kx[3])
    phase=np.exp(-1j*p['K']*k2*tau/p['hbar']);u=p['beta']*k2*tau/p['hbar']
    initial=np.stack((wave,np.zeros_like(wave)))
    actual=kinetic_step(initial,factor)
    expected=np.stack((phase*np.cos(u)*wave,-1j*phase*np.exp(2j*phi)*np.sin(u)*wave))
    record('free_plus_wave_complex_conversion',abs(actual-expected).max(),3e-13)
    eigen_error=0.
    for sign in (-1,1):
        eigen=np.array([np.exp(-1j*phi),sign*np.exp(1j*phi)])/np.sqrt(2)
        field=eigen[:,None,None]*wave
        exact=field*np.exp(-1j*(p['K']+sign*p['beta'])*k2*tau/p['hbar'])
        eigen_error=max(eigen_error,float(abs(kinetic_step(field,factor)-exact).max()))
    record('TE_TM_eigenwaves',eigen_error,3e-13)
    rng=np.random.default_rng(41);field=rng.normal(size=(2,32,32))+1j*rng.normal(size=(2,32,32))
    out=kinetic_step(field,factor)
    record('unitarity',abs(np.sum(abs(out)**2)/np.sum(abs(field)**2)-1),2e-14)
    plain=kinetic_factor(g,{**p,'beta':0.},tau)
    old=np.exp(-1j*p['K']*g.k2*tau/p['hbar'])
    record('beta_zero_scalar_factor',abs(plain-old).max(),1e-15)
    constant=np.broadcast_to(np.array([1.+.2j,.4-.1j])[:,None,None],(2,32,32)).copy()
    record('k0_unchanged_by_beta',abs(kinetic_step(constant,factor)-constant).max(),1e-14)
    # Pure circular, x-linear, +45-degree diagonal; fixed e+/- convention.
    spins=np.array([[1.,0.],[0.,1.],[1.,1.],[1.,1j]],complex).T
    spins[:,2:]/=np.sqrt(2)
    S=stokes(spins);target=np.array([[1,1,1,1],[0,0,1,0],[0,0,0,1],[1,-1,0,0]])
    record('known_Stokes_and_analyzers',max(abs(S-target).max(),
           abs(np.sum(S[1:]**2,axis=0)-S[0]**2).max()),2e-15)
    assert min(((S[0]+S[1:])/2).min(),((S[0]-S[1:])/2).min())>=-1e-15
    pj={**p,'beta':0.,'J':.01,'alpha1':0.,'alpha2':0.,'gA':0.,'gI':0.,'gammaC':0.}
    z=np.array([1.2*np.exp(.3j),.8*np.exp(-.2j)])
    rhs=local_rhs((z,np.zeros(2),np.zeros(2)),np.zeros(2),pj)[0]
    source=2*np.real(z.conj()*rhs);IJ=-pj['J']/pj['hbar']*stokes(z)[2]
    record('minus_J_transfer_sign_with_new_S2',abs(source-np.array([IJ,-IJ])).max(),1e-15)
    dt=.25;state=(np.array([1.+0j,0j]),np.zeros(2),np.zeros(2))
    for _ in range(400):state=local_rk4(state,np.zeros(2),pj,dt)
    exact=np.array([np.cos(pj['J']*100/pj['hbar']),1j*np.sin(pj['J']*100/pj['hbar'])])
    record('minus_J_Rabi_complex_phase',abs(state[0]-exact).max(),1e-11)
    if device is not None:
        backend=TorchSpinStepper(c,g,device,compiled)
        backend.factor=tuple(backend.tensor(v) for v in factor)
        value=backend.kinetic(backend.tensor(initial)).cpu().numpy()
        record('device_free_wave',abs(value-expected).max(),3e-13)
        plain_backend=TorchSpinStepper(c,g,device,compiled)
        fused_backend=TorchSpinStepper(c,g,device,compiled)
        seed=initial_state(c,g)
        plain_state=plain_backend.to_device(seed);fused_state=fused_backend.to_device(seed)
        for i in range(20):
            plain_state,_,_=plain_backend.step(plain_state)
            fused_state,_,_=fused_backend.step(fused_state,close=i==19)
        fusion=max(relative_norm(a,b) for a,b in zip(fused_backend.to_numpy(fused_state),plain_backend.to_numpy(plain_state)))
        record('device_merged_FFT_half_steps',fusion,2e-13)
        # An exact k=0 plane wave reduces the complete device step to 0D.
        cc=report3_config('T0',pump={'kind':'uniform','s':2.,'epsilon':.3},
                          physical={'J':.01},space={'N':8,'L':8.},
                          time={'final':20.,'sample':.25})
        pp=parameters(cc);backend=TorchSpinStepper(cc,device=device,compiled=compiled)
        zero=(np.array([.2+.03j,.1-.02j]),np.array([7.,8.]),np.array([8.,9.]))
        state=backend.to_device(tuple(np.broadcast_to(v[:,None,None],(2,8,8)).copy() for v in zero))
        for _ in range(round(20/backend.dt)):state,_,_=backend.step(state)
        state=backend.to_numpy(state);ref=solve_0d(cc,state0=zero)
        for a,key in zip(state,('psi','active','inactive')):
            record('device_uniform_2D_0D_'+key,abs(a-ref[key][-1,:,None,None]).max(),2e-9)
        jc=report3_config('B0',space={'N':8,'L':8.},time={'dt':.25})
        rab=TorchSpinStepper(jc,device=device,compiled=compiled)
        rab.p.update(J=.01,R=0.,gammaC=0.,alpha1=0.,alpha2=0.,gA=0.,gI=0.)
        init=(np.stack((np.ones((8,8),complex),np.zeros((8,8),complex))),np.zeros((2,8,8)),np.zeros((2,8,8)))
        rs=rab.to_device(init)
        for _ in range(400):rs,_,_=rab.step(rs)
        record('device_minus_J_Rabi_sign',abs(rab.to_numpy(rs)[0]-exact[:,None,None]).max(),1e-11)
        rows['device']={'name':str(backend.device),'precision':'complex128/float64',
                        'compiled_local':compiled,'torch':backend.torch.__version__}
    passed=all(v.get('status','PASS')=='PASS' for v in rows.values())
    return {'status':'PASS' if passed else 'FAIL','checks':rows}


class _ProbeStore:
    """Disk-backed rolling full-precision fields, sampled every 5 ps, not each step."""
    def __init__(self,path,n,duration=400.,cadence=5.,times=None):
        self.path=Path(path);self.cadence=cadence;self.n=n
        if self.path.exists():
            self.maps=np.lib.format.open_memmap(self.path,mode='r+')
            self.times=np.asarray(times) if times is not None else np.full(len(self.maps),-np.inf)
            if len(self.times)!=len(self.maps):self.times=np.full(len(self.maps),-np.inf)
        else:
            slots=int(np.ceil(duration/cadence))+2
            self.maps=np.lib.format.open_memmap(self.path,mode='w+',dtype='float64',shape=(slots,8,n,n))
            self.times=np.full(slots,-np.inf)

    def ensure(self,duration):
        slots=int(np.ceil(duration/self.cadence))+2
        if slots<=len(self.maps):return
        tmp=self.path.with_suffix('.new.npy')
        bigger=np.lib.format.open_memmap(tmp,mode='w+',dtype='float64',shape=(slots,8,self.n,self.n))
        valid=np.flatnonzero(np.isfinite(self.times));new_times=np.full(slots,-np.inf)
        for j,i in enumerate(valid):bigger[j]=self.maps[i];new_times[j]=self.times[i]
        bigger.flush();del bigger;del self.maps
        tmp.replace(self.path);self.maps=np.lib.format.open_memmap(self.path,mode='r+')
        self.times=new_times

    def add(self,t,state):
        i=int(np.argmin(self.times));z,a,b=state
        self.maps[i,:4]=np.stack((z[0].real,z[0].imag,z[1].real,z[1].imag))
        self.maps[i,4:6]=a;self.maps[i,6:8]=b;self.times[i]=t

    def state(self,i):
        v=self.maps[i]
        return np.stack((v[0]+1j*v[1],v[2]+1j*v[3])),v[4:6],v[6:8]

    def ordered(self,left=-np.inf,right=np.inf):
        ix=np.flatnonzero(np.isfinite(self.times)&(self.times>=left-1e-8)&(self.times<=right+1e-8))
        return ix[np.argsort(self.times[ix])]


def _report3_maps(state):
    S=stokes(state[0])
    return np.concatenate((abs(state[0])**2,state[1],state[2],S[1:])),S[0]


def _report3_map_errors(current,reference,S0):
    denominator=np.linalg.norm(reference.reshape(9,-1),axis=1)
    for j in (0,2,4):
        total=np.linalg.norm(reference[j]+reference[j+1])
        denominator[j:j+2]=np.where(denominator[j:j+2]<1e-6*total,total,denominator[j:j+2])
    denominator[6:]=np.linalg.norm(S0)
    difference=np.linalg.norm((current-reference).reshape(9,-1),axis=1)
    return difference/np.maximum(denominator,1e-30)


def _period_from_features(time_values,features):
    """Candidate only. Full density/Stokes maps must independently recur."""
    tt=np.asarray(time_values);values=np.asarray(features)
    if len(tt)<100:return None
    cadence=float(np.median(np.diff(tt)))
    if values.shape[1]==17:
        # Keep amplitudes in physical relative units; a tiny reservoir drift
        # must not receive the same weight as an order-one Stokes oscillation.
        totals=[max(float(np.mean(values[:,j:j+2].sum(1))),1e-12) for j in (0,2,4)]
        scale=np.r_[np.repeat(totals[0],2),np.repeat(totals[1],2),np.repeat(totals[2],2),
                    np.repeat(totals[0],3),np.ones(8)]
        y=(values-values.mean(0))/scale;span=np.ptp(y,axis=0)
        valid=span>max(1e-6,.01*span.max())
        if not valid.any():return None
        y=y[:,valid]
    else:
        scale=np.maximum(np.mean(abs(values),axis=0),1e-12)
        span=np.ptp(values,axis=0);valid=span>np.maximum(1e-7*scale,1e-10)
        if not valid.any():return None
        y=values[:,valid];y=(y-y.mean(axis=0))/np.maximum(span[valid],1e-30)
    # Correlation normalized by overlapping sample count, without a zero-lag peak.
    n=len(y);f=np.fft.rfft(y,n=2*n,axis=0)
    ac=np.fft.irfft(f.conj()*f,n=2*n,axis=0)[:n].sum(axis=1)/np.arange(n,0,-1)
    peaks,_=find_peaks(ac[:n//3],prominence=.05*max(ac[0],1e-30))
    peaks=peaks[peaks*cadence>=50.]
    if not len(peaks):return None
    candidates=[]
    def error(period):
        use=tt+period<=tt[-1];a=y[use]
        b=np.column_stack([np.interp(tt[use]+period,tt,y[:,j]) for j in range(y.shape[1])])
        return float(np.mean((a-b)**2))
    for lag in peaks[:8]:
        fit=minimize_scalar(error,bounds=((lag-1)*cadence,(lag+1)*cadence),method='bounded')
        candidates.append((fit.fun,float(fit.x)))
    best=min(v[0] for v in candidates)
    eligible=[v for v in candidates if v[0]<=max(4*best,1e-8)]
    return min(eligible,key=lambda v:v[1])[1]


def _lag_polynomial_weights(times,target,count=6):
    """Local polynomial interpolation in time, never of complex field phases."""
    count=min(count,len(times));j=int(np.searchsorted(times,target))
    start=max(0,min(j-count//2,len(times)-count));indices=np.arange(start,start+count)
    nodes=np.asarray(times)[indices];weights=np.ones(count)
    for a in range(count):
        for b in range(count):
            if a!=b:weights[a]*=(target-nodes[b])/(nodes[a]-nodes[b])
    return indices,weights


def analyze_report3(store,history,feature_rows,end,config,common_window=None,stationarity_only=False):
    """Two consecutive windows; density, polarization and periodicity are separate."""
    if not stationarity_only:
        static=analyze_report3(store,history,feature_rows,end,config,common_window,stationarity_only=True)
        if static['status']=='STATIONARY':return static
    minimum=config['analysis']['min_window'];h=np.asarray(history);f=np.asarray(feature_rows)
    candidate=None
    if not stationarity_only and len(f)>100:
        late=f[:,0]>=max(0.,end-2000.)
        candidate=_period_from_features(f[late,0],f[late,1:])
    window=minimum if candidate is None else max(minimum,5*candidate)
    window=np.ceil(max(window,common_window or 0.)/store.cadence)*store.cadence
    result={'status':'UNRESOLVED','candidate_period_ps':candidate,'window_length_ps':float(window),
            'common_interval_ps':[float(end-2*window),float(end)],'windows':[]}
    ordered=store.ordered()
    if not len(ordered) or store.times[ordered[0]]>end-2*window+1e-8:
        result['reason']='Insufficient full-map history for two required windows';return result
    for left,right in ((end-2*window,end-window),(end-window,end)):
        indices=store.ordered(left,right);ref=store.state(indices[0]);maps,S0=_report3_maps(ref)
        maximum=np.zeros(9);field_error=0.;map_history=[]
        for i in indices[1:]:
            state=store.state(i);current,_=_report3_maps(state)
            errors=_report3_map_errors(current,maps,S0)
            maximum=np.maximum(maximum,errors);map_history.append([float(store.times[i]),*errors.tolist()])
            if parameters(config)['J']!=0 or parameters(config)['beta']!=0:
                field_error=max(field_error,aligned_field_error(state[0],ref[0])[0])
        rows=h[(h[:,0]>=left-1e-8)&(h[:,0]<=right+1e-8)]
        pop=rows[:,1:7];scale=np.maximum(np.mean(abs(pop),axis=0),1e-30)
        for j in (0,2,4):
            total=scale[j:j+2].sum()
            scale[j:j+2]=np.where(scale[j:j+2]<1e-6*total,total,scale[j:j+2])
        half=len(pop)//2;span=np.ptp(pop,axis=0)/scale
        drift=abs(pop[:half].mean(0)-pop[half:].mean(0))/scale
        density=bool(max(span.max(),drift.max())<.005 and maximum[:6].max()<.01)
        polarization=bool(maximum[6:].max()<.01)
        result['windows'].append({'interval_ps':[float(left),float(right)],
            'population_span':span,'population_half_mean_drift':drift,'map_errors':maximum,'map_error_history':map_history,
            'density_stationary':density,'polarization_stationary':polarization,
            'common_phase_aligned_field_error':field_error,
            'field_stationary':bool(field_error<.01)})
    result['density_stationary']=all(v['density_stationary'] for v in result['windows'])
    result['polarization_stationary']=all(v['polarization_stationary'] for v in result['windows'])
    if result['density_stationary'] and result['polarization_stationary'] and all(v['field_stationary'] for v in result['windows']):
        result['status']='STATIONARY';result['period_ps']=None;return result
    if candidate is None:
        result['reason']='No repeatable period candidate in measured features';return result
    # A candidate is accepted only after map recurrence throughout BOTH windows.
    periods=[];amplitudes=[]
    for left,right in [v['interval_ps'] for v in result['windows']]:
        use=(f[:,0]>=left)&(f[:,0]<=right)
        period=_period_from_features(f[use,0],f[use,1:])
        periods.append(period);amplitudes.append(np.ptp(f[use,1:],axis=0))
    if any(v is None for v in periods):
        result['reason']='Period cannot be estimated independently in both windows';return result
    period_error=abs(periods[1]-periods[0])/max(periods)
    amplitude_scale=np.maximum(np.maximum(*amplitudes),1e-6*np.mean(abs(f[f[:,0]>=end-2*window,1:]),axis=0))
    if f.shape[1]==18:
        # Float32 jitter in a constant population must not determine the
        # amplitude stability of a clearly oscillating polarization channel.
        late=abs(f[f[:,0]>=end-2*window,1:]).mean(0)
        totals=[max(float(late[j:j+2].sum()),1e-12) for j in (0,2,4)]
        physical_scale=np.r_[np.repeat(totals[0],2),np.repeat(totals[1],2),np.repeat(totals[2],2),
                             np.repeat(totals[0],3),np.ones(8)]
        relative_span=np.maximum(*amplitudes)/physical_scale
        significant=relative_span>max(1e-6,.01*relative_span.max())
    else:significant=np.ones(len(amplitude_scale),dtype=bool)
    amplitude_error=float(np.max((abs(amplitudes[1]-amplitudes[0])/np.maximum(amplitude_scale,1e-12))[significant])) if significant.any() else 0.
    maximum=np.zeros(9);interpolation_difference=np.zeros(9);ts=store.times[ordered];recurrence_history=[]
    for i in store.ordered(end-2*window+candidate,end):
        lag=store.times[i]-candidate
        ix6,w6=_lag_polynomial_weights(ts,lag,8);ix4,w4=_lag_polynomial_weights(ts,lag,6)
        ref=np.zeros((9,*store.maps.shape[-2:]));S0=np.zeros(store.maps.shape[-2:]);ref4=np.zeros_like(ref)
        for j,weight in zip(ix6,w6):
            maps,s0=_report3_maps(store.state(ordered[j]));ref+=weight*maps;S0+=weight*s0
            take=np.flatnonzero(ix4==j)
            if len(take):ref4+=w4[take[0]]*maps
        actual,_=_report3_maps(store.state(i))
        errors=_report3_map_errors(actual,ref,S0)
        interpolation_difference=np.maximum(interpolation_difference,_report3_map_errors(ref4,ref,S0))
        maximum=np.maximum(maximum,errors);recurrence_history.append([float(store.times[i]),*errors.tolist()])
    result['periodic']={'periods_ps':periods,'period_relative_difference':period_error,
                        'amplitude_relative_difference':amplitude_error,'amplitude_significant_channels':np.flatnonzero(significant),
                        'map_recurrence_errors':maximum,
                        'map_cadence_ps':store.cadence,'map_recurrence_history':recurrence_history,
                        'interpolation_difference_6_8':interpolation_difference,
                        'lag_interpolation':'local eight-point polynomial in densities and unnormalized Stokes; six-point cross-check'}
    if period_error<.01 and amplitude_error<.01 and maximum.max()<.01 and interpolation_difference.max()<.002:
        result['status']='PERIODIC';result['period_ps']=float(np.mean(periods))
    else:result['reason']='Period, amplitude or full-map recurrence not yet stable'
    return result


def _report3_gpu_observables(backend,state,roi):
    """Only small scalar/angular arrays leave Arc at the specified cadence."""
    t=backend.torch;z,a,b=state;g=backend.grid
    n=z.real*z.real+z.imag*z.imag;coherence=z[0].conj()*z[1]
    S=t.stack((n.sum(0),2*coherence.real,2*coherence.imag,n[0]-n[1]))
    pop=t.cat((n.sum((-2,-1),dtype=t.float64),a.sum((-2,-1),dtype=t.float64),b.sum((-2,-1),dtype=t.float64)))*g.da
    total=S[0].sum(dtype=t.float64);integrated=S[1:].sum((-2,-1),dtype=t.float64)*g.da
    roi_power=(S[0]*roi['mask']).sum(dtype=t.float64)
    c2=(S[3]*roi['harmonic']).sum(dtype=t.complex128)/t.clamp(roi_power,min=1e-30)
    intensity_c2=(S[0]*roi['harmonic']).sum(dtype=t.complex128)/t.clamp(roi_power,min=1e-30)
    # Additional moments help detect motion invisible in integrated populations.
    moments=t.stack([(S[i]*roi['harmonic']).sum(dtype=t.complex128)/t.clamp(total,min=1e-30) for i in (1,2)])
    values=t.cat((pop,integrated,t.stack((c2.real,c2.imag,intensity_c2.real,intensity_c2.imag)),
                  t.view_as_real(moments).ravel(),t.stack((a.min(),b.min()))))
    return values.detach().cpu().numpy()


def _report3_roi(backend,limits):
    g=backend.grid;xx,yy=np.meshgrid(g.x,g.y);r=np.hypot(xx,yy);phi=np.arctan2(yy,xx)
    mask=(r>=limits[0])&(r<=limits[1])
    sectors=np.floor((phi+np.pi)/(2*np.pi)*72).astype(int)%72
    return {'limits_um':list(limits),'mask':backend.tensor(mask.astype(float)),
            'harmonic':backend.tensor(mask*np.exp(-2j*phi)),
            'sectors':sectors,'counts':np.bincount(sectors[mask],minlength=72)}


def run_report3_pilot(config,name,*,resume=True,max_steps=None,roi_limits=None,common_window=None,deadline=None):
    """One independent resumable XPU trajectory for report 3."""
    path=ROOT/'results/runs/report3'/name;path.mkdir(parents=True,exist_ok=True)
    export=ROOT/'results/data/report3';export.mkdir(parents=True,exist_ok=True)
    p=parameters(config);g=make_grid(config);dt=config['time']['dt'];end=config['time']['final']
    if not np.isclose(round(end/dt)*dt,end):raise ValueError('Final time must align with dt')
    signature=copy.deepcopy(config);signature['time'].pop('final')
    signature=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    source_hash=hashlib.sha256((ROOT/'model.py').read_bytes()+(ROOT/'numerics.py').read_bytes()).hexdigest()
    checkpoint=path/'checkpoint.npz';step=0;elapsed=0.;integral=0.;scale=0.;cond=0.;cond_scale=0.
    if not resume:checkpoint.unlink(missing_ok=True)
    history=[];features=[];spatial=[];times=None;initial=initial_state(config,g)
    initial_pop=component_populations(initial,g);initial_total=float(initial_pop.sum());initial_nc=float(initial_pop[0].sum())
    state_np=initial;limits=roi_limits or ((5.,15.) if config['pump']['kind']=='gaussian' else (5.,7.5))
    if checkpoint.exists() and resume:
        with np.load(checkpoint) as z:
            if str(z['signature'])!=signature:raise ValueError('Pilot checkpoint parameters changed')
            state_np=tuple(z[k].copy() for k in ('psi','active','inactive'));step=int(z['step'])
            elapsed=float(z['elapsed']);integral=float(z['integral']);scale=float(z['scale'])
            cond=float(z['cond_integral']);cond_scale=float(z['cond_scale'])
            history=z['history'].tolist();features=z['features'].tolist()
            spatial=json.loads(str(z['spatial_json']));times=z['probe_times'].copy();limits=z['roi_limits'].tolist()
    if step>round(end/dt):raise ValueError('Requested final time precedes pilot checkpoint')
    backend=TorchSpinStepper(config,g,compiled=config['execution'].get('compiled_local',True))
    roi=_report3_roi(backend,limits);state=backend.to_device(state_np)
    store=_ProbeStore(path/'probe_maps.npy',len(g.x),cadence=config['save']['probe'],times=times)
    if common_window is not None:store.ensure(min(config['analysis']['max_final'],2*common_window+10.))
    metadata={'signature':signature,'source_sha256':source_hash,'derived':p,'backend':'torch-xpu',
              'torch':backend.torch.__version__,'device':backend.torch.xpu.get_device_name(),
              'precision':backend.precision,'reduction_precision':'float64','compiled_local':backend.compiled,
              'initial_kind':config['initial'].get('kind','plane_waves'),'initial_NC':initial_nc,
              'PA_integrals':injection(config,g).sum((-2,-1))*g.da,
              'Pin_integrals':p['injection_ratio']*injection(config,g).sum((-2,-1))*g.da,
              'roi_limits_um':limits,'roi_sector_node_counts':roi['counts'],
              'method':'exact kinetic+TE-TM half / joint local RK4 / exact half',
              'Stokes':'S2=2 Im(conj(psi_plus)*psi_minus); Si unnormalized; si=Si/S0',
              'units':{'length':'um','time':'ps','energy':'meV','density':'um^-2'},
              'thermal_stop':False,'merged_adjacent_FFT_half_steps':True}
    write_json(path/'run_metadata.json',metadata)
    (path/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    source=backend.torch.zeros((),device=backend.device,dtype=backend.torch.float64)
    normalizer=source.clone();cg=source.clone();cs=source.clone()
    sample_steps=round(config['save']['observable']/dt);probe_steps=round(config['save']['probe']/dt)
    angular_steps=round(config['save']['angular']/dt);check_steps=round(config['save']['checkpoint']/dt)
    diag_steps=round(config['save']['diagnostic']/dt)
    for interval in (config['save']['observable'],config['save']['probe'],config['save']['angular'],config['save']['checkpoint'],config['save']['diagnostic']):
        if interval<dt or not np.isclose(round(interval/dt)*dt,interval):raise ValueError('Save cadence must align with dt')
    angular=[]
    angular_file=path/'angular.npz'
    if angular_file.exists():
        with np.load(angular_file) as z:
            rows=z['rows']
            if rows.size:angular=rows[rows[:,0]<=step*dt+1e-8].tolist()
    stop=round(end/dt) if max_steps is None else min(round(end/dt),step+max_steps)
    import signal,threading
    interrupted=False;old_handler=None
    def interrupt(*_):
        nonlocal interrupted
        interrupted=True
    if threading.current_thread() is threading.main_thread():old_handler=signal.signal(signal.SIGINT,interrupt)
    start=time.perf_counter();stage='RUNNING'
    def checkpoint_save():
        store.maps.flush()
        np.savez(path/'checkpoint.tmp.npz',signature=signature,step=step,
                 psi=state_np[0],active=state_np[1],inactive=state_np[2],elapsed=elapsed+time.perf_counter()-start,
                 integral=integral,scale=scale,cond_integral=cond,cond_scale=cond_scale,
                 history=np.asarray(history),features=np.asarray(features),spatial_json=json.dumps(spatial),
                 probe_times=store.times,roi_limits=limits)
        (path/'checkpoint.tmp.npz').replace(checkpoint)
        write_json(path/'progress.json',{'time_ps':step*dt,'target_ps':end,
                   'elapsed_wall_s':elapsed+time.perf_counter()-start,'backend':'Intel Arc '+backend.precision,
                   'status':stage})
        np.savez_compressed(path/'angular.npz',rows=np.asarray(angular).reshape(-1,145),sector_centers=np.linspace(-np.pi,np.pi,72,endpoint=False)+np.pi/72)
        if end>=1000:save_report3_animation_frames(path)
    try:
        if step==0:
            v=_report3_gpu_observables(backend,state,roi)
            history.append([0.,*v[:6],0.,0.,0.,0.]);features.append([0.,*v[:17]])
            store.add(0.,state_np)
        for next_step in range(step+1,stop+1):
            if deadline is not None and next_step%sample_steps==0 and time.monotonic()>=deadline:
                interrupted=True
            close=next_step%sample_steps==0 or next_step==stop or interrupted
            state,ds,dn=backend.step(state,close=close)
            step=next_step  # Commit time only after a successful propagation step.
            source+=ds;normalizer+=dn
            cg+=backend.last_condensate[0];cs+=backend.last_condensate[1]
            now=step*dt
            if step%sample_steps==0 or step==stop or (interrupted and not backend.staggered):
                v=_report3_gpu_observables(backend,state,roi)
                integral+=float(source.cpu());scale+=float(normalizer.cpu());cond+=float(cg.cpu());cond_scale+=float(cs.cpu())
                source.zero_();normalizer.zero_();cg.zero_();cs.zero_()
                if not np.isfinite(v).all() or min(v[-2:]) < -1e-10:
                    raise FloatingPointError('Nonfinite fields or negative reservoir; no clipping')
                history.append([now,*v[:6],integral,scale,cond,cond_scale]);features.append([now,*v[:17]])
            if step%angular_steps==0:
                z=state[0];n=z.real*z.real+z.imag*z.imag
                mask=roi['mask'].bool();sector=backend.torch.as_tensor(roi['sectors'][np.asarray(roi['mask'].cpu(),bool)],device=backend.device)
                total=backend.torch.bincount(sector,weights=n.sum(0)[mask],minlength=72)
                difference=backend.torch.bincount(sector,weights=(n[0]-n[1])[mask],minlength=72)
                angular.append([now,*total.cpu().numpy()*g.da,*difference.cpu().numpy()*g.da])
            if step%probe_steps==0:
                state_np=backend.to_numpy(state);store.add(now,state_np)
            if step%diag_steps==0:
                if step%probe_steps:state_np=backend.to_numpy(state)
                checks=spatial_checks(state_np,g,p)
                power=abs(fft.fft2(state_np[0]))**2
                high=(abs(g.kx[None,:])>.7*np.pi/g.dx)|(abs(g.ky[:,None])>.7*np.pi/g.dx)
                checks['high_k']=float(power[:,high].sum()/max(float(power.sum()),1e-30))
                for j,component in enumerate(checks['components']):component['high_k']=float(power[j,high].sum()/max(float(power[j].sum()),1e-30))
                spatial.append({'time':now,**checks})
            if step%check_steps==0:
                state_np=backend.to_numpy(state)
                # Adapt retention before the final classification, using measured features.
                f=np.asarray(features);mask=f[:,0]>=max(0,now-2000.)
                period=_period_from_features(f[mask,0],f[mask,1:])
                if period is not None:store.ensure(min(6000.,max(400.,10*period+10.)))
                checkpoint_save()
                print(f'{name}: {now:g}/{end:g} ps, {(elapsed+time.perf_counter()-start)/60:.2f} min',flush=True)
            if interrupted and not backend.staggered:break
        stage='RUN_COMPLETED' if step==round(end/dt) else 'RUN_INCOMPLETE'
    except Exception:
        stage='RUN_FAILED'
        # A failed GPU step may leave asynchronous state unusable. Preserve the
        # last atomic checkpoint instead of overwriting it with a partial step.
        saved_step=0
        if checkpoint.exists():
            with np.load(checkpoint) as saved:saved_step=int(saved['step'])
        write_json(path/'progress.json',{'time_ps':saved_step*dt,'target_ps':end,
                   'backend':'Intel Arc '+backend.precision,'status':stage,
                   'checkpoint_preserved':True})
        if old_handler is not None:signal.signal(signal.SIGINT,old_handler)
        raise
    finally:
        if stage!='RUN_FAILED':
            state=backend.physical_state(state)
            state_np=backend.to_numpy(state)
            # Include the final fields even when interrupted between regular probes.
            if not np.any(np.isclose(store.times,step*dt,rtol=0,atol=1e-8)):store.add(step*dt,state_np)
            checkpoint_save()
            # Once the physical state is committed, a pause can interrupt the
            # expensive CPU analysis immediately. Resume rebuilds exports only.
            if old_handler is not None:signal.signal(signal.SIGINT,old_handler)
            if stage=='RUN_INCOMPLETE' or interrupted:
                stage='RUN_INCOMPLETE'
                summary={'run_status':stage,'time_ps':step*dt,
                         'elapsed_wall_s':elapsed+time.perf_counter()-start,
                         'analysis':{'status':'PENDING','reason':'Paused; checkpoint saved'}}
                write_json(path/'summary.json',summary)
                return {'path':path,'summary':summary}
            pop=component_populations(state_np,g)
            analysis=analyze_report3(store,history,features,step*dt,config,common_window)
            balance=abs(pop.sum()-initial_total-integral)/max(scale,1e-30)
            c_balance=abs(pop[0].sum()-initial_nc-cond)/max(cond_scale,1e-30)
            tail=[v for v in spatial if v['time']>=step*dt-2*analysis['window_length_ps']]
            edge=max((v['edge_mass'] for v in tail),default=1.)
            high=max((v['high_k'] for v in tail),default=1.)
            last=np.asarray(history);late=last[last[:,0]>=step*dt-400]
            mean_nc=float(late[:,1:3].sum(1).mean())
            midpoint=max(1,len(late)//2)
            drift=float(late[midpoint:,1:3].sum(1).mean()/max(late[:midpoint,1:3].sum(1).mean(),1e-300))
            survival=(mean_nc>=100*initial_nc and (drift>=.995 or analysis['status']=='PERIODIC')
                      and balance<1e-3 and c_balance<1e-3)
            summary={'run_status':stage,'time_ps':step*dt,'elapsed_wall_s':elapsed+time.perf_counter()-start,
                     'NC':pop[0],'NA':pop[1],'NI':pop[2],'Z':float((pop[0,0]-pop[0,1])/max(pop[0].sum(),1e-300)),
                     'total_balance_relative':float(balance),'condensate_balance_relative':float(c_balance),
                     'edge_mass_max_late':float(edge),'high_k_max_late':float(high),
                     'analysis':analysis,'nonzero_condensate':bool(survival),
                     'late_NC_to_seed':mean_nc/max(initial_nc,1e-300),'late_half_NC_ratio':drift,
                     'roi_intensity_fraction':float((abs(state_np[0])**2).sum(0)[np.asarray(roi['mask'].cpu(),bool)].sum()/max(pop[0].sum()/g.da,1e-300)),
                     'numerical_checks':'PASS' if balance<1e-3 and c_balance<1e-3 and edge<1e-5 and high<1e-6 else 'NEEDS_REVIEW'}
            write_json(path/'summary.json',summary);write_json(export/(name+'.json'),{'config':config,'metadata':metadata,'summary':summary})
            S=stokes(state_np[0]);ix=store.ordered(step*dt-analysis['window_length_ps'],step*dt)
            means=np.zeros((10,len(g.y),len(g.x)))
            ts=store.times[ix];weights=np.ones(len(ix))
            if len(ix)>1:
                weights=np.r_[np.diff(ts)[0]/2,(ts[2:]-ts[:-2])/2,np.diff(ts)[-1]/2]
                weights/=ts[-1]-ts[0]
            for weight,i in zip(weights,ix):
                ss=store.state(i);maps,s0=_report3_maps(ss)
                means+=weight*np.concatenate((s0[None],maps))
            desired=[step*dt-2*analysis['window_length_ps'],step*dt-analysis['window_length_ps'],step*dt]
            if analysis.get('candidate_period_ps'):
                desired.extend(step*dt-analysis['candidate_period_ps']*v/4 for v in (1,2,3,4))
            valid=store.ordered()
            selected=sorted({int(valid[np.argmin(abs(store.times[valid]-v))]) for v in desired},key=lambda i:store.times[i])
            shots=[store.state(i) for i in selected]
            np.savez_compressed(export/(name+'.tmp.npz'),psi=state_np[0],active=state_np[1],inactive=state_np[2],
                                x=g.x,y=g.y,history=np.asarray(history),features=np.asarray(features),angular=np.asarray(angular).reshape(-1,145),
                                selected_times=store.times[selected],selected_psi=np.array([v[0] for v in shots]),
                                selected_active=np.array([v[1] for v in shots]),selected_inactive=np.array([v[2] for v in shots]),
                                history_columns=np.array(['time','NC_plus','NC_minus','NA_plus','NA_minus','NI_plus','NI_minus','total_source_integral','total_scale','condensate_source_integral','condensate_scale']),
                                feature_columns=np.array(['time','NC_plus','NC_minus','NA_plus','NA_minus','NI_plus','NI_minus','integral_S1','integral_S2','integral_S3','C2_real','C2_imag','S0_m2_real','S0_m2_imag','S1_m2_real','S1_m2_imag','S2_m2_real','S2_m2_imag']),
                                spatial_checks_json=json.dumps(spatial),
                                mean_interval_ps=ts[[0,-1]],mean_maps=means,mean_map_names=np.array(['S0','n_plus','n_minus','A_plus','A_minus','I_plus','I_minus','S1','S2','S3']),
                                stokes=S,config_json=json.dumps(config),summary_json=json.dumps(summary,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),
                                metadata_json=json.dumps(metadata,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
            (export/(name+'.tmp.npz')).replace(export/(name+'.npz'))
            if old_handler is not None:signal.signal(signal.SIGINT,old_handler)
    return {'path':path,'summary':summary}


def _report3_choose_roi(path,geometry):
    """Density-only B0 rule, fixed for all further cases of the geometry."""
    with np.load(Path(path)/'checkpoint.npz') as z:density=(abs(z['psi'])**2).sum(0)
    config=yaml.safe_load((Path(path)/'resolved_config.yaml').read_text());g=make_grid(config)
    r=np.hypot(g.x[None,:],g.y[:,None]);total=max(float(density.sum()),1e-300)
    limits=(5.,15.) if geometry=='spot' else (5.,7.5)
    fraction=float(density[(r>=limits[0])&(r<=limits[1])].sum()/total)
    if fraction>=.01:return {'limits_um':limits,'fraction_B0':fraction,'rule':'specified annulus','valid':True}
    bins=np.arange(0.,g.x.max()+g.dx,g.dx)
    radial,_=np.histogram(r,bins,weights=density);centers=(bins[:-1]+bins[1:])/2
    radial[centers<2*g.dx]=0.;center=centers[np.argmax(radial)]
    limits=(max(2*g.dx,center-2.5),center+2.5)
    fraction=float(density[(r>=limits[0])&(r<=limits[1])].sum()/total)
    return {'limits_um':limits,'fraction_B0':fraction,'rule':'B0 radial-particle maximum +/- 2.5 um','valid':fraction>=.01}


def run_report3_preliminary(*,pump_values=(4.,8.,16.),final=3000.,max_final=6000.):
    """ONLY B0/A1/T0 pilots. Remaining matrix is deliberately not dispatched."""
    out=ROOT/'results/data/report3';out.mkdir(parents=True,exist_ok=True)
    check_file=out/'analytic_checks.json'
    if not check_file.exists() or json.loads(check_file.read_text())['status']!='PASS':
        raise RuntimeError('Run and save tetm_reference_checks(device="xpu",compiled=True) first')
    report={'status':'RUNNING','stage':'B0/A1/T0 only','cases':{},'pump_attempts':[],
            'selected_p':None,'precision':'complex128/float64','thermal_stop':False,
            'remaining_series_authorized':False}
    manifest=out/'pilot_summary.json'
    write_json(manifest,report)
    def launch(case,geometry,power,end,roi=None,window=None,fresh=False):
        name=f'p{power:g}_{case}_{geometry}'
        c=report3_config(case,geometry,power,time={'final':float(end)})
        result=run_report3_pilot(c,name,roi_limits=roi,common_window=window,resume=not fresh)
        report['cases'][name]=result['summary'];write_json(manifest,report)
        if result['summary']['run_status']!='RUN_COMPLETED':
            report['status']='RUN_INCOMPLETE';write_json(manifest,report)
            return None
        return result
    selected=None;rois={}
    for power in pump_values:
        report['pump_attempts'].append(float(power));baseline=[]
        for geometry in ('spot','ring'):
            for case in ('B0','A1'):
                result=launch(case,geometry,power,final,None if case=='B0' else rois[geometry]['limits_um'])
                if result is None:return report
                if case=='B0':
                    rois[geometry]=_report3_choose_roi(result['path'],geometry)
                    default=(5.,15.) if geometry=='spot' else (5.,7.5)
                    if (result['summary']['nonzero_condensate'] and
                        tuple(rois[geometry]['limits_um'])!=default):
                        # The density-only ROI rule is fixed before polarization.
                        # Repeat this same deterministic B0 start, rather than
                        # mixing angular histories from two different annuli.
                        result=launch(case,geometry,power,final,rois[geometry]['limits_um'],fresh=True)
                        if result is None:return report
                # A declining but significant population is not diagnosed by an endpoint.
                end=final
                while (not result['summary']['nonzero_condensate'] and
                       result['summary']['late_NC_to_seed']>=100 and end<max_final):
                    end=min(end+1000,max_final)
                    result=launch(case,geometry,power,end,rois[geometry]['limits_um'])
                    if result is None:return report
                baseline.append(result['summary'])
        if all(v['nonzero_condensate'] for v in baseline):
            selected=power;break
        if any(not v['nonzero_condensate'] and v['late_NC_to_seed']>=100 for v in baseline):
            report['status']='PUMP_SURVIVAL_UNRESOLVED'
            report['reason']='Significant condensate remains; do not raise p merely for nonstationarity'
            write_json(manifest,report);return report
    if selected is None:
        report['status']='NO_COMMON_PUMP';write_json(manifest,report);return report
    report['selected_p']=float(selected);report['roi']=rois;write_json(manifest,report)
    for geometry in ('spot','ring'):
        result=launch('T0',geometry,selected,final,rois[geometry]['limits_um'])
        if result is None:return report
    for geometry in ('spot','ring'):
        names=[f'p{selected:g}_{case}_{geometry}' for case in ('B0','A1','T0')]
        end=max(report['cases'][name]['time_ps'] for name in names)
        while True:
            window=max(report['cases'][name]['analysis']['window_length_ps'] for name in names)
            # Re-analyze/resume all comparison participants on one common interval.
            for case in ('B0','A1','T0'):
                name=f'p{selected:g}_{case}_{geometry}'
                result=launch(case,geometry,selected,end,rois[geometry]['limits_um'],window)
                if result is None:return report
            settled=all(report['cases'][name]['analysis']['status'] in ('STATIONARY','PERIODIC') for name in names)
            if (settled and max(report['cases'][name]['analysis']['window_length_ps'] for name in names)<=window) or end>=max_final:break
            end=min(end+1000,max_final)
        report.setdefault('comparison_intervals',{})[geometry]=[float(end-2*window),float(end)]
    report['status']='COMPLETED'
    report['assessment']='REVIEW_PILOTS_BEFORE_REMAINING_SERIES'
    report['estimated_remaining_spatial_jobs_minimum']=19
    report['measured_min_per_3000ps']={name:row['elapsed_wall_s']/60*3000/row['time_ps'] for name,row in report['cases'].items()}
    write_json(manifest,report)
    return report


def save_report3_animation_frames(path, *, cadence=20., halfwidth=40.):
    """Append cropped n+/n- at real saved times; never interpolate or change psi.

    Reads committed checkpoints and their rolling probes. This also allows an
    already running, older solver to supply animation frames without a restart.
    """
    import fcntl
    path=Path(path);checkpoint=path/'checkpoint.npz'
    if not checkpoint.exists():return None
    output=ROOT/'results/data/report3';output.mkdir(parents=True,exist_ok=True)
    lock_path=ROOT/'results/runs/report3/.animation_capture.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=checkpoint.stat()
        config=yaml.safe_load((path/'resolved_config.yaml').read_text())
        if config['time']['final']<1000:return None
        with np.load(checkpoint,allow_pickle=False) as z:
            times=z['probe_times'].copy();end=float(z['step'])*config['time']['dt']
            signature=str(z['signature'])
        grid=make_grid(config);ix=np.flatnonzero(abs(grid.x)<=halfwidth)
        iy=np.flatnonzero(abs(grid.y)<=halfwidth)
        destination=output/(path.name+'_animation.npz');tt=[];frames=[]
        if destination.exists():
            with np.load(destination,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata_json']))
                if meta['signature']==signature:
                    if not (np.array_equal(z['x'],grid.x[ix]) and np.array_equal(z['y'],grid.y[iy])):
                        raise ValueError('Animation crop/grid changed within a run')
                    valid=z['time']<=end+1e-8
                    tt=z['time'][valid].tolist();frames=list(z['density'][valid])
        maps=np.load(path/'probe_maps.npy',mmap_mode='r',allow_pickle=False)
        if len(maps)!=len(times):return None  # Producer is resizing its probe buffer.
        eligible=np.flatnonzero(np.isfinite(times)&(times<=end+1e-8))
        eligible=eligible[np.isclose(times[eligible]/cadence,np.rint(times[eligible]/cadence),rtol=0,atol=1e-9)]
        added=0
        for i in eligible[np.argsort(times[eligible])]:
            if tt and times[i]<=tt[-1]+1e-8:continue
            planes=maps[i,:4][:,iy[:,None],ix]
            density=np.stack((planes[0]**2+planes[1]**2,planes[2]**2+planes[3]**2)).astype(np.float32)
            if not np.isfinite(density).all():raise FloatingPointError('Invalid animation density')
            tt.append(float(times[i]));frames.append(density);added+=1
        after=checkpoint.stat()
        if (before.st_ino,before.st_mtime_ns,before.st_size)!=(after.st_ino,after.st_mtime_ns,after.st_size):
            return None  # Retry the newer committed checkpoint next time.
        if not tt:return None
        expected=set(range(int(np.floor((end+1e-8)/cadence))+1))
        missing=sorted(expected-{round(t/cadence) for t in tt})
        metadata={'quantity':'n_plus=abs(psi_plus)^2; n_minus=abs(psi_minus)^2',
                  'units':'um^-2','signature':signature,'cadence_ps':cadence,
                  'crop_halfwidth_um':halfwidth,'source_N':len(grid.x),'source_L_um':config['space']['L'],
                  'density_dtype':'float32','solver_precision':'complex128/float64',
                  'purpose':'density animations only; unnormalized, no temporal interpolation',
                  'source_checkpoint_ps':end,'first_time_ps':tt[0],'last_time_ps':tt[-1],
                  'frame_count':len(tt),'missing_times_ps':[v*cadence for v in missing],
                  'complete_on_saved_cadence':not missing}
        temporary=destination.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,time=np.asarray(tt),density=np.asarray(frames,dtype=np.float32),
                            x=grid.x[ix],y=grid.y[iy],components=np.array(['+','-']),
                            metadata_json=json.dumps(metadata))
        temporary.replace(destination)
        return {'archive':destination.name,'frames':len(tt),'added':added,
                'first_ps':tt[0],'last_ps':tt[-1],'missing_frames':len(missing)}


def watch_report3_animation_frames(producer_pid, *, exclude=(), poll_seconds=10.):
    """Lightweight recorder for the existing in-memory batch; no GPU work."""
    root=ROOT/'results/runs/report3';seen={};records={};excluded=set(exclude)
    process=Path('/proc')/str(producer_pid)
    def identity():
        try:return (process/'stat').read_text().rsplit(')',1)[1].split()[19]
        except (OSError,IndexError):return None
    original=identity()
    if original is None:raise RuntimeError('Producer is not running')
    while True:
        alive=identity()==original
        for checkpoint in sorted(root.glob('*/checkpoint.npz')):
            name=checkpoint.parent.name
            if name in excluded:continue
            try:stamp=checkpoint.stat().st_mtime_ns
            except OSError:continue
            if seen.get(name)==stamp:continue
            try:
                row=save_report3_animation_frames(checkpoint.parent)
                if row is not None:
                    records[name]=row;seen[name]=stamp
                    print(name,row,flush=True)
            except (OSError,ValueError,KeyError,FloatingPointError) as error:
                records[name]={'error':str(error)}
                print('Animation capture will retry:',name,str(error),flush=True)
        write_json(root/'animation_capture.json',{'status':'WATCHING' if alive else 'FINISHED',
                   'producer_pid':producer_pid,'excluded_cases':sorted(excluded),'cases':records})
        if not alive:break
        time.sleep(poll_seconds)
    return records


def frozen_spin_reservoirs(config,grid):
    """Exact stationary reservoir branch at psi=0, including both spin flips."""
    p=parameters(config);pump=injection(config,grid)*p['injection_ratio']
    isum=pump.sum(0)/(p['gammaI']+p['W'])
    idiff=(pump[0]-pump[1])/(p['gammaI']+p['W']+2*p['GammaI'])
    inactive=np.stack(((isum+idiff)/2,(isum-idiff)/2))
    asum=p['W']*isum/p['gammaA'];adiff=p['W']*idiff/(p['gammaA']+2*p['GammaA'])
    active=np.stack(((asum+adiff)/2,(asum-adiff)/2))
    return active,inactive


def _threshold_seed(config,grid,seed):
    """Same continuous broad seed on all grids, not the nonlinear radial start."""
    from model import pump_profile
    rng=np.random.default_rng(seed);waves=np.vstack((.6*rng.normal(size=(2,24)),rng.uniform(0,2*np.pi,24)))
    z=np.zeros(grid.k2.shape,complex)
    for kx,ky,phase in waves.T:z+=np.exp(1j*(kx*grid.x[None,:]+ky*grid.y[:,None]+phase))/np.sqrt(24)
    return z*np.sqrt(pump_profile(config,grid))


def _growth_fit(times,values,window=100.):
    t=np.asarray(times);y=np.asarray(values);fits=[]
    if t[-1]<2*window:return {'status':'UNRESOLVED','reason':'short history'}
    for left,right in ((t[-1]-2*window,t[-1]-window),(t[-1]-window,t[-1])):
        take=(t>=left-1e-8)&(t<=right+1e-8);x=t[take];v=y[take]
        slope,intercept=np.polyfit(x,v,1);residual=v-(slope*x+intercept)
        error=float(np.sqrt(np.sum(residual**2)/max(len(x)-2,1)/np.sum((x-x.mean())**2)))
        r2=float(1-np.sum(residual**2)/max(float(np.sum((v-v.mean())**2)),1e-30))
        fits.append({'sigma':float(slope),'stderr':error,'R2':r2})
    a,b=fits;agreement=abs(a['sigma']-b['sigma'])<=max(.05*abs(b['sigma']),2e-5)
    if agreement and abs(b['sigma'])<5e-5 and b['stderr']<1e-5:status='NEAR_ZERO'
    elif agreement and min(a['R2'],b['R2'])>=.99 and abs(b['sigma'])>max(2*b['stderr'],2e-5):
        status='UNSTABLE' if b['sigma']>0 else 'STABLE'
    else:status='UNRESOLVED'
    return {'status':status,'sigma':b['sigma'],'stderr':b['stderr'],'fits':fits,'window_ps':[float(t[-1]-2*window),float(t[-1])]}


def linear_threshold_batch(points,*,N=128,L=128.,dt=.05,precision='complex64',seed=20261008,max_time=2400.):
    """Batched Arc linear propagation. Rescale every 1 ps, fit amplitude growth.

    B0/A1 are normalized independently per circular component, so the slower
    component's threshold is retained. T0 is a coupled, jointly normalized spinor.
    Reliable points leave the active batch; unresolved signs are never filled in.
    """
    import torch
    if not torch.xpu.is_available():raise RuntimeError('Intel Arc unavailable')
    configs=[report3_config(case,geom,power,space={'N':N,'L':L},time={'dt':dt}) for geom,case,power in points]
    if any(parameters(c)['J']!=0 for c in configs):raise ValueError('This threshold stage requires J=0')
    payload={'points':points,'N':N,'L':L,'dt':dt,'precision':precision,'seed':seed,'max_time':max_time,
             'configs':configs,'algorithm':'frozen-spin-strang-v1','torch':torch.__version__,'numpy':np.__version__}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:12]
    folder=ROOT/'results/runs/report3/linear';folder.mkdir(parents=True,exist_ok=True)
    result_file=folder/(digest+'.json');state_file=folder/(digest+'.npz')
    if result_file.exists():
        cached=json.loads(result_file.read_text())
        if cached['status']=='COMPLETED':return cached['points']
    dtype=torch.complex64 if precision=='complex64' else torch.complex128
    real=torch.float32 if dtype==torch.complex64 else torch.float64
    grid=make_grid(configs[0]);initial=[];local=[];half=[[],[],[]];full=[[],[],[]];joint=[]
    for c in configs:
        p=parameters(c);a,b=frozen_spin_reservoirs(c,grid)
        local.append(np.exp(((p['R']*a-p['gammaC'])/2-1j*(p['gA']*a+p['gI']*b)/p['hbar'])*dt))
        wave=_threshold_seed(c,grid,seed);initial.append(np.stack((wave,wave)))
        joint.append(p['beta']!=0)
        for tau,target in ((dt/2,half),(dt,full)):
            factor=kinetic_factor(grid,p,tau)
            if not isinstance(factor,tuple):factor=(factor,np.zeros_like(factor),np.zeros_like(factor))
            for j,value in enumerate(factor):target[j].append(value)
    tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=dtype,device='xpu')
    z=tensor(np.stack(initial,axis=1));local=tensor(np.stack(local,axis=1))
    half=tuple(tensor(v) for v in half);full=tuple(tensor(v) for v in full)
    joint=np.asarray(joint);active=np.arange(len(points));histories=[[[0.,0.,0.]] for _ in points]
    accumulated=np.zeros((len(points),2));records=[None]*len(points);previous=[None]*len(points)
    snapshots=np.zeros((2,len(points),N,N),dtype=np.complex64 if precision=='complex64' else np.complex128)
    def kinetic(field,factors):
        f=torch.fft.fft2(field);a,b,c=factors
        return torch.fft.ifft2(torch.stack((a*f[0]+b*f[1],c*f[0]+a*f[1])))
    def normalize(field):
        mass=(field.real**2+field.imag**2).sum((-2,-1),dtype=torch.float64)
        m=mass.cpu().numpy();norm=np.where(joint[active][None,:],m.sum(0)[None,:],m)
        if not np.isfinite(norm).all() or norm.min()<=0:raise FloatingPointError('Invalid linear norm')
        field=field/torch.as_tensor(np.sqrt(norm),dtype=real,device='xpu')[:,:,None,None]
        return field,.5*np.log(norm).T
    z,_=normalize(z);sample=round(1/dt);check=round(100/dt);steps=round(max_time/dt)
    if abs(sample*dt-1)>1e-9:raise ValueError('Linear dt must divide 1 ps')
    start=time.perf_counter();need_half=True;now=0.;completed=False
    write_json(result_file,{'status':'RUNNING','settings':payload,'points':[]})
    try:
        for step in range(1,steps+1):
            if need_half:z=kinetic(z,half)
            z=z*local;close=step%sample==0
            z=kinetic(z,half if close else full);need_half=close;now=step*dt
            if close:
                z,increment=normalize(z);accumulated[active]+=increment
                for i in active:histories[i].append([now,*accumulated[i]])
            if step%check or now<300:continue
            done=[]
            for j,i in enumerate(active):
                h=np.asarray(histories[i]);channels=1 if joint[i] else 2
                fits=[_growth_fit(h[:,0],h[:,k+1]) for k in range(channels)]
                settled=all(v['status']!='UNRESOLVED' for v in fits)
                old=previous[i]
                consistent=old is not None and all(v['status']==w['status'] and abs(v.get('sigma',0)-w.get('sigma',0))<=max(.02*abs(v.get('sigma',0)),5e-5) for v,w in zip(fits,old))
                previous[i]=fits
                if settled and consistent or step==steps:
                    geom,case,power=points[i]
                    records[i]={'geometry':geom,'case':case,'p':float(power),'channels':fits,'duration_ps':now,
                                'status':'RESOLVED' if settled and consistent else 'UNRESOLVED',
                                'N':N,'L_um':L,'dt_ps':dt,'precision':precision,'seed':seed}
                    snapshots[:,i]=z[:,j].cpu().numpy();done.append(j)
            if done:
                keep=np.setdiff1d(np.arange(len(active)),done)
                active=active[keep];z=z[:,keep];local=local[:,keep]
                half=tuple(v[keep] for v in half);full=tuple(v[keep] for v in full)
            write_json(result_file,{'status':'RUNNING','settings':payload,'time_ps':now,
                       'elapsed_wall_s':time.perf_counter()-start,'active_points':len(active),
                       'points':[v for v in records if v is not None]})
            print(f'Linear {N} {precision}: {now:g} ps, {len(active)} active, {time.perf_counter()-start:.1f} s',flush=True)
            if not len(active):break
        completed=True
    finally:
        # Only completed physical sample times enter reusable growth histories.
        for j,i in enumerate(active):
            h=np.asarray(histories[i]);geom,case,power=points[i]
            records[i]={'geometry':geom,'case':case,'p':float(power),
                        'channels':[_growth_fit(h[:,0],h[:,k+1]) for k in range(1 if joint[i] else 2)],
                        'duration_ps':float(h[-1,0]),'status':'UNRESOLVED','N':N,'L_um':L,'dt_ps':dt,'precision':precision,'seed':seed}
            snapshots[:,i]=z[:,j].cpu().numpy()
        np.savez_compressed(state_file,psi=snapshots,settings_json=json.dumps(payload),
                            histories_json=json.dumps(histories),x=grid.x,y=grid.y)
        write_json(result_file,{'status':'COMPLETED' if completed else 'RUN_INCOMPLETE','settings':payload,'elapsed_wall_s':time.perf_counter()-start,'points':records})
    return records


def _threshold_bounds(records,geometry,case,channel):
    values=[]
    for r in records:
        if r['geometry']!=geometry or r['case']!=case or r['status']!='RESOLVED':continue
        fit=r['channels'][channel]
        if fit['status'] in ('STABLE','UNSTABLE'):values.append((r['p'],fit['status']))
    positive=sorted(p for p,s in values if s=='UNSTABLE')
    if not positive:return None
    hi=positive[0];negative=[p for p,s in values if s=='STABLE' and p<hi]
    if not negative:return None
    lo=max(negative)
    return {'lo':lo,'hi':hi,'relative_width':(hi-lo)/((hi+lo)/2)}


def search_report3_thresholds(*,N=128,precision='complex64',seed=20261008,initial=None,max_rounds=7):
    records=[];attempted=set();out=ROOT/'results/data/report3';out.mkdir(parents=True,exist_ok=True)
    stage='coarse' if N==128 else 'refined'
    pending=initial or [(g,c,p) for g,powers in (('spot',(10.,11.,12.,14.)),('ring',(2.3,2.6,2.9,3.3))) for c in ('B0','A1','T0') for p in powers]
    for round_index in range(max_rounds):
        pending=sorted({(g,c,round(float(p),8)) for g,c,p in pending}-attempted)
        if pending:
            write_json(out/(stage+'_thresholds.json'),{'status':'RUNNING','round':round_index,'pending':pending,'points':records})
            batch=linear_threshold_batch(pending,N=N,precision=precision,seed=seed)
            records.extend(batch);attempted.update(pending)
        bounds={};next_points=[];complete=True
        for geometry in ('spot','ring'):
            for case in ('B0','A1','T0'):
                for channel in range(1 if case=='T0' else 2):
                    key=f'{geometry}_{case}_'+('joint' if case=='T0' else ('plus','minus')[channel])
                    bracket=_threshold_bounds(records,geometry,case,channel)
                    bounds[key]=bracket
                    if bracket is not None and bracket['relative_width']<=.05:continue
                    complete=False
                    if bracket is not None:
                        mid=(bracket['lo']+bracket['hi'])/2
                        if (geometry,case,round(mid,8)) in attempted:
                            next_points.extend([(geometry,case,mid*.985),(geometry,case,mid*1.015)])
                        else:next_points.append((geometry,case,mid))
                    else:
                        done=[r for r in records if r['geometry']==geometry and r['case']==case]
                        powers=[r['p'] for r in done]
                        has_positive=any(r['status']=='RESOLVED' and r['channels'][channel]['status']=='UNSTABLE' for r in done)
                        next_points.append((geometry,case,min(powers)*.85 if has_positive else max(powers)*1.15))
        result={'status':'BOUNDED' if complete else 'REFINING','N':N,'L_um':128.,'precision':precision,
                'dt_ps':.05,'seed':seed,'target_relative_width':.05,'bounds':bounds,'points':records}
        write_json(out/(stage+'_thresholds.json'),result)
        if complete:return result
        pending=next_points
    result['status']='UNRESOLVED';write_json(out/(stage+'_thresholds.json'),result)
    return result


def run_report3_threshold_stage():
    """Queued stage: coarse thresholds -> double-precision checks -> short starts.

    Uses a separate 1.2*max(threshold upper bounds) pump for each geometry;
    B0/A1/T0 share that pump within a geometry. Stops before long production.
    """
    out=ROOT/'results/data/report3';out.mkdir(parents=True,exist_ok=True)
    manifest=out/'threshold_selection.json'
    report={'status':'RUNNING','stage':'coarse thresholds','pump_rule':'1.2 times largest verified upper bound within each geometry',
            'long_production_started':False,'stages':{}}
    write_json(manifest,report);start=time.perf_counter()
    coarse=search_report3_thresholds();report['stages']['coarse']={'elapsed_s':time.perf_counter()-start,'status':coarse['status']}
    if coarse['status']!='BOUNDED':
        report.update(status='NEEDS_REVIEW',reason='Coarse threshold signs unresolved');write_json(manifest,report);return report
    report['stage']='refined thresholds';write_json(manifest,report)
    endpoints=[]
    for key,b in coarse['bounds'].items():
        geom,case,_=key.split('_');endpoints.extend([(geom,case,b['lo']),(geom,case,b['hi'])])
    start=time.perf_counter();fine=search_report3_thresholds(N=256,precision='complex128',seed=20261009,initial=endpoints)
    report['stages']['refined']={'elapsed_s':time.perf_counter()-start,'status':fine['status']}
    if fine['status']!='BOUNDED':
        report.update(status='NEEDS_REVIEW',reason='Fine threshold signs unresolved');write_json(manifest,report);return report
    powers={g:1.2*max(b['hi'] for key,b in fine['bounds'].items() if key.startswith(g+'_')) for g in ('spot','ring')}
    report['selected_p']=powers;report['stage']='working-point step check';write_json(manifest,report)
    start=time.perf_counter()
    checks=linear_threshold_batch([(g,c,powers[g]) for g in ('spot','ring') for c in ('B0','A1','T0')],
                                 N=256,dt=.025,precision='complex128',seed=20261009)
    supported=all(r['status']=='RESOLVED' and all(v['status']=='UNSTABLE' for v in r['channels']) for r in checks)
    report['stages']['working_point']={'elapsed_s':time.perf_counter()-start,'points':checks,'status':'PASS' if supported else 'NEEDS_REVIEW'}
    if not supported:
        report.update(status='NEEDS_REVIEW',reason='Working point not independently confirmed above threshold');write_json(manifest,report);return report
    report['stage']='short nonlinear startups';write_json(manifest,report);start=time.perf_counter();startups={}
    for geometry in ('spot','ring'):
        for case in ('B0','A1','T0'):
            name=f'near_threshold_{case}_{geometry}_startup'
            config=report3_config(case,geometry,powers[geometry],space={'N':256,'L':128.},time={'final':400.})
            while True:
                run=run_report3_pilot(config,name);row=run['summary']
                if row['run_status']!='RUN_COMPLETED':
                    report.update(status='RUN_INCOMPLETE');write_json(manifest,report);return report
                with np.load(run['path']/'checkpoint.npz') as z:h=z['history'].copy()
                last=h[h[:,0]>=h[-1,0]-100];nc=last[:,1:3].sum(1)
                slope=float(np.polyfit(last[:,0],np.log(np.maximum(nc,1e-300)),1)[0])
                acceptable=row['late_NC_to_seed']>=100 and slope>=-1e-4 and row['total_balance_relative']<1e-3 and row['condensate_balance_relative']<1e-3
                startups[name]={'status':'SUPPORTED' if acceptable else 'UNRESOLVED','time_ps':row['time_ps'],
                                'NC':row['NC'],'log_NC_slope':slope,'summary':row}
                report['startups']=startups;write_json(manifest,report)
                if acceptable or config['time']['final']>=1200:break
                config['time']['final']+=400
    report['stages']['startup']={'elapsed_s':time.perf_counter()-start}
    report['status']='READY_FOR_LONG_RUNS' if all(v['status']=='SUPPORTED' for v in startups.values()) else 'NEEDS_REVIEW'
    report['stage']='finished; review before 512^2 production'
    write_json(manifest,report)
    return report


def run_report3_night(hours=9.5):
    """One finite overnight queue; preserve unfinished jobs for exact continuation."""
    import fcntl
    from datetime import datetime,timezone
    root=ROOT/'results/runs/report3';root.mkdir(parents=True,exist_ok=True)
    out=ROOT/'results/data/report3';out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+hours*3600
    cases=[(case,g) for case in ('B0','A1','T0','C1','A0','T1','J1') for g in ('spot','ring')]
    report={'status':'WAITING_FOR_THRESHOLD_STAGE','budget_hours':hours,
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'planned':[{'case':c,'geometry':g,'target_ps':3000} for c,g in cases],
            'precision':'complex128/float64','results':{},'errors':{},'thermal_stop':False,
            'policy':'3000 ps per case first; unresolved stationarity retained for morning review'}
    manifest=out/'night_queue.json';write_json(manifest,report)
    with (root/'pilots.lock').open('a') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:
                    report['status']='TIME_BUDGET_REACHED';write_json(manifest,report);return report
                time.sleep(5)
        selection=json.loads((out/'threshold_selection.json').read_text())
        if selection['status']!='READY_FOR_LONG_RUNS':
            report.update(status='NEEDS_REVIEW',reason='Threshold/startup stage did not pass',threshold_status=selection['status'])
            write_json(manifest,report);return report
        report.update(status='RUNNING',selected_p=selection['selected_p']);write_json(manifest,report)
        # Chapter 1: cheap, independent homogeneous calculations, same switch protocol.
        from model import load_config
        scalar={}
        for label,model in (('one','M0'),('two','M1')):
            c=load_config('F1',model=model,time={'final':1600.,'sample':.1})
            data=solve_0d(c);obs=observables_0d(data)
            for key in ('time','psi','active','inactive'):scalar[label+'_'+key]=data[key]
            scalar[label+'_feed']=obs['feed'];scalar[label+'_config_json']=json.dumps(c)
            report.setdefault('scalar_checks',{})[label]=balance_0d(data,obs)
        np.savez_compressed(out/'reservoir_switching.npz',**scalar);write_json(manifest,report)
        for case,geometry in cases:
            name=f'production_{case}_{geometry}'
            if time.monotonic()>=deadline:
                report['status']='TIME_BUDGET_REACHED';break
            report['active_case']=name;write_json(manifest,report)
            config=report3_config(case,geometry,report['selected_p'][geometry],time={'final':3000.})
            try:
                run=run_report3_pilot(config,name,deadline=deadline)
                row=run['summary'];report['results'][name]=row
                write_json(manifest,report)
                print('NIGHT',name,row['run_status'],row['time_ps'],row['numerical_checks'],flush=True)
                if row['run_status']!='RUN_COMPLETED':
                    report['status']='TIME_BUDGET_REACHED' if time.monotonic()>=deadline else 'RUN_INCOMPLETE'
                    break
                if max(row['total_balance_relative'],row['condensate_balance_relative'])>=1e-3:
                    report.update(status='NEEDS_REVIEW',reason='Particle balance failed');break
            except KeyboardInterrupt:
                report['status']='RUN_INCOMPLETE';break
            except Exception as error:
                report['errors'][name]={'type':type(error).__name__,'message':str(error)}
                report['status']='NEEDS_REVIEW';write_json(manifest,report);break
        else:report['status']='COMPLETED_MAIN_MATRIX'
        report['finished_utc']=datetime.now(timezone.utc).isoformat();report.pop('active_case',None)
        write_json(manifest,report)
    return report


# Precision audit: empirical sensitivity estimates, not certified error bounds.
def _precision_state_error(state,reference):
    """One common field phase; reservoirs scaled separately, never component phases."""
    field=aligned_field_error(state[0],reference[0])[0]
    return float(np.sqrt((field**2+sum(relative_norm(a,b)**2 for a,b in zip(state[1:],reference[1:])))/3))


def _precision_metrics(state,reference,grid):
    state=tuple(np.asarray(v,dtype=np.complex128 if i==0 else np.float64) for i,v in enumerate(state))
    reference=tuple(np.asarray(v,dtype=np.complex128 if i==0 else np.float64) for i,v in enumerate(reference))
    maps,S0=_report3_maps(reference);current,_=_report3_maps(state)
    errors=_report3_map_errors(current,maps,S0)
    n=abs(state[0])**2;nr=abs(reference[0])**2
    populations=n.sum((-2,-1),dtype=np.float64)*grid.da
    refpop=nr.sum((-2,-1),dtype=np.float64)*grid.da
    floor=max(float(refpop.sum())*1e-6,1e-30)
    S=stokes(state[0]);Sr=stokes(reference[0]);mask=Sr[0]>1e-3*Sr[0].max()
    weight=Sr[0][mask];ss=S[1:,mask]/np.maximum(S[0,mask],1e-30)
    sr=Sr[1:,mask]/np.maximum(Sr[0,mask],1e-30)
    stokes_rms=np.sqrt(((ss-sr)**2*weight).sum(1)/max(weight.sum(),1e-30))
    return {'population_relative':abs(populations-refpop)/np.maximum(refpop,floor),
            'density_relative_L2':errors[:2],'reservoir_relative_L2':errors[2:6],
            'unnormalized_stokes_relative_L2':errors[6:],'normalized_stokes_weighted_RMS':stokes_rms,
            'common_phase_aligned_field_relative_L2':aligned_field_error(state[0],reference[0])[0],
            'raw_field_relative_L2':relative_norm(state[0],reference[0]),
            'scaled_state_error':_precision_state_error(state,reference)}


def _precision_advance(backend,state,duration):
    """Same observable closing cadence as production; no reference resets inside a block."""
    steps=round(duration/backend.dt);close_every=round(.25/backend.dt)
    if not np.isclose(steps*backend.dt,duration):raise ValueError('Audit duration must align with dt')
    backend.staggered=False;state=backend.to_device(state)
    for step in range(1,steps+1):state,_,_=backend.step(state,close=step%close_every==0 or step==steps)
    return backend.to_numpy(state)


def _precision_perturb(state,grid,epsilon,seed):
    """Smooth deterministic directions in real/imag fields and both reservoirs."""
    rng=np.random.default_rng(seed);xx=grid.x[None,:];yy=grid.y[:,None]
    waves=[]
    for j in range(8):
        k=rng.integers(-6,7,size=(3,2));ph=rng.uniform(-np.pi,np.pi,3)
        v=sum(np.cos(2*np.pi*(kx*xx+ky*yy)/(len(grid.x)*grid.dx)+phase) for (kx,ky),phase in zip(k,ph))
        waves.append(v/np.sqrt(np.mean(v*v)))
    z,a,b=state
    noise=np.stack([waves[0]+1j*waves[1],waves[2]+1j*waves[3]])
    # Local envelope avoids injecting a large population into empty far-field cells.
    dz=noise*np.sqrt(np.maximum(abs(z)**2,1e-3*np.mean(abs(z)**2,axis=(-2,-1),keepdims=True)))/np.sqrt(2)
    return z+epsilon*dz,a*(1+epsilon*np.stack(waves[4:6])),b*(1+epsilon*np.stack(waves[6:8]))


def _precision_reference(name):
    path=ROOT/'results/runs/report3'/name
    config=yaml.safe_load((path/'resolved_config.yaml').read_text())
    with np.load(path/'checkpoint.npz') as z:times=z['probe_times'].copy()
    store=_ProbeStore(path/'probe_maps.npy',config['space']['N'],times=times)
    return config,store


def audit_report3_precision(block_ps=10.,samples=3):
    """Short float32 shadows plus directional amplification against saved float64 runs."""
    out=ROOT/'results/data/report3';manifest=out/'precision_audit.json'
    source=hashlib.sha256((ROOT/'numerics.py').read_bytes()).hexdigest()
    report=json.loads(manifest.read_text()) if manifest.exists() else {}
    if report and (report.get('source_sha256')!=source or report.get('block_ps')!=block_ps or report.get('samples')!=samples):
        raise ValueError('Existing precision audit has different code/settings; preserve it before rerunning')
    if not report:
        report={'status':'RUNNING','source_sha256':source,'block_ps':block_ps,'samples':samples,
                'epsilon':1e-5,'results':{},'interpretation':'empirical directional sensitivity; NOT a rigorous upper error bound',
                'caveat':'saved rolling history samples late dynamics, not the entire transient',
                'policy':'all original double trajectories and checkpoints remain unchanged'}
    for case in ('B0','A0','A1','T0','T1','J1','C1'):
        for geometry in ('spot','ring'):
            name=f'production_{case}_{geometry}'
            if name in report['results']:continue
            report['active_case']=name;write_json(manifest,report)
            config,store=_precision_reference(name);g=make_grid(config)
            valid=store.ordered();eligible=[i for i in valid if np.any(np.isclose(store.times,store.times[i]+block_ps,atol=1e-8,rtol=0))]
            chosen=[eligible[i] for i in np.unique(np.rint(np.linspace(0,len(eligible)-1,samples)).astype(int))]
            c64=copy.deepcopy(config);c64['execution']['precision']='complex64'
            double=TorchSpinStepper(config,g,compiled=True);single=TorchSpinStepper(c64,g,compiled=True)
            rows=[];started=time.perf_counter()
            for number,i in enumerate(chosen):
                t0=float(store.times[i]);j=int(np.flatnonzero(np.isclose(store.times,t0+block_ps,atol=1e-8,rtol=0))[0])
                state=store.state(i);reference=store.state(j)
                baseline=_precision_advance(double,state,block_ps)
                single_end=_precision_advance(single,state,block_ps)
                seed=20261025+number
                perturbed=_precision_perturb(state,g,report['epsilon'],seed)
                perturb_end=_precision_advance(double,perturbed,block_ps)
                input_error=_precision_state_error(perturbed,state)
                gain=_precision_state_error(perturb_end,baseline)/max(input_error,1e-30)
                metrics=_precision_metrics(single_end,baseline,g)
                # No exponential extrapolation: finite directions do not bound all errors.
                score=max(float(np.max(metrics['population_relative'])),float(np.max(metrics['density_relative_L2'])),
                          float(np.max(metrics['normalized_stokes_weighted_RMS'])))*max(1.,gain)
                row={'interval_ps':[t0,t0+block_ps],'single_vs_double':metrics,
                     'single_vs_saved':_precision_metrics(single_end,reference,g),
                     'double_restart_vs_saved':_precision_metrics(baseline,reference,g),
                     'input_perturbation_norm':input_error,'directional_amplification':gain,'risk_score':score,'perturbation_seed':seed}
                rows.append(row)
                report['partial']={'name':name,'blocks':rows};write_json(manifest,report)
                print('PRECISION BLOCK',name,t0,'score',score,'gain',gain,flush=True)
            report['results'][name]={'case':case,'geometry':geometry,'blocks':rows,'risk_score':max(v['risk_score'] for v in rows),
                'elapsed_s':time.perf_counter()-started}
            report.pop('partial',None);write_json(manifest,report)
            del double,single;import torch;torch.xpu.empty_cache()
    # Distinct dynamics: a transitioning ring and a periodic spot. No single universal worst case.
    ring=[(k,v) for k,v in report['results'].items() if v['geometry']=='ring' and v['case'] in ('B0','A0','A1','T1','J1')]
    spot=[(k,v) for k,v in report['results'].items() if v['geometry']=='spot' and v['case'] in ('A0','A1','T1','J1','C1')]
    selected=[max(group,key=lambda item:item[1]['risk_score'])[0] for group in (ring,spot)]
    report['selected_full_repeats']=selected
    for name in selected:
        row=report['results'][name]
        if 'half_epsilon_check' in row:continue
        config,store=_precision_reference(name);g=make_grid(config);block=max(row['blocks'],key=lambda v:v['risk_score'])
        state=store.state(int(np.flatnonzero(np.isclose(store.times,block['interval_ps'][0],atol=1e-8,rtol=0))[0]))
        backend=TorchSpinStepper(config,g,compiled=True)
        baseline=_precision_advance(backend,state,block_ps)
        half=_precision_perturb(state,g,report['epsilon']/2,block['perturbation_seed'])
        end=_precision_advance(backend,half,block_ps)
        gain=_precision_state_error(end,baseline)/max(_precision_state_error(half,state),1e-30)
        discrepancy=abs(gain/block['directional_amplification']-1)
        row['half_epsilon_check']={'amplification':gain,'relative_change':discrepancy,
                                  'status':'PASS' if discrepancy<.05 else 'WARN'}
        write_json(manifest,report);del backend
    report.update(status='SHORT_AUDIT_COMPLETED');report.pop('active_case',None);write_json(manifest,report)
    return report


def compare_report3_precision(reference_name,single_name):
    """Compare full independent trajectories on a shared late interval, plus cycle lag."""
    c,ref=_precision_reference(reference_name);cs,single=_precision_reference(single_name)
    for key in ('physical','space','time','pump','initial'):
        if c[key]!=cs[key]:raise ValueError('Precision comparison changed physics/grid/time/seed')
    data=ROOT/'results/data/report3'
    with np.load(data/(reference_name+'.npz')) as z:hr=z['history'].copy();fr=z['features'].copy()
    with np.load(data/(single_name+'.npz')) as z:hs=z['history'].copy();fs=z['features'].copy()
    end=min(hr[-1,0],hs[-1,0]);g=make_grid(c)
    # Reanalyze both with the corrected classifier and identical available window.
    ar=analyze_report3(ref,hr,fr,end,c);ass=analyze_report3(single,hs,fs,end,cs)
    window=max(ar['window_length_ps'],ass['window_length_ps'])
    left=end-2*window;available=max(float(ref.times[ref.ordered()[0]]),float(single.times[single.ordered()[0]]))
    enough=available<=left+1e-8
    left=max(left,available)
    period_r=ar.get('period_ps');period_s=ass.get('period_ps')
    shift=0.
    if period_r and period_s:
        # Fit a SINGLE common cycle shift to populations and integrated polarization.
        use=(fs[:,0]>=left+period_r)&(fs[:,0]<=end-period_r)
        if use.sum()>20:
            tt=fs[use,0];values=fs[use,1:10];scale=np.maximum(np.mean(abs(fr[:,1:10]),axis=0),1.)
            def cost(lag):
                target=np.column_stack([np.interp(tt+lag,fr[:,0],fr[:,j]) for j in range(1,10)])
                return float(np.mean(((values-target)/scale)**2))
            trials=np.linspace(-period_r/2,period_r/2,81);j=int(np.argmin([cost(v) for v in trials]))
            fit=minimize_scalar(cost,bounds=(trials[max(0,j-1)],trials[min(len(trials)-1,j+1)]),method='bounded')
            shift=float(fit.x)
    rows=[];maximum=np.zeros(9);aligned_maximum=np.zeros(9);srms=np.zeros(3);aligned_srms=np.zeros(3)
    rt=ref.times[ref.ordered()];ri=ref.ordered()
    indices=single.ordered(left,end)
    # 21 distributed full-map comparisons; populations retain dense 0.25 ps histories.
    indices=indices[np.unique(np.rint(np.linspace(0,len(indices)-1,min(21,len(indices)))).astype(int))]
    for i in indices:
        now=float(single.times[i]);j=ref.ordered(now,now)
        if not len(j):continue
        state=single.state(i);reference=ref.state(j[0]);raw=_precision_metrics(state,reference,g)
        maximum=np.maximum(maximum,np.r_[raw['density_relative_L2'],raw['reservoir_relative_L2'],raw['unnormalized_stokes_relative_L2']])
        srms=np.maximum(srms,raw['normalized_stokes_weighted_RMS'])
        target=now+shift
        if target<rt[0] or target>rt[-1]:continue
        ix,w=_lag_polynomial_weights(rt,target,8)
        maps=np.zeros((9,len(g.x),len(g.x)));S0=np.zeros((len(g.x),len(g.x)))
        # Interpolate physical maps, never independently align component phases.
        for k,weight in zip(ix,w):
            m,s0=_report3_maps(ref.state(ri[k]));maps+=weight*m;S0+=weight*s0
        current,sc0=_report3_maps(state);errors=_report3_map_errors(current,maps,S0)
        aligned_maximum=np.maximum(aligned_maximum,errors)
        mask=S0>1e-3*S0.max();weights=S0[mask]
        pol=current[6:,mask]/np.maximum(sc0[mask],1e-30)-maps[6:,mask]/np.maximum(S0[mask],1e-30)
        rms=np.sqrt((pol**2*weights).sum(1)/max(weights.sum(),1e-30));aligned_srms=np.maximum(aligned_srms,rms)
        rows.append({'time_ps':now,'raw':raw,'cycle_aligned_map_errors':errors,'cycle_aligned_stokes_RMS':rms})
    use=(hs[:,0]>=left)&(hs[:,0]<=end);tt=hs[use,0]
    target=np.column_stack([np.interp(tt,hr[:,0],hr[:,j]) for j in range(1,7)])
    poperr=np.max(abs(hs[use,1:7]-target)/np.maximum(np.mean(abs(target),axis=0),1e-12),axis=0)
    aligned_use=(tt+shift>=hr[0,0])&(tt+shift<=hr[-1,0]);ta=tt[aligned_use]
    aligned_target=np.column_stack([np.interp(ta+shift,hr[:,0],hr[:,j]) for j in range(1,7)])
    aligned_poperr=np.max(abs(hs[use,1:7][aligned_use]-aligned_target)/np.maximum(np.mean(abs(aligned_target),axis=0),1e-12),axis=0)
    period_error=abs(period_s/period_r-1) if period_r and period_s else None
    rf=fr[(fr[:,0]>=left)&(fr[:,0]<=end),1:10];sf=fs[(fs[:,0]>=left)&(fs[:,0]<=end),1:10]
    amplitude_r=np.ptp(rf,axis=0);amplitude_s=np.ptp(sf,axis=0)
    meaningful=amplitude_r>max(1e-6*np.max(np.mean(abs(rf),axis=0)),1e-3*amplitude_r.max())
    amplitude_error=float(np.max(abs(amplitude_s[meaningful]-amplitude_r[meaningful])/amplitude_r[meaningful])) if meaningful.any() else 0.
    arithmetic_pass=bool(enough and aligned_poperr.max()<.001 and aligned_maximum[:6].max()<.001 and aligned_srms.max()<.001 and
                         (period_error is None or (period_error<.001 and amplitude_error<.001)))
    passed=arithmetic_pass and ar['status']==ass['status']
    # A finite-time precision comparison can pass even while physical settling is unresolved.
    return {'reference':reference_name,'single':single_name,'status':'PASS' if passed else 'NEEDS_REVIEW',
            'interval_ps':[left,end],'enough_shared_history':enough,'analysis_reference':ar,'analysis_single':ass,
            'single_common_cycle_shift_ps':shift,'period_relative_error':period_error,
            'feature_amplitude_relative_error':amplitude_error,'arithmetic_observables_pass':arithmetic_pass,
            'temporal_classification_matches':ar['status']==ass['status'],
            'population_max_relative_error_same_time':poperr,'population_max_relative_error_cycle_aligned':aligned_poperr,
            'map_max_relative_errors_same_time':maximum,'map_max_relative_errors_cycle_aligned':aligned_maximum,
            'stokes_max_weighted_RMS_same_time':srms,'stokes_max_weighted_RMS_cycle_aligned':aligned_srms,
            'map_samples':rows,'caveat':'UNRESOLVED is temporal classification, not automatic arithmetic failure; no universal precision certificate'}


def run_report3_precision_queue():
    """Finite queue: short audits, then two full independently seeded-as-reference repeats."""
    import fcntl
    from datetime import datetime,timezone
    root=ROOT/'results/runs/report3';out=ROOT/'results/data/report3';manifest=out/'precision_queue.json'
    with (root/'pilots.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        report={'status':'SHORT_AUDITS','started_utc':datetime.now(timezone.utc).isoformat(),'results':{},
                'policy':'no thermal stops; no automatic precision promotion or physical-parameter changes'}
        write_json(manifest,report)
        try:
            audit=audit_report3_precision();report['selected']=audit['selected_full_repeats']
            for reference in report['selected']:
                name=reference+'_c64';config,_=_precision_reference(reference)
                config['execution']['precision']='complex64'
                report.update(status='FULL_REPEAT',active_case=name);write_json(manifest,report)
                result=run_report3_pilot(config,name)
                if result['summary']['run_status']!='RUN_COMPLETED':
                    report['status']='RUN_INCOMPLETE';write_json(manifest,report);return report
                report['results'][reference]=compare_report3_precision(reference,name)
                write_json(manifest,report)
            report.update(status='COMPLETED',finished_utc=datetime.now(timezone.utc).isoformat())
            report.pop('active_case',None);write_json(manifest,report)
            return report
        except BaseException as error:
            report.update(status='RUN_INCOMPLETE' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
            write_json(manifest,report);raise


def run_report3_boundary_queue(hours=1.5):
    """Matched domain controls with a physical seed overriding machine symmetry noise."""
    import fcntl
    from datetime import datetime,timezone
    root=ROOT/'results/runs/report3';out=ROOT/'results/data/report3'
    manifest=out/'boundary_queue.json'
    plans=[('boundary_J1_ring_seeded_L256',512,256.),('boundary_J1_ring_seeded_L384',768,384.)]
    report={'status':'WAITING_FOR_GPU','started_utc':datetime.now(timezone.utc).isoformat(),
            'budget_hours':hours,'planned':[{'name':n,'N':N,'L_um':L,'target_ps':2000.} for n,N,L in plans],
            'precision':'complex128','dt_ps':.025,'dx_um':.5,'results':{},
            'initial':'same continuous real 0.1% modulation of radial field, angular harmonics 1..8, seed 20261025',
            'caveat':'tests domain influence for controlled preparation; does not identify the unique charge of the old unperturbed preparation',
            'thermal_stop':False}
    write_json(manifest,report)
    with (root/'pilots.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);deadline=time.monotonic()+hours*3600
        try:
            configs=[report3_config('J1','ring',3.3,space={'N':N,'L':L},time={'final':2000.},
                      initial={'kind':'radial_perturbed','perturbation':1e-3,'perturb_seed':20261025},
                      execution={'precision':'complex128'}) for _,N,L in plans]
            small=initial_state(configs[0],make_grid(configs[0]))[0]
            large=initial_state(configs[1],make_grid(configs[1]))[0]
            offset=(plans[1][1]-plans[0][1])//2;crop=large[:,offset:offset+plans[0][1],offset:offset+plans[0][1]]
            error=relative_norm(crop,small)
            if error>1e-15:raise ValueError('Physical initial seeds differ between domains')
            report['initial_overlap_relative_error']=error
            for (name,_,_),config in zip(plans,configs):
                report.update(status='RUNNING',active_case=name);write_json(manifest,report)
                result=run_report3_pilot(config,name,common_window=1000.,deadline=deadline)
                report['results'][name]=result['summary'];write_json(manifest,report)
                if result['summary']['run_status']!='RUN_COMPLETED':
                    report['status']='RUN_INCOMPLETE';write_json(manifest,report);return report
            name_a,name_b=plans[0][0],plans[1][0]
            with np.load(out/(name_a+'.npz')) as z:a=tuple(z[k] for k in ('psi','active','inactive'))
            with np.load(out/(name_b+'.npz')) as z:b=tuple(z[k] for k in ('psi','active','inactive'))
            common=tuple(v[:,offset:offset+plans[0][1],offset:offset+plans[0][1]] for v in b)
            grid=make_grid(configs[0]);report['domain_comparison_common_L256']=_precision_metrics(common,a,grid)
            mask=np.hypot(grid.x[:,None],grid.x[None,:])<=40.
            report['density_relative_L2_r40']=[relative_norm(abs(common[0][j][mask])**2,abs(a[0][j][mask])**2) for j in (0,1)]
            for name,state,config in zip((name_a,name_b),(a,b),configs):
                g=make_grid(config);p=parameters(config)
                report['results'][name]['topology']=[vortex_diagnostics(state[0][j],g,p,radii=(4.,6.,8.,10.,12.)) for j in (0,1)]
            report.update(status='COMPLETED',finished_utc=datetime.now(timezone.utc).isoformat())
            report.pop('active_case',None);write_json(manifest,report);return report
        except BaseException as error:
            report.update(status='RUN_INCOMPLETE' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
            write_json(manifest,report);raise
