"""Compact physical and numerical checks from section 12 of the specification."""
import copy
import numpy as np
from numba import set_num_threads
set_num_threads(1)
from model import (load_config,parameters,make_grid,injection,initial_state,seed_waves,
                   local_rhs,spin_stationary)
from numerics import (local_rk4,solve_0d,split_step,pack,unpack,kinetic_step,
                      _scalar_local_inplace,scalar_coefficients,masked_unwrap,
                      observables_0d,classify_spin,run_2d)


def test_parameter_matching_and_analytic_spin_branch():
    for variant,tau in [('R1',9.628438728117185),('R2',19.29824561403509)]:
        p=parameters(load_config('F1',reservoir=variant))
        assert abs(p['tauI']-tau)<1e-12
        assert abs(p['gI']/p['W']-p['gR']*p['eta'])<1e-14
    p=parameters(load_config('F3'));n,a,i,e=spin_stationary(2,.3,p)
    assert abs(n.sum()-4.761904761904762)<1e-12
    assert abs((n[0]-n[1])/n.sum()-.6835999511539871*.3)<1e-12
    assert abs(-(e[0]-e[1])/p['hbar']+.016619037553377278)<1e-12


def test_spin_flip_has_factor_two_and_conserves_sum():
    c=load_config('F3');p=parameters(c);p.update(R=0.,W=0.,gammaA=0.,gammaI=0.)
    state=(np.zeros(2,complex),np.array([2.,1.]),np.array([3.,1.]))
    dt=.0125
    for _ in range(80):state=local_rk4(state,np.zeros(2),p,dt)
    np.testing.assert_allclose(state[1].sum(),3.,atol=1e-14)
    np.testing.assert_allclose(state[2].sum(),4.,atol=1e-14)
    np.testing.assert_allclose(state[1][0]-state[1][1],np.exp(-.2),atol=6e-14)


def test_scalar_limit_and_matched_stationary_reservoir():
    c=load_config('F3');p=parameters(c);p.update(GammaA=0.,GammaI=0.)
    scalar=parameters(load_config('F1'));pump=np.array([.65])
    a=(np.array([.1+.03j]),np.array([2.]),np.array([4.]))
    b=tuple(np.r_[v,np.zeros(1)] for v in a)
    for _ in range(100):
        a=local_rk4(a,pump,scalar,.025);b=local_rk4(b,np.r_[pump,0.],p,.025)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y[:1],atol=1e-14,rtol=0)
    old=parameters(load_config('F1',model='M0'))
    state=(np.array([.1+.03j]),np.array([2.]),pump/scalar['W'])
    for x,y in zip(local_rhs(state,pump,old)[:2],local_rhs(state,pump,scalar)[:2]):
        np.testing.assert_allclose(x,y,atol=1e-15)


def test_rabi_solution_sign_period_and_fourth_order():
    c=load_config('F4');p=parameters(c)
    for key in ('gC','gR','gA','gI','alpha1','alpha2','gammaC'):p[key]=0.
    initial=(np.array([1.+0j,0j]),np.zeros(2),np.zeros(2))
    errors=[]
    for dt in (.5,.25):
        state=tuple(x.copy() for x in initial)
        for _ in range(round(100/dt)):state=local_rk4(state,np.zeros(2),p,dt)
        expected=np.array([np.cos(p['J']*100/p['hbar']),1j*np.sin(p['J']*100/p['hbar'])])
        errors.append(np.max(abs(state[0]-expected)))
    assert errors[0]/errors[1]>14
    assert errors[1]<5e-12


def test_inactive_step_and_pump_switch():
    c=load_config('F1',time={'final':10.,'sample':.1},pump={'off_time':5.})
    data=solve_0d(c);p=parameters(c);t=data['time'];steady=injection(c)[0]/p['W']
    ref=steady*(1-np.exp(-np.minimum(t,5)/p['tauI']))*np.exp(-np.maximum(t-5,0)/p['tauI'])
    np.testing.assert_allclose(data['inactive'][:,0],ref,atol=1e-10,rtol=1e-10)
    fixed=solve_0d(c,method='RK4')
    for key in ('psi','active','inactive'):
        np.testing.assert_allclose(data[key],fixed[key],atol=1e-10,rtol=1e-9)


def test_compiled_local_step_matches_joint_numpy_rk4():
    rng=np.random.default_rng(7)
    for model in ('M0','M1'):
        p=parameters(load_config('F1',model=model));pump=rng.uniform(0,2,(1,8,8))
        state=(rng.normal(size=(1,8,8))+1j*rng.normal(size=(1,8,8)),
               rng.uniform(0,10,(1,8,8)),rng.uniform(0,10,(1,8,8)))
        expected=local_rk4(state,pump,p,.025);actual=tuple(x.copy() for x in state)
        _scalar_local_inplace(*(x.ravel() for x in actual),pump.ravel(),scalar_coefficients(p),.025,model=='M0')
        for x,y in zip(actual,expected):np.testing.assert_allclose(x,y,atol=2e-15,rtol=1e-15)


def test_kinetic_plane_wave_and_parseval():
    c=load_config('demo');g=make_grid(c);p=parameters(c)
    wave=np.exp(1j*(g.kx[3]*g.x[None,:]+g.ky[2]*g.y[:,None]))[None,...]
    factor=np.exp(-1j*p['K']*g.k2*.1/p['hbar'])
    result=kinetic_step(wave,factor)
    expected=wave*np.exp(-1j*p['K']*(g.kx[3]**2+g.ky[2]**2)*.1/p['hbar'])
    np.testing.assert_allclose(result,expected,atol=1e-13)
    spectrum=g.da*np.fft.fft2(wave)
    dk=2*np.pi/c['space']['L']
    np.testing.assert_allclose(np.sum(abs(wave)**2)*g.da,np.sum(abs(spectrum)**2)*dk**2/(2*np.pi)**2,rtol=1e-13)


def test_uniform_spinor_2d_matches_independent_0d():
    c=load_config('demo',time={'final':200.,'sample':.5})
    g=make_grid(c);p=parameters(c);pump=injection(c,g)
    zero=initial_state(c);state=tuple(np.broadcast_to(v[:,None,None],(2,32,32)).copy() for v in zero)
    for _ in range(8000):state=split_step(state,pump,p,.025,g,accelerated=True)
    reference=solve_0d(c)
    for value,key in zip(state,('psi','active','inactive')):
        expected=np.broadcast_to(reference[key][-1,:,None,None],value.shape)
        np.testing.assert_allclose(value,expected,atol=2e-8,rtol=1e-7)


def test_josephson_current_and_reduced_phase_equations():
    c=load_config('F4');p=parameters(c)
    psi=np.array([1.2*np.exp(.3j),.8*np.exp(-.2j)])
    state=(psi,np.array([8.,9.]),np.array([10.,11.]));rhs=local_rhs(state,injection(c),p)
    nc=abs(psi)**2;actual=2*np.real(np.conj(psi)*rhs[0]);IJ=2*p['J']/p['hbar']*np.imag(np.conj(psi[1])*psi[0])
    np.testing.assert_allclose(actual,(p['R']*state[1]-p['gammaC'])*nc+np.array([IJ,-IJ]),atol=1e-15)
    z=(nc[0]-nc[1])/nc.sum();theta=.5
    V=p['alpha1']*nc+p['alpha2']*nc[::-1]+p['gA']*state[1]+p['gI']*state[2]
    reduced=-(V[0]-V[1])/p['hbar']-2*p['J']/p['hbar']*z/np.sqrt(1-z*z)*np.cos(theta)
    np.testing.assert_allclose(np.imag(rhs[0][0]/psi[0]-rhs[0][1]/psi[1]),reduced,atol=1e-15)


def test_continuous_seed_is_identical_on_nested_grids():
    a=load_config('F2',space={'N':64});b=load_config('F2',space={'N':128})
    coarse=initial_state(a,make_grid(a))[0];fine=initial_state(b,make_grid(b))[0]
    np.testing.assert_array_equal(coarse,fine[:,::2,::2])


def test_checkpoint_continuation_matches_uninterrupted(tmp_path,monkeypatch):
    import numerics
    monkeypatch.setattr(numerics,'ROOT',tmp_path)
    c=load_config('demo',model='M1',time={'final':2.})
    run_2d(c,'whole');run_2d(c,'split',max_steps=31);run_2d(c,'split')
    with np.load(tmp_path/'results/runs/whole/checkpoint.npz') as a,np.load(tmp_path/'results/runs/split/checkpoint.npz') as b:
        for key in ('psi','active','inactive'):
            np.testing.assert_array_equal(a[key],b[key])
        for key in ('history','integral','scale'):
            np.testing.assert_allclose(a[key],b[key],rtol=2e-15,atol=1e-12)


def test_phase_unwrap_does_not_bridge_invalid_samples():
    values=masked_unwrap(np.array([3.,-3.,0.,-2.,2.]),np.array([1,1,0,1,1],bool))
    assert np.isnan(values[2])
    assert values[3]==-2.
    np.testing.assert_allclose(values[1],2*np.pi-3)


# A negligible component makes relative phase undefined even when z is measurable.
def test_spin_summary_retains_density_without_inventing_phase():
    from numerics import spin_summary
    t=np.linspace(4000.,6000.,11)
    data={'time':t,'psi':np.tile([1.+0j,1e-4+0j],(len(t),1)),
          'active':np.ones((len(t),2)),'inactive':np.ones((len(t),2)),
          'config':load_config('F5'),'method':'synthetic phase-validity check'}
    result=spin_summary(data)
    assert result['status']=='UNRESOLVED' and result['phase_slope'] is None
    assert result['phase_valid_fraction']==0.
    np.testing.assert_allclose(result['mean_z'],(1-1e-8)/(1+1e-8),atol=1e-14)


def test_slow_decay_is_not_mistaken_for_a_settled_cycle():
    # Each short half-window looks periodic; the longer amplitude lag exposes decay.
    t=np.arange(0.,18000.5,.5);oscillation=np.sin(2*np.pi*t/250.)
    z=.2+.2*np.exp(-t/30000.)*oscillation;theta=.2*oscillation
    density=np.column_stack(((1+z)/2,(1-z)/2))
    psi=np.sqrt(density)*np.column_stack((np.exp(.5j*theta),np.exp(-.5j*theta)))
    data={'time':t,'psi':psi,'active':np.ones_like(density),'inactive':np.ones_like(density),
          'config':load_config('F5',physical={'J':.005}),'method':'synthetic settling check'}
    result=classify_spin(data)
    assert result['status']=='UNRESOLVED'
    assert result['long_lag_amplitude_difference']>.2
    assert result['period_half_difference']<.02
    assert result['amplitude_half_difference']<.05



def test_compiled_spin_step_matches_joint_rk4():
    from numerics import _spin_local_inplace,spin_coefficients
    rng=np.random.default_rng(19)
    for model,J in (('M2',0.),('M3',.01)):
        c=load_config('N3',model=model,physical={'J':J});p=parameters(c)
        pump=rng.uniform(0,2,(2,8,8))
        state=(rng.normal(size=(2,8,8))+1j*rng.normal(size=(2,8,8)),
               rng.uniform(0,10,(2,8,8)),rng.uniform(0,10,(2,8,8)))
        expected=local_rk4(state,pump,p,.025);actual=tuple(x.copy() for x in state)
        _spin_local_inplace(*actual,pump,spin_coefficients(p),.025)
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=3e-15,rtol=2e-15)


def test_component_stationarity_accepts_phase_drift_and_rejects_hidden_exchange():
    from numerics import _stationarity,_history_row
    c=load_config('N3',space={'N':8,'L':8.});g=make_grid(c);p=parameters(c)
    def evaluate(exchange):
        frames=[];history=[]
        for t in np.arange(0.,101.,5.):
            density=np.array([1.+exchange*t/100,2.-exchange*t/100])
            psi=np.sqrt(density)*np.exp(1j*np.array([.3,-.2])*t)
            state=tuple(np.broadcast_to(v[:,None,None],(2,8,8)).copy()
                        for v in (psi,np.array([8.,9.]),np.array([12.,13.])))
            frames.append((t,state));history.append(_history_row(t,state,p,g,0.,0.))
        return _stationarity(frames,history,p)
    assert evaluate(0.)['status']=='PASS'
    result=evaluate(.02)
    assert result['total_populations']['passed']
    assert result['status']=='UNRESOLVED'


def test_spatial_suite_uses_saved_common_time_and_short_extensions(tmp_path,monkeypatch):
    import numerics
    monkeypatch.setattr(numerics,'ROOT',tmp_path)
    path=tmp_path/'results/runs/N3_eps_03_grid256';path.mkdir(parents=True)
    np.savez(path/'checkpoint.npz',step=round(2300/.025))
    monkeypatch.setattr(numerics,'spatial_spin_reference_checks',lambda:{'status':'PASS'})
    calls=[];exported=[]
    def run(config,name):
        t=config['time']['final'];calls.append((name,t))
        passed=not(name=='N3_eps_03_dt_half' and t<2400)
        return {'summary':{'run_status':'RUN_COMPLETED','time_ps':t,'two_windows_pass':passed,
                           'checks':'PASS' if passed else 'UNRESOLVED','symmetry_density_relative':0.}}
    monkeypatch.setattr(numerics,'run_2d',run)
    monkeypatch.setattr(numerics,'compare_spatial_spin',lambda a,b:{'status':'PASS'})
    monkeypatch.setattr(numerics,'export_spatial_spin_data',lambda r:exported.append(r['assessment']))
    result=numerics.run_spatial_spin_suite(max_workers=1)
    assert result['assessment']=='PASS'
    assert [t for _,t in calls]==[2300.]*4+[2400.]*4
    assert exported==['PASS']


def test_tetm_exact_limits_and_stokes_signs():
    from numerics import tetm_reference_checks
    result=tetm_reference_checks()
    assert result['status']=='PASS',result


def test_new_radial_seed_and_stokes_are_separate_from_legacy():
    from model import report3_config,stokes,normalized_stokes
    a=report3_config('T0',space={'N':32});b=report3_config('T0',space={'N':64})
    state=initial_state(a,make_grid(a));fine=initial_state(b,make_grid(b))
    np.testing.assert_array_equal(state[0],fine[0][:,::2,::2])
    assert np.all(state[0].imag==0) and np.all(state[1]==0) and np.all(state[2]==0)
    S=stokes(state[0]);s,valid=normalized_stokes(state[0])
    np.testing.assert_allclose(s[0,valid],1.,atol=2e-15)
    np.testing.assert_array_equal(S[2:],0.)


def test_report3_map_checks_reject_hidden_dynamics_and_require_long_windows(tmp_path):
    from numerics import _ProbeStore,analyze_report3,_report3_map_errors
    ref=np.ones((9,8,8));ref[1]=1e-8
    current=ref.copy();current[1]*=1.5
    assert _report3_map_errors(current,ref,ref[0]+ref[1])[1]<1e-8
    from model import report3_config
    c=report3_config('T0',space={'N':8,'L':8.})
    store=_ProbeStore(tmp_path/'maps.npy',8,duration=1100.)
    history=[];features=[]
    # Full polarization rotates with T=100 ps while both populations stay fixed.
    for t in np.arange(0.,1100.1,5.):
        theta=.4*np.sin(2*np.pi*t/100.)
        psi=np.broadcast_to(np.exp(1j*np.array([theta/2,-theta/2]))[:,None,None],(2,8,8)).copy()
        state=psi,np.ones((2,8,8))*8,np.ones((2,8,8))*10
        store.add(t,state)
        history.append([t,64.,64.,512.,512.,640.,640.])
        features.append([t,np.sin(2*np.pi*t/100.),np.cos(2*np.pi*t/100.)])
    r=analyze_report3(store,history,features,1100.,c)
    assert r['status']=='PERIODIC',r
    assert r['density_stationary'] and not r['polarization_stationary']
    assert r['window_length_ps']>=500.
    # Same scalar trace is insufficient when the maps do not recur.
    i=store.ordered()[-1];store.maps[i,0,0,0]*=1.3
    assert analyze_report3(store,history,features,1100.,c)['status']=='UNRESOLVED'
    short=_ProbeStore(tmp_path/'short.npy',8)
    for i in store.ordered(700.,1100.):short.add(store.times[i],store.state(i))
    assert analyze_report3(short,history,features,1100.,c)['status']=='UNRESOLVED'


def test_stokes_new_names_keep_old_classifier_physics():
    psi=np.array([[np.sqrt(2)*np.exp(.4j),1.+0j]])
    data={'config':load_config('F4'),'time':np.array([0.]),'psi':psi,
          'active':np.ones((1,2)),'inactive':np.ones((1,2))}
    o=observables_0d(data)
    np.testing.assert_allclose(o['S0'],3.)
    np.testing.assert_allclose(o['legacy_s1'],o['S1']/o['S0'])
    np.testing.assert_allclose(o['legacy_s2'],-o['S2']/o['S0'])
    p=parameters(data['config'])
    np.testing.assert_allclose(o['IJ'],-p['J']/p['hbar']*o['S2'])


def test_animation_archive_keeps_early_frames_after_probe_rollover(tmp_path,monkeypatch):
    import json,yaml,numerics
    from model import report3_config
    monkeypatch.setattr(numerics,'ROOT',tmp_path)
    path=tmp_path/'results/runs/report3/test';path.mkdir(parents=True)
    config=report3_config('B0',space={'N':8,'L':8.},time={'final':3000.})
    (path/'resolved_config.yaml').write_text(yaml.safe_dump(config))
    def checkpoint(times,end):
        maps=np.zeros((len(times),8,8,8))
        for i,t in enumerate(times):
            maps[i,0]=t+1.;maps[i,1]=2.;maps[i,2]=3.;maps[i,3]=t/10
        np.save(path/'probe_maps.npy',maps)
        np.savez(path/'checkpoint.npz',probe_times=times,step=round(end/.025),signature='same physics')
    checkpoint(np.arange(0.,40.1,5.),40.)
    row=numerics.save_report3_animation_frames(path,halfwidth=1.1)
    assert row['frames']==3 and row['missing_frames']==0
    checkpoint(np.arange(40.,80.1,5.),80.)
    row=numerics.save_report3_animation_frames(path,halfwidth=1.1)
    assert row['frames']==5 and row['added']==2
    target=tmp_path/'results/data/report3/test_animation.npz'
    with np.load(target,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['time'],[0.,20.,40.,60.,80.])
        assert z['density'].shape==(5,2,3,3) and z['density'].dtype==np.float32
        np.testing.assert_allclose(z['density'][:,0,0,0],(z['time']+1)**2+4)
        np.testing.assert_allclose(z['density'][:,1,0,0],9+(z['time']/10)**2)
        np.testing.assert_array_equal(z['components'],['+','-'])
        assert json.loads(str(z['metadata_json']))['complete_on_saved_cadence']
    # Checkpoint rollback removes frames from the uncommitted future.
    checkpoint(np.arange(40.,80.1,5.),60.)
    numerics.save_report3_animation_frames(path,halfwidth=1.1)
    with np.load(target) as z:np.testing.assert_array_equal(z['time'],[0.,20.,40.,60.])


def test_frozen_spin_branch_matches_scalar_and_accounts_for_spin_flip():
    from model import report3_config
    from numerics import frozen_spin_reservoirs,_growth_fit,_threshold_bounds
    c=report3_config('B0',space={'N':16,'L':16.});p=parameters(c);g=make_grid(c)
    a,i=frozen_spin_reservoirs(c,g);pump=injection(c,g)
    np.testing.assert_allclose(a,pump/p['gammaA'],rtol=2e-15)
    np.testing.assert_allclose(i,pump/p['W'],rtol=2e-15)
    np.testing.assert_allclose(p['gA']*a+p['gI']*i,p['gR']*(pump/p['gammaA']+p['eta']*pump),atol=1e-16)
    c=report3_config('A1',space={'N':16,'L':16.});a,i=frozen_spin_reservoirs(c,g)
    zero=np.zeros_like(a,dtype=complex)
    rhs=local_rhs((zero,a,i),injection(c,g),parameters(c))
    for v in rhs:np.testing.assert_allclose(v,0.,atol=2e-15)
    t=np.arange(401.)
    assert _growth_fit(t,.01*t)['status']=='UNSTABLE'
    assert _growth_fit(t,-.01*t)['status']=='STABLE'
    assert _growth_fit(t,0*t)['status']=='NEAR_ZERO'
    records=[{'geometry':'spot','case':'T0','p':p,'status':status,'channels':[{'status':sign}]} for p,status,sign in [(10.,'RESOLVED','STABLE'),(11.,'UNRESOLVED','UNSTABLE'),(12.,'RESOLVED','UNSTABLE')]]
    assert _threshold_bounds(records,'spot','T0',0)['hi']==12.


def test_periodic_stokes_lag_interpolation_resolves_short_cycles():
    from numerics import _lag_polynomial_weights
    t=np.arange(0.,301.,5.);period=55.5678051982;target=173.4
    exact=np.sin(2*np.pi*target/period)
    linear=np.interp(target,t,np.sin(2*np.pi*t/period))
    ix,w=_lag_polynomial_weights(t,target,6)
    corrected=np.dot(w,np.sin(2*np.pi*t[ix]/period))
    assert abs(linear-exact)>.005
    assert abs(corrected-exact)<.0002
    ix,w=_lag_polynomial_weights(t,175.,6)
    np.testing.assert_allclose(np.dot(w,t[ix]**2),175.**2,rtol=0,atol=1e-10)


def test_period_detection_does_not_amplify_tiny_reservoir_drift():
    from numerics import _period_from_features
    t=np.arange(0.,2000.25,.25);features=np.ones((len(t),17));features[:,9:]=0.
    features[:,2]+=1e-6*t;features[:,3]-=1e-6*t
    features[:,6]=.4*np.cos(2*np.pi*t/80.);features[:,7]=.4*np.sin(2*np.pi*t/80.)
    assert abs(_period_from_features(t,features)-80.)<.01


def test_field_precision_and_exact_J_rotation():
    import torch
    from model import report3_config
    from numerics import TorchSpinStepper,_precision_advance
    for precision,tolerance in [('complex64',2e-6),('complex128',2e-11)]:
        c=report3_config('J1',space={'N':8,'L':8.},execution={'precision':precision})
        backend=TorchSpinStepper(c,device='cpu')
        for key in ('alpha1','alpha2','gA','gI','R','gammaC'):backend.p[key]=0.
        z=np.zeros((2,8,8),complex);z[0]=1.
        state=(z,np.zeros(z.shape),np.zeros(z.shape))
        expected=torch.complex64 if precision=='complex64' else torch.complex128
        assert backend.to_device(state)[0].dtype==expected
        result=_precision_advance(backend,state,2.)
        angle=backend.p['J']*2/backend.p['hbar']
        np.testing.assert_allclose(result[0][0],np.cos(angle),rtol=0,atol=tolerance)
        np.testing.assert_allclose(result[0][1],1j*np.sin(angle),rtol=0,atol=tolerance)
        assert backend.last_condensate[0].dtype==torch.float64


def test_precision_audit_preserves_relative_phase_and_smooth_perturbations():
    from model import report3_config
    from numerics import _precision_metrics,_precision_perturb,_precision_state_error
    g=make_grid(report3_config('B0',space={'N':8,'L':8.}))
    state=(np.ones((2,8,8),complex),np.ones((2,8,8)),np.ones((2,8,8)))
    rotated=(state[0]*np.exp(.4j),*state[1:])
    assert _precision_state_error(rotated,state)<1e-14
    assert max(_precision_metrics(rotated,state,g)['normalized_stokes_weighted_RMS'])<1e-14
    changed=state[0].copy();changed[0]*=np.exp(.4j)
    assert max(_precision_metrics((changed,*state[1:]),state,g)['normalized_stokes_weighted_RMS'])>.3
    a=_precision_perturb(state,g,1e-5,123);b=_precision_perturb(state,g,5e-6,123)
    ratio=_precision_state_error(a,state)/_precision_state_error(b,state)
    assert abs(ratio-2)<1e-4


def test_full_precision_comparison_on_shared_stationary_interval(tmp_path,monkeypatch):
    import json,yaml,numerics
    from model import report3_config
    monkeypatch.setattr(numerics,'ROOT',tmp_path)
    out=tmp_path/'results/data/report3';out.mkdir(parents=True)
    for name,precision,amplitude in [('ref','complex128',1.),('single','complex64',1.00001)]:
        c=report3_config('B0',space={'N':8,'L':8.},time={'final':600.},execution={'precision':precision})
        path=tmp_path/'results/runs/report3'/name;path.mkdir(parents=True)
        (path/'resolved_config.yaml').write_text(yaml.safe_dump(c))
        store=numerics._ProbeStore(path/'probe_maps.npy',8)
        z=np.ones((2,8,8),complex)*amplitude;a=np.ones((2,8,8));state=(z,a,a)
        for t in np.arange(0.,600.1,5.):store.add(t,state)
        store.maps.flush();np.savez(path/'checkpoint.npz',probe_times=store.times)
        tt=np.arange(0.,600.1,.25);h=np.zeros((len(tt),11));h[:,0]=tt
        h[:,1:3]=64*amplitude**2;h[:,3:7]=64.
        features=np.zeros((len(tt),18));features[:,:7]=h[:,:7];features[:,7]=128*amplitude**2
        np.savez(out/(name+'.npz'),history=h,features=features)
    r=numerics.compare_report3_precision('ref','single')
    assert r['status']=='PASS' and r['arithmetic_observables_pass']
    assert r['enough_shared_history'] and r['interval_ps']==[200.,600.]
    assert max(r['population_max_relative_error_same_time'])<3e-5
    assert len(r['map_samples'])==21


def test_periodic_classifier_ignores_float32_jitter_in_constant_channels(tmp_path):
    from model import report3_config,stokes
    from numerics import _ProbeStore,analyze_report3
    c=report3_config('T0',space={'N':8,'L':8.})
    store=_ProbeStore(tmp_path/'maps.npy',8,duration=1100.)
    history=[];features=[]
    for t in np.arange(0.,1100.1,5.):
        theta=.4*np.sin(2*np.pi*t/100.)
        psi=np.broadcast_to(np.exp(1j*np.array([theta/2,-theta/2]))[:,None,None],(2,8,8)).copy()
        a=np.ones((2,8,8))*8*(1+4e-7*(1+t/1100)*np.sin(.73*t))
        b=np.ones((2,8,8))*10;store.add(t,(psi,a,b))
        pop=np.r_[np.array([64.,64.]),a.sum((1,2)),b.sum((1,2))]
        history.append([t,*pop]);features.append([t,*pop,*stokes(psi)[1:].sum((1,2)),*np.zeros(8)])
    r=analyze_report3(store,history,features,1100.,c)
    assert r['status']=='PERIODIC',r
    assert set(r['periodic']['amplitude_significant_channels']).issubset({6,7})


def test_controlled_seed_is_identical_across_domains_and_has_no_initial_current():
    from model import report3_config
    c=report3_config('J1','ring',3.3,space={'N':32,'L':16.},
                     initial={'kind':'radial_perturbed','perturbation':1e-3,'perturb_seed':20261025})
    a=initial_state(c,make_grid(c))[0]
    d=copy.deepcopy(c);d['space']={'N':48,'L':24.};b=initial_state(d,make_grid(d))[0]
    np.testing.assert_array_equal(a,b[:,8:40,8:40])
    assert np.max(abs(a.imag))==0 and a.real.min()>=0
    e=copy.deepcopy(c);e['initial']['kind']='radial';plain=initial_state(e,make_grid(e))[0]
    assert np.linalg.norm(a-plain)>0


def test_declarative_queue_rebuilds_missing_fork_exports_without_steps(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    import json
    sys.path.insert(0,str(Path(__file__).parent/'scripts'))
    import experiment_queue as q
    c={'time':{'dt':.025,'final':3600.}}
    from types import SimpleNamespace
    monkeypatch.setattr(q,'compute_imports',lambda:None)
    monkeypatch.setattr(q,'saved',SimpleNamespace(RUNS=tmp_path,DATA=tmp_path,fork_run=lambda source,name:None))
    monkeypatch.setattr(q,'resolve',lambda job:c)
    monkeypatch.setattr(q,'checkpoint',lambda name,config:3600.)
    monkeypatch.setattr(q,'artifacts_ready',lambda name,t:False)
    monkeypatch.setattr(q.saved,'fork_run',lambda source,name:None)
    calls=[]
    def finalize(config,name,**kwargs):
        calls.append(kwargs)
        return {'summary':{'time_ps':3600.,'run_status':'RUN_COMPLETED'}}
    monkeypatch.setattr(q,'run_report3_pilot',finalize)
    monkeypatch.setattr(q,'review_saved',lambda name,end:{'status':'STATIONARY'})
    record={}
    q.execute({'id':'copy','source':'reference','name':'copy','final_ps':3600.,'resume_from':'reference'},record)
    assert calls==[{'max_steps':0}]
    assert record['action']=='REBUILD_EXPORTS' and record['status']=='COMPLETED'
    monkeypatch.setattr(q,'artifacts_ready',lambda name,t:True)
    (tmp_path/'copy').mkdir();(tmp_path/'copy/summary.json').write_text(json.dumps({'time_ps':3600.,'run_status':'RUN_COMPLETED'}))
    monkeypatch.setattr(q.saved,'RUNS',tmp_path)
    q.execute({'id':'copy','source':'reference','name':'copy','final_ps':3600.},record)
    assert len(calls)==1 and record['action']=='REUSE'


def test_declarative_queue_rejects_dependency_cycles_and_wrong_checkpoint(tmp_path):
    import sys
    from pathlib import Path
    import pytest,yaml
    sys.path.insert(0,str(Path(__file__).parent/'scripts'))
    import experiment_queue as q
    plan=tmp_path/'jobs.yaml'
    plan.write_text(yaml.safe_dump({'experiments':[{'id':'a','name':'a','depends_on':['b']},{'id':'b','name':'b','depends_on':['a']}]}))
    with pytest.raises(ValueError,match='dependency'):
        q.load_plan(plan)
    config={'time':{'dt':.025,'final':3600.}}
    assert q.signature(config)==q.signature({'time':{'dt':.025,'final':6000.}})
    assert q.signature(config)!=q.signature({'time':{'dt':.0125,'final':3600.}})


def test_full_map_cycle_amplitude_survives_cancelling_integrals():
    import sys
    from pathlib import Path
    from model import stokes
    sys.path.insert(0,str(Path(__file__).parent/'scripts'))
    from check_ring_cycles import cycle_statistics,amplitude_errors
    times=np.arange(0.,1000.1,5.)
    checker=(-1.)**np.indices((8,8)).sum(0)
    def maps(i):
        psi=np.stack((np.ones((8,8),complex),checker*np.exp(2j*np.pi*times[i]/100)))
        S=stokes(psi)
        return np.concatenate((abs(psi)**2,np.ones((4,8,8)),S[1:]))
    a,aa,_=cycle_statistics(times,maps,0.,100.)
    b,ab,_=cycle_statistics(times,maps,500.,100.)
    assert abs(maps(7)[6].sum())<1e-12  # Integral vanishes despite full local oscillation.
    assert np.max(abs(aa[6:8]-np.sqrt(2)))<1e-12
    assert max(amplitude_errors(aa,ab,a[0]+a[1])[6:8])<1e-12
    assert np.max(abs(a-b))<1e-12


def test_full_map_cycle_amplitude_rejects_hidden_amplitude_change():
    import sys
    from pathlib import Path
    from model import stokes
    sys.path.insert(0,str(Path(__file__).parent/'scripts'))
    from check_ring_cycles import cycle_statistics,amplitude_errors
    times=np.arange(0.,1000.1,5.);checker=(-1.)**np.indices((8,8)).sum(0)
    def maps(i):
        amplitude=.3 if times[i]<500 else .6
        phase=amplitude*np.sin(2*np.pi*times[i]/100)
        psi=np.stack((np.ones((8,8),complex),checker*np.exp(1j*phase)))
        return np.concatenate((abs(psi)**2,np.ones((4,8,8)),stokes(psi)[1:]))
    a,aa,_=cycle_statistics(times,maps,0.,100.)
    b,ab,_=cycle_statistics(times,maps,500.,100.)
    assert max(amplitude_errors(aa,ab,a[0]+a[1])[6:8])>.01


def test_report3_pause_keeps_state_and_skips_expensive_analysis(tmp_path,monkeypatch):
    import pytest,torch,numerics as n
    from pathlib import Path
    from model import report3_config
    for name in ('model.py','numerics.py'):(tmp_path/name).write_text('test source')
    monkeypatch.setattr(n,'ROOT',tmp_path)
    actual=n.TorchSpinStepper
    monkeypatch.setattr(n,'TorchSpinStepper',lambda c,g,compiled=True:actual(c,g,device='cpu',compiled=False))
    monkeypatch.setattr(torch.xpu,'get_device_name',lambda:'CPU test')
    def no_analysis(*args,**kwargs):raise AssertionError('A paused run must not start full analysis')
    full_analysis=n.analyze_report3
    monkeypatch.setattr(n,'analyze_report3',no_analysis)
    c=report3_config('T0',space={'N':8,'L':32.},time={'final':1.})
    first=n.run_report3_pilot(c,'pause_test',max_steps=10)
    assert first['summary']['run_status']=='RUN_INCOMPLETE'
    assert first['summary']['time_ps']==.25
    resumed=n.run_report3_pilot(c,'pause_test',max_steps=10)
    reference=n.run_report3_pilot(c,'reference',max_steps=20)
    assert resumed['summary']['time_ps']==reference['summary']['time_ps']==.5
    with np.load(resumed['path']/'checkpoint.npz') as a,np.load(reference['path']/'checkpoint.npz') as b:
        for key in ('psi','active','inactive','history','integral','cond_integral'):
            np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-14)
    assert not (tmp_path/'results/data/report3/pause_test.npz').exists()
    # Interrupting export preserves the previous archive; resume performs zero new steps.
    monkeypatch.setattr(n,'analyze_report3',full_analysis)
    archive=tmp_path/'results/data/report3/pause_test.npz'
    np.savez(archive,sentinel=np.array([123]));original=archive.read_bytes()
    compressed=np.savez_compressed
    def interrupt_export(path,*args,**kwargs):
        if Path(path).name=='pause_test.tmp.npz':
            Path(path).write_bytes(b'incomplete export')
            raise KeyboardInterrupt
        return compressed(path,*args,**kwargs)
    monkeypatch.setattr(np,'savez_compressed',interrupt_export)
    with pytest.raises(KeyboardInterrupt):n.run_report3_pilot(c,'pause_test')
    assert archive.read_bytes()==original
    with np.load(first['path']/'checkpoint.npz') as z:
        step=int(z['step']);fields=[z[k].copy() for k in ('psi','active','inactive')]
    monkeypatch.setattr(np,'savez_compressed',compressed)
    rebuilt=n.run_report3_pilot(c,'pause_test',max_steps=0)
    assert rebuilt['summary']['run_status']=='RUN_COMPLETED'
    with np.load(archive) as z:
        for key,field in zip(('psi','active','inactive'),fields):np.testing.assert_array_equal(z[key],field)
    with np.load(first['path']/'checkpoint.npz') as z:assert int(z['step'])==step


def test_queue_add_note_extend_and_measured_eta(tmp_path,monkeypatch):
    import sys,yaml,json
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).parent/'scripts'))
    import experiment_queue as q
    monkeypatch.setattr(q,'ROOT',tmp_path);monkeypatch.setattr(q,'RUNS',tmp_path/'runs');monkeypatch.setattr(q,'DATA',tmp_path/'data')
    (tmp_path/'scripts').mkdir();(q.RUNS/'reference').mkdir(parents=True);q.DATA.mkdir()
    c=load_config('F4');c['space']={'N':8,'L':8.}
    (q.RUNS/'reference/resolved_config.yaml').write_text(yaml.safe_dump(c))
    plan=tmp_path/'jobs.yaml';plan.write_text('experiments: []\n')
    args=SimpleNamespace(command='add',item='demo',source='reference',until=2000.,priority=10,note='First sample',after=[],resume_from=None,set=['physical.J=0.005'])
    q.edit_plan(args,plan);job=q.load_plan(plan)['experiments'][0]
    assert job['parameters']['physical']['J']==.005 and job['note']=='First sample'
    args.command='extend';args.until=3000.;q.edit_plan(args,plan)
    args.command='note';args.text='Check the phase';q.edit_plan(args,plan)
    job=q.load_plan(plan)['experiments'][0];assert job['final_ps']==3000. and job['note']=='Check the phase'
    (q.DATA/'benchmark.json').write_text(json.dumps({'config':c,'summary':{'run_status':'RUN_COMPLETED','time_ps':1000.,'elapsed_wall_s':100.}}))
    assert 'unknown' not in q.eta({'experiments':[job]},{})
    job['parameters']['execution']={'precision':'complex64'}
    assert 'Not enough data' in q.eta({'experiments':[job]},{})
    args.command='disable';q.edit_plan(args,plan);assert not q.pending_jobs(q.load_plan(plan),{})
    args.command='enable';q.edit_plan(args,plan);assert len(q.pending_jobs(q.load_plan(plan),{}))==1


def test_queue_pause_during_review_and_resume(tmp_path,monkeypatch):
    import sys,subprocess,time,json
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parent/'scripts'))
    import experiment_queue as q
    state=tmp_path/'queue.json';runs=tmp_path/'runs';runs.mkdir();release=tmp_path/'release'
    code=f'''import sys,time,ctypes
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,{str(Path(__file__).parent/'scripts')!r})
import experiment_queue as q
q.STATE=Path({str(state)!r});q.RUNS=Path({str(runs)!r})
q.log_dir=lambda: q.ROOT/'results/runs/test_logs'
q.log_dir().mkdir(parents=True,exist_ok=True)
q.load_plan=lambda *a,**k:{{'experiments':[{{'id':'demo','name':'demo','final_ps':1.}}]}}
q.fingerprint=lambda job:'same physics'
q.compute_imports=lambda:None
q.saved=SimpleNamespace(short_checks=lambda:{{'status':'PASS'}})
ctypes.CDLL=lambda *a:None
def work(job,record,notify):
    record['phase']='REVIEW';notify()
    while not Path({str(release)!r}).exists():time.sleep(.05)
    record.update(status='COMPLETED',end_ps=1.,review={{'status':'STATIONARY'}})
q.execute=work
q.run(Path('unused'),'Test launch note')
'''
    child=subprocess.Popen([sys.executable,'-c',code,'experiment_queue.py'],cwd=Path(__file__).parent,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if state.exists() and json.loads(state.read_text()).get('active_experiment')=='demo':break
            time.sleep(.05)
        else:raise AssertionError('Queue test worker did not start')
        monkeypatch.setattr(q,'STATE',state)
        q.pause(wait=2);assert child.wait(timeout=3)==0
        assert json.loads(state.read_text())['status']=='PAUSED'
        release.touch()
        resumed=subprocess.run([sys.executable,'-c',code,'experiment_queue.py'],cwd=Path(__file__).parent,capture_output=True,timeout=5)
        assert resumed.returncode==0,resumed.stderr
        result=json.loads(state.read_text());assert result['status']=='COMPLETED'
        assert len(result['attempts'])==2 and result['attempts'][-1]['note']=='Test launch note'
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=3)
