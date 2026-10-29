"""Independent CPU audit of saved ring cycles; original classifier and runs stay intact."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from numerics import _report3_maps,_report3_map_errors,_lag_polynomial_weights,write_json

LABELS=['n_plus','n_minus','A_plus','A_minus','I_plus','I_minus','S1','S2','S3']


def cycle_statistics(times,load_maps,left,period,cycles=5):
    """Positive trapezoid weights over complete cycles; pointwise weighted variance."""
    end=left+cycles*period
    if left<times[0]-1e-8 or end>times[-1]+1e-8:raise ValueError('Missing complete-cycle history')
    grid=np.r_[left,times[(times>left+1e-8)&(times<end-1e-8)],end]
    weights=np.r_[np.diff(grid)[0]/2,(grid[2:]-grid[:-2])/2,np.diff(grid)[-1]/2]
    mean=None;m2=None;total=0.
    for t,w in zip(grid,weights):
        exact=np.flatnonzero(np.isclose(times,t,rtol=0,atol=1e-8))
        if len(exact):value=load_maps(int(exact[0]))
        else:
            ix,coeff=_lag_polynomial_weights(times,t,8)
            value=sum(c*load_maps(int(i)) for i,c in zip(ix,coeff))
        if mean is None:mean=np.zeros_like(value);m2=np.zeros_like(value)
        delta=value-mean;new_total=total+w;mean+=w/new_total*delta
        m2+=w*delta*(value-mean);total=new_total
    return mean,np.sqrt(np.maximum(m2/total,0.)),{'interval_ps':[float(left),float(end)],'cycles':cycles}


def amplitude_errors(first,second,mean_S0):
    difference=np.linalg.norm((first-second).reshape(9,-1),axis=1)
    denominator=np.maximum(np.linalg.norm(first.reshape(9,-1),axis=1),np.linalg.norm(second.reshape(9,-1),axis=1))
    return difference/np.maximum(denominator,1e-6*np.linalg.norm(mean_S0))


def main():
    D=ROOT/'results/data/report3';R=ROOT/'results/runs/report3'
    queue=json.loads((D/'experiment_queue.json').read_text())
    report={'method':'Two original >=5-period windows; same periods/recurrence tolerances. Compare pointwise full-map RMS oscillation amplitudes (not spatially cancelling integrals) over5 complete cycles, and cycle means. Map channels stationary within the original1% criterion are reported as stationary, not tested for a nonzero relative oscillation amplitude. Original classifier flags and all run data remain unchanged.',
        'tolerances':{'period_relative':.01,'map_amplitude_relative':.01,'mean_map_relative':.01,'mean_population_Stokes_relative':.01,'map_recurrence_relative':.01,'interpolation_6_8_relative':.002},
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':{}}
    for key in ('ring_A0_late','ring_A1_late','ring_T1_late'):
        original=queue['experiments'][key];name=original['name'];review=original['review'];p=review['periodic']
        with np.load(R/name/'checkpoint.npz',allow_pickle=False) as z:probe_times=z['probe_times'].copy();features=z['features'].copy()
        raw=np.load(R/name/'probe_maps.npy',mmap_mode='r');order=np.flatnonzero(np.isfinite(probe_times));order=order[np.argsort(probe_times[order])];times=probe_times[order]
        def load(i):
            v=raw[order[i]];psi=np.stack((v[0]+1j*v[1],v[2]+1j*v[3]))
            return _report3_maps((psi,v[4:6],v[6:8]))[0]
        stats=[];means_features=[]
        for window,period in zip(review['windows'],p['periods_ps']):
            stats.append(cycle_statistics(times,load,window['interval_ps'][0],period))
            interval=stats[-1][2]['interval_ps'];a,b=interval
            ft=np.r_[a,features[(features[:,0]>a)&(features[:,0]<b),0],b]
            fv=np.column_stack([np.interp(ft,features[:,0],features[:,j]) for j in range(1,10)])
            means_features.append(np.trapezoid(fv,ft,axis=0)/(b-a))
            print(name,'complete-cycle maps read',interval,flush=True)
        m0,a0,_=stats[0];m1,a1,_=stats[1];S0=m0[0]+m0[1]
        centre=_report3_map_errors(m1,m0,S0);amp=amplitude_errors(a0,a1,S0)
        old_changes=np.maximum(review['windows'][0]['map_errors'],review['windows'][1]['map_errors'])
        active=np.flatnonzero(old_changes>=.01)
        fa,fb=means_features;scale=np.maximum(np.maximum(abs(fa),abs(fb)),1e-30)
        for j in (0,2,4):scale[j:j+2]=np.maximum(scale[j:j+2],1e-6*max(float(fa[j:j+2].sum()),float(fb[j:j+2].sum())))
        scale[6:]=max(float(fa[:2].sum()),float(fb[:2].sum()));feature_centre=abs(fa-fb)/scale
        # Independent pointwise relative phase at a bright point with large Stokes oscillation.
        score=a1[6]**2+a1[7]**2;iy,ix=np.unravel_index(np.argmax(score),score.shape)
        kept=np.flatnonzero((times>=stats[0][2]['interval_ps'][0])&(times<=6000.))
        vv=raw[order[kept],:,iy,ix];plus=vv[:,0]+1j*vv[:,1];minus=vv[:,2]+1j*vv[:,3]
        coherence=plus.conj()*minus;total=abs(plus)**2+abs(minus)**2
        trace={'time_ps':times[kept],'S1_over_S0':2*coherence.real/total,'S2_over_S0':2*coherence.imag/total,'S3_over_S0':(abs(plus)**2-abs(minus)**2)/total,'fits':[]}
        for interval in [v[2]['interval_ps'] for v in stats]:
            use=(trace['time_ps']>=interval[0])&(trace['time_ps']<=interval[1]);tt=trace['time_ps'][use];phase=np.unwrap(np.angle(coherence[use]));fit=np.polyfit(tt-tt[0],phase,1)
            trace['fits'].append({'interval_ps':interval,'frequency_difference_minus_theta_dot_rad_ps':float(fit[0]),'phase_linear_fit_RMS_rad':float(np.sqrt(np.mean((phase-np.polyval(fit,tt-tt[0]))**2))),'period_from_local_phase_ps':float(2*np.pi/abs(fit[0]))})
        passed=(p['period_relative_difference']<.01 and max(p['map_recurrence_errors'])<.01 and max(p['interpolation_difference_6_8'])<.002 and max(centre)<.01 and max(feature_centre)<.01 and (not len(active) or max(amp[active])<.01))
        cfg=json.loads((D/(name+'.json')).read_text())['config'];dx=cfg['space']['L']/cfg['space']['N'];point=[-cfg['space']['L']/2+ix*dx,-cfg['space']['L']/2+iy*dx]
        result={'name':name,'source_status':review['status'],'audit_status':'PERIODIC_OBSERVABLES_PASS' if passed else 'UNRESOLVED','intervals':[v[2] for v in stats],'periods_ps':p['periods_ps'],'period_relative_difference':p['period_relative_difference'],'map_recurrence_max':max(p['map_recurrence_errors']),'mean_map_errors':dict(zip(LABELS,centre)),'mean_feature_errors':feature_centre,'RMS_amplitude_relative_errors':dict(zip(LABELS,amp)),'oscillating_map_channels':[LABELS[i] for i in active],'amplitude_max_active':float(max(amp[active])) if len(active) else 0.,'point_xy_um':point,'local_component_min_fraction':float(np.min(np.minimum(abs(plus)**2,abs(minus)**2)/total)),'trace':trace,'scope':'Finite-time periodic densities/reservoirs/Stokes in stated tolerances. No claim of mathematical attraction or universal initial-state outcome.'}
        report['cases'][name]=result
        data=json.loads((D/'report_readiness.json').read_text());data['full_map_cycle_recheck']=report;write_json(D/'report_readiness.json',data)
        print(name,result['audit_status'],'amplitude',result['amplitude_max_active'],'centre',max(centre),'local periods',[v['period_from_local_phase_ps'] for v in trace['fits']],flush=True)
    make_plot(report)


def make_plot(report):
    import matplotlib.pyplot as plt
    from plotting import STYLE,WIDTH,save
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(3,1,figsize=(WIDTH,112/25.4),layout='constrained',sharex=True)
        for ax,(name,row) in zip(axes,report['cases'].items()):
            trace=row['trace'];t=np.asarray(trace['time_ps']);use=t>=5000
            ax.plot(t[use]-5000,np.asarray(trace['S1_over_S0'])[use],label='$s_1$',lw=1)
            ax.plot(t[use]-5000,np.asarray(trace['S2_over_S0'])[use],label='$s_2$',lw=1)
            label=name.replace('matched_','').replace('_ring','')
            ax.set(title=label+f": local polarization at ({row['point_xy_um'][0]:g}, {row['point_xy_um'][1]:g}) "+r'$\mu$m',ylabel='Local Stokes',ylim=(-1.05,1.05))
        axes[0].legend(frameon=False,ncol=2,loc='upper right');axes[-1].set_xlabel('Time after5000 ps (ps)')
        save(fig,ROOT/'results/diagnostics/report3','ring_cycle_recheck')

if __name__=='__main__':main()
