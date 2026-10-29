"""Compact local histories; --apply removes probes after preserving restart and plot data."""
from pathlib import Path
import argparse,json,sys,fcntl
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'results/runs/report3';DATA=ROOT/'results/data/report3'

def compact(names=None,apply=False,prepare=False):
    state=DATA/'experiment_queue.json'
    q=json.loads(state.read_text()) if state.exists() else {}
    if q.get('status') in ('RUNNING','PREFLIGHT','WAITING_FOR_GPU'):
        raise RuntimeError('Queue is active; pause it before compacting histories')
    import yaml
    plan=yaml.safe_load((ROOT/'scripts/experiments.yaml').read_text())
    protected={j['name'] for j in plan['experiments'] if j.get('keep_dense_fields',False)}
    paths=[p for p in sorted(RUNS.glob('*/probe_maps.npy')) if p.parent.name not in protected]
    if names is not None:paths=[p for p in paths if p.parent.name in names]
    released=sum(p.stat().st_blocks*512 for p in paths)
    if not apply and not prepare:return {'runs':len(paths),'potential_GiB':released/1024**3}
    # Archives required by the current report and animations must already exist.
    for name in ('stokes_animation_compact.npz','regime_center_data.npz','spot_J_diagnostic.json'):
        if not (DATA/name).exists():raise RuntimeError('Missing portable evidence: '+name)
    archive=DATA/'local_phase_histories.npz';values={};meta={}
    if archive.exists():
        with np.load(archive,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json']));values={k:z[k].copy() for k in z.files if k!='metadata_json'}
    validated=[]
    for path in paths:
        name=path.parent.name;cp=path.parent/'checkpoint.npz'
        if not cp.exists():raise RuntimeError('No restart checkpoint: '+name)
        with np.load(cp,allow_pickle=False) as z:
            times=z['probe_times'].copy()
            for k in ('psi','active','inactive'):assert np.isfinite(z[k]).all(),(name,k)
            assert int(z['step'])>=0 and str(z['signature'])
        maps=np.load(path,mmap_mode='r',allow_pickle=False)
        if len(times)!=len(maps):raise RuntimeError('Probe/checkpoint size mismatch: '+name)
        ids=np.flatnonzero(np.isfinite(times));ids=ids[np.argsort(times[ids])];n=maps.shape[-1]
        import yaml
        c=yaml.safe_load((path.parent/'resolved_config.yaml').read_text());dx=c['space']['L']/n
        xy=[(0,0),(2,0),(-2,0),(0,2),(0,-2)];points=[]
        for x,y in xy:
            v=np.asarray(maps[ids,:4,round(n/2+y/dx),round(n/2+x/dx)])
            points.append(np.stack((v[:,0]+1j*v[:,1],v[:,2]+1j*v[:,3]),axis=1))
        values[name+'_time']=times[ids];values[name+'_psi']=np.stack(points,axis=1)
        meta[name]={'xy_um':xy,'source':'probe_maps.npy','dtype':'complex128','frames':len(ids),'checkpoint_preserved':True}
        validated.append((path,cp))
    values['metadata_json']=np.array(json.dumps(meta));tmp=archive.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**values)
    with np.load(tmp,allow_pickle=False) as z:
        for k,v in values.items():assert np.array_equal(z[k],v),k
    tmp.replace(archive)
    if prepare:
        plan={'files':[str(p.relative_to(ROOT)) for p,cp in validated],'runs':len(validated),'potential_GiB':released/1024**3,'phase_archive_MiB':archive.stat().st_size/1024**2,'loss':'Deleting full histories prevents exact full-map reanalysis of their old time windows without rerunning; checkpoints, final fields, selected maps, aggregate histories, local phase traces and report/GIF data remain.'}
        (DATA/'storage_compaction_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        return plan
    audit=DATA/'storage_compaction.json';report=json.loads(audit.read_text()) if audit.exists() else {'removed':[]}
    # Checkpoint physical state/signature is unchanged. Only invalidated probe times are cleared.
    for path,cp in validated:
        with np.load(cp,allow_pickle=False) as z:checkpoint={k:z[k].copy() for k in z.files}
        checkpoint['probe_times']=np.empty(0,dtype=float)
        tmp=cp.with_suffix('.compact.npz');np.savez_compressed(tmp,**checkpoint)
        with np.load(tmp,allow_pickle=False) as z:
            for k in ('psi','active','inactive','step','signature','history','features'):
                assert np.array_equal(z[k],checkpoint[k]),(str(cp),k)
        # Clear old indexes first: a restart remains safe even if interrupted before unlink.
        tmp.replace(cp);size=path.stat().st_blocks*512;path.unlink()
        report['removed'].append({'run':path.parent.name,'bytes':size,'retained_checkpoint':str(cp.relative_to(ROOT))})
        t=audit.with_suffix('.tmp.json');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(audit)
    report.update(released_GiB=sum(r['bytes'] for r in report['removed'])/1024**3,
                  note='Dense full-map histories removed. Final states, configurations, aggregate histories, selected exported maps and phase traces retained. Exact full-map reanalysis of deleted intervals requires rerunning. Restart physical state unchanged; new diagnostics need a fresh dense window.')
    audit.write_text(json.dumps(report,indent=2)+'\n')
    return {'runs':len(validated),'released_GiB':released/1024**3,'phase_archive_MiB':archive.stat().st_size/1024**2}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='append');args=parser.parse_args()
    # The same lock used by the queue prevents cleanup during an Arc trajectory.
    with (RUNS/'pilots.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(compact(args.run,args.apply,args.prepare),indent=2))
