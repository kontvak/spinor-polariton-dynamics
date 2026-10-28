"""Local Arc queue: add, extend, start, status/eta, pause/stop, resume, note, check."""
from pathlib import Path
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'local'))
DEFAULT_PLAN=ROOT/'local/experiments.yaml'
DATA=ROOT/'results/data/report3';RUNS=ROOT/'results/runs/report3'
STATE=DATA/'experiment_queue.json'
saved=run_report3_pilot=review_saved=retained_fields=None


def now():return datetime.now(timezone.utc).isoformat()
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else copy.deepcopy(default)
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n');tmp.replace(path)

def compute_imports():
    global saved,run_report3_pilot,review_saved,retained_fields
    if saved is None:
        import run_charge_checks as saved_module
        from numerics import run_report3_pilot as pilot
        from run_report3_series import review_saved as review,retained_fields as retain
        saved,run_report3_pilot,review_saved,retained_fields=saved_module,pilot,review,retain

def merge(config,patch):
    for key,value in patch.items():
        if isinstance(value,dict):merge(config.setdefault(key,{}),value)
        else:config[key]=value
    return config

def signature(config):
    c=copy.deepcopy(config);c['time'].pop('final')
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()

def resolve(job):
    import yaml
    from model import parameters,make_grid
    c=merge(yaml.safe_load((RUNS/job['source']/'resolved_config.yaml').read_text()),copy.deepcopy(job.get('parameters',{})))
    c['time']['final']=float(job['final_ps']);parameters(c);make_grid(c)
    dt=c['time']['dt'];end=c['time']['final']
    if not math.isfinite(dt) or dt<=0 or not math.isfinite(end) or end<=0 or abs(round(end/dt)*dt-end)>1e-8:raise ValueError('Invalid times: '+job['id'])
    return c

def checkpoint(name,config):
    import numpy as np
    path=RUNS/name/'checkpoint.npz'
    if not path.exists():return 0.
    with np.load(path,allow_pickle=False) as z:
        if str(z['signature'])!=signature(config):raise ValueError('Checkpoint parameters differ: '+name)
        return int(z['step'])*config['time']['dt']

def load_plan(path,validate=True):
    import yaml
    plan=yaml.safe_load(path.read_text()) or {};jobs=plan.setdefault('experiments',[])
    for field in ('id','name'):
        values=[j[field] for j in jobs]
        if len(set(values))!=len(values):raise ValueError('Duplicate '+field)
    seen=set();pending=jobs.copy()
    while pending:
        ready=[j for j in pending if set(j.get('depends_on',[]))<=seen]
        if not ready:raise ValueError('Unknown dependency or dependency cycle')
        for job in ready:
            for key in ('id','name','source'):
                if not re.fullmatch(r'[A-Za-z0-9_.-]+',job.get(key,'')) or job[key] in ('.','..'):raise ValueError('Invalid '+key)
            if job.get('priority',50)<0:raise ValueError('Priority must be nonnegative')
            seen.add(job['id']);pending.remove(job)
    if validate:
        for job in jobs:
            c=resolve(job);t=checkpoint(job['name'],c)
            if t>job['final_ps']+1e-8:raise ValueError('Target precedes checkpoint: '+job['id'])
            if job.get('resume_from'):checkpoint(job['resume_from'],c)
    return plan

def fingerprint(job):
    return hashlib.sha256(json.dumps({'name':job['name'],'signature':signature(resolve(job)),'resume_from':job.get('resume_from')},sort_keys=True).encode()).hexdigest()

def pending_jobs(plan,state):
    return [j for j in plan['experiments'] if j.get('enabled',True) and (state.get('experiments',{}).get(j['id'],{}).get('status')!='COMPLETED' or state['experiments'][j['id']].get('end_ps',0)<j['final_ps']-1e-8)]

def alive(state):
    try:
        proc=Path('/proc')/str(state['pid'])
        return b'experiment_queue.py' in (proc/'cmdline').read_bytes() and (proc/'cwd').resolve()==ROOT
    except (KeyError,OSError):return False

def progress(job):
    p=read(RUNS/job['name']/'progress.json',{})
    if not p and job.get('resume_from'):p=read(RUNS/job['resume_from']/'progress.json',{})
    return p

def artifacts_ready(name,t):
    import numpy as np
    run=RUNS/name;export=DATA/(name+'.npz')
    if not all(p.exists() for p in (run/'summary.json',run/'resolved_config.yaml',DATA/(name+'.json'),export)):return False
    try:
        summary=read(run/'summary.json')
        with np.load(export,allow_pickle=False) as z:stored=json.loads(str(z['summary_json']))
        return all(s.get('run_status')=='RUN_COMPLETED' and abs(s.get('time_ps',-1)-t)<1e-8 for s in (summary,stored))
    except (OSError,ValueError,KeyError,EOFError):return False


def execute(job, record, notify=lambda:None):
    compute_imports()
    config = resolve(job)
    name = job['name']; target = config['time']['final']
    if job.get('resume_from'):
        saved.fork_run(job['resume_from'], name)
    time_ps = checkpoint(name, config)
    if time_ps > target + 1e-8:
        raise ValueError(f'{name}: checkpoint {time_ps:g} ps is beyond target {target:g}; increase final_ps')
    record.pop('independent_scientific_review',None)
    record.update(name=name, target_ps=target, start_ps=time_ps,
                  config_signature=signature(config), action='REUSE')
    if time_ps < target - 1e-8:
        retained_fields(saved.RUNS/name, config, until=float(job.get('retain_ps', 1200.)))
        record.update(action='INTEGRATE',phase='INTEGRATING');notify()
        result = run_report3_pilot(config, name)
        summary = result['summary']
    elif not artifacts_ready(name, time_ps):
        # fork_run intentionally copies raw data only. Rebuild exports/metadata
        # from those exact fields, without performing even one integration step.
        record.update(action='REBUILD_EXPORTS',phase='FINALIZING');notify()
        summary = run_report3_pilot(config, name, max_steps=0)['summary']
    else:
        summary = json.loads((saved.RUNS/name/'summary.json').read_text())
    record.update(end_ps=summary['time_ps'], run_status=summary['run_status'])
    if summary['run_status'] != 'RUN_COMPLETED':
        record['status'] = 'PAUSED'
        return
    record['phase']='REVIEW';notify();review_start=time.monotonic()
    record['review'] = review_saved(name, target)
    record['numerical_checks'] = summary.get('numerical_checks')
    if job.get('compare_to'):
        comparison=saved.compare_late(name,job['compare_to'])
        record['comparison']=comparison
        pop=max(comparison.get('max_cycle_aligned_population_errors',[1.]))
        density=max(comparison['max_cycle_aligned_map_errors'][:2])
        stokes=max(comparison.get('max_cycle_aligned_normalized_stokes_RMS',[1.]))
        reference_review=review_saved(job['compare_to'],target)
        record['reference_review']=reference_review
        established=all(v['status'] in ('STATIONARY','PERIODIC') for v in (record['review'],reference_review))
        record['convergence']={'status':'PASS' if established and max(pop,density,stokes)<.01 else 'NEEDS_REVIEW',
            'population_relative_max':float(pop),'density_relative_L2_max':float(density),
            'normalized_stokes_weighted_RMS_max':float(stokes),'tolerance':.01,
            'scope':'13 late samples at most; one shared cycle shift, full domain on common grid'}
    record['status'] = 'COMPLETED'  # Execution status; scientific status is in review.
    record.update(finished_utc=now(),review_wall_s=time.monotonic()-review_start,phase='DONE')


def run(path,note=''):
    plan=load_plan(path);STATE.parent.mkdir(parents=True,exist_ok=True)
    with STATE.with_suffix('.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        report=read(STATE,{'started_utc':now(),'experiments':{},'attempts':[]})
        if not pending_jobs(plan,report):print('Nothing pending.');return
        report.setdefault('attempts',[]).append({'started_utc':now(),'pid':os.getpid(),'note':note,'previous_status':report.get('status'),'previous_error':report.get('error')})
        report.update(status='WAITING_FOR_GPU',pid=os.getpid(),plan=str(path),note=note)
        report.pop('error',None);report.pop('finished_utc',None)
        pause_file=STATE.with_suffix('.pause');pause_file.unlink(missing_ok=True)
        def save():write_json(STATE,report)
        def request_pause(*_):pause_file.write_text(now());raise KeyboardInterrupt
        old_int=signal.signal(signal.SIGINT,request_pause)
        old_term=signal.signal(signal.SIGTERM,lambda *_:os.kill(os.getpid(),signal.SIGINT))
        save()
        try:
            with (RUNS/'pilots.lock').open('a') as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX)
                if pause_file.exists():raise KeyboardInterrupt
                compute_imports()
                import ctypes
                ctypes.CDLL('libze_loader.so');report['status']='PREFLIGHT';save()
                report['preflight']=saved.short_checks();print('PREFLIGHT PASS',now(),note,flush=True)
                while True:
                    plan=load_plan(path)
                    for job in plan['experiments']:
                        old=report['experiments'].get(job['id'],{})
                        expected=fingerprint(job)
                        if old.get('definition_sha256',expected)!=expected:raise ValueError('Physics changed; use a new id: '+job['id'])
                    pending=pending_jobs(plan,report)
                    if not pending:break
                    if pause_file.exists():raise KeyboardInterrupt
                    done={k for k,v in report['experiments'].items() if v.get('status')=='COMPLETED'}
                    ready=[j for j in pending if set(j.get('depends_on',[]))<=done]
                    if not ready:raise ValueError('Pending jobs blocked by disabled/unfinished dependencies')
                    job=min(ready,key=lambda j:(j.get('priority',50),plan['experiments'].index(j)))
                    record=report['experiments'].setdefault(job['id'],{})
                    record.update(status='RUNNING',name=job['name'],started_utc=now(),definition_sha256=fingerprint(job),note=job.get('note',job.get('purpose','')))
                    record['log']=str(Path('local')/('experiment_'+job['id']+'.log'))
                    report.update(status='RUNNING',active_experiment=job['id']);save()
                    start=time.monotonic();print('START',job['id'],record['note'],now(),flush=True)
                    with (ROOT/record['log']).open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                        try:execute(job,record,save)
                        except BaseException:traceback.print_exc();raise
                    record['wall_s']=time.monotonic()-start;save()
                    if record['status']=='PAUSED':raise KeyboardInterrupt
                    print('DONE',job['id'],record['review']['status'],now(),flush=True)
                report.update(status='COMPLETED',finished_utc=now());report.pop('active_experiment',None);save()
        except BaseException as error:
            paused=isinstance(error,KeyboardInterrupt)
            report.update(status='PAUSED' if paused else 'FAILED',error=None if paused else repr(error),finished_utc=now())
            if report.get('active_experiment'):report['experiments'][report['active_experiment']]['status']=report['status']
            save()
            if not paused:raise
        finally:
            signal.signal(signal.SIGINT,old_int);signal.signal(signal.SIGTERM,old_term)
    print('QUEUE',report['status'],now(),flush=True)

def launch(path,note):
    plan=load_plan(path);state=read(STATE,{})
    if alive(state):print('Already running, PID',state['pid']);return
    if not pending_jobs(plan,state):print('Nothing pending; no GPU work started.');return
    log=ROOT/'local/experiment_queue.log'
    with log.open('a') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--plan',str(path),'--state',str(STATE),'--note',note],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print('Started PID',child.pid,'| log:',log)

def pause(after_current=False,wait=10):
    state=read(STATE,{})
    if not alive(state):print('No active queue:',state.get('status','NOT_STARTED'));return
    STATE.with_suffix('.pause').write_text(now())
    if not after_current:os.kill(state['pid'],signal.SIGINT)
    if after_current:print('Will pause after the current experiment.');return
    end=time.monotonic()+wait
    while alive(state) and time.monotonic()<end:
        latest=read(STATE,{})
        if latest.get('status') not in ('RUNNING','PREFLIGHT','WAITING_FOR_GPU'):break
        time.sleep(.1)
    latest=read(STATE,{})
    if latest.get('status') in ('RUNNING','PREFLIGHT','WAITING_FOR_GPU'):
        print('PAUSE_REQUESTED; still saving/finalizing. Use status to confirm.')
    else:print(latest.get('status','PAUSED'),'| use resume to continue')

def edit_plan(args,path):
    import yaml
    plan=load_plan(path,validate=False);jobs=plan['experiments']
    if args.command=='add':
        if not args.item or not args.source or args.until is None:raise ValueError('add needs ID, --source and --until')
        job={'id':args.item,'name':args.item,'source':args.source,'final_ps':args.until,'priority':args.priority}
        if args.note:job['note']=args.note
        if args.after:job['depends_on']=args.after
        if args.resume_from:job['resume_from']=args.resume_from
        for assignment in args.set:
            key,raw=assignment.split('=',1);patch={};node=patch;keys=key.split('.')
            for k in keys[:-1]:node=node.setdefault(k,{})
            node[keys[-1]]=yaml.safe_load(raw);merge(job.setdefault('parameters',{}),patch)
        jobs.append(job)
    else:
        job=next((j for j in jobs if j['id']==args.item),None)
        if job is None:raise ValueError('Unknown experiment: '+str(args.item))
        if args.command=='extend':
            if args.until is None or args.until<=job['final_ps']:raise ValueError('extend needs a larger --until')
            job['final_ps']=args.until
        elif args.command=='note':job['note']=args.text or args.note
        else:job['enabled']=args.command=='enable'
    # Validate a temporary plan before replacing the user's plan atomically.
    tmp=path.with_suffix('.checking.yaml');tmp.write_text(yaml.safe_dump(plan,sort_keys=False,allow_unicode=True))
    try:load_plan(tmp);tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)
    if args.command=='note' or args.note:
        with (ROOT/'local/experiment_queue.log').open('a') as log:log.write(now()+' NOTE '+job['id']+' '+job.get('note','')+'\n')
    print('Updated',job['id'],'| no calculation started')
    if job.get('note'):print('Note:',job['note'])

def eta(plan,state):
    samples=[]
    for path in DATA.glob('*.json'):
        item=read(path,{})
        if not isinstance(item,dict) or 'config' not in item or 'summary' not in item:continue
        c=item['config'];s=item['summary'];duration=s.get('time_ps',0);wall=s.get('elapsed_wall_s',0)
        if s.get('run_status')!='RUN_COMPLETED' or duration<1000 or wall<=0:continue
        samples.append((c,wall/duration))
    low=high=0.;unknown=0
    for job in pending_jobs(plan,state):
        current=progress(job).get('time_ps',0.);remaining=max(0.,job['final_ps']-current)
        if not remaining:unknown+=1;continue  # Saved physics may still need CPU finalization.
        c=resolve(job);precision=c.get('execution',{}).get('precision','complex128');near=[]
        for ref,rate in samples:
            if ref.get('execution',{}).get('precision','complex128')!=precision:continue
            n=c['space']['N'];nr=ref['space']['N'];dt=c['time']['dt'];dtr=ref['time']['dt']
            distance=abs(math.log2(n/nr))+.5*abs(math.log2(dt/dtr))+.2*(c['pump']['kind']!=ref['pump']['kind'])+.2*(bool(c['physical'].get('beta',0))!=bool(ref['physical'].get('beta',0)))
            factor=(n/nr)**2*math.log2(n)/math.log2(nr)*dtr/dt;near.append((distance,rate*factor))
        if not near:unknown+=1;continue
        best=min(v[0] for v in near);rates=[v for score,v in near if score<=best+.05];rate=statistics.median(rates)
        low+=remaining*rate*(.8 if best==0 else .65);high+=remaining*rate*(1.25 if best==0 else 1.6)
    if unknown and high==0:return 'Not enough data; finalization/unknown cases: '+str(unknown)
    label=f'{low/3600:.1f}–{high/3600:.1f} h'
    return label+(' + finalization/unknown cases: '+str(unknown) if unknown else '')

def status(path):
    plan=load_plan(path,validate=False);state=read(STATE,{});current=state.get('status','NOT_STARTED')
    if current in ('RUNNING','PREFLIGHT','WAITING_FOR_GPU') and not alive(state):current='STOPPED_WITHOUT_FINAL_STATUS'
    print('Queue:',current,'| PID:',state.get('pid','—'))
    if state.get('note'):print('Launch note:',state['note'])
    for job in sorted(plan['experiments'],key=lambda j:j.get('priority',50)):
        record=state.get('experiments',{}).get(job['id'],{});p=progress(job);t=p.get('time_ps',record.get('end_ps',0))
        scientific=record.get('independent_scientific_review',record.get('review',{})).get('status','')
        label=record.get('status','PENDING') if job.get('enabled',True) else 'DISABLED'
        print(f"{job['id']:25} {label:10} {t:7g}/{job['final_ps']:g} ps  {scientific}")
    print('ETA (measured runs, approximate):',eta(plan,state))
    if state.get('error'):print('Last error:',state['error'])

def main():
    global STATE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','run','resume','status','eta','pause','stop','check','add','extend','note','disable','enable'),nargs='?',default='status')
    parser.add_argument('item',nargs='?');parser.add_argument('text',nargs='?')
    parser.add_argument('--plan',type=Path,default=DEFAULT_PLAN);parser.add_argument('--state',type=Path,default=STATE)
    parser.add_argument('--note',default='');parser.add_argument('--source');parser.add_argument('--until',type=float)
    parser.add_argument('--set',action='append',default=[]);parser.add_argument('--priority',type=int,default=50)
    parser.add_argument('--after',action='append',default=[]);parser.add_argument('--resume-from');parser.add_argument('--after-current',action='store_true')
    args=parser.parse_args();path=args.plan.resolve();STATE=args.state.resolve()
    if args.command=='run':run(path,args.note)
    elif args.command in ('start','resume'):launch(path,args.note)
    elif args.command in ('pause','stop'):pause(args.after_current)
    elif args.command=='status':status(path)
    elif args.command=='eta':print('Approximate remaining time:',eta(load_plan(path,validate=False),read(STATE,{})))
    elif args.command=='check':print('Plan valid:',len(load_plan(path)['experiments']),'experiments; checkpoint parameters match')
    else:edit_plan(args,path)

if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as error:raise SystemExit(str(error))
