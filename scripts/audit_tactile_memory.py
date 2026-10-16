"""Audit accepted recordings once, or alongside a native collection queue."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.validate_tactile_recording import audit_recording

TASKS=('tension_strap','arm_shadow_piano','play_piano')


def write_state(path,state):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2))
    temporary.replace(path)


def scan(root,tasks,state,target,save):
    complete=True
    for task in tasks:
        progress=root/task/'dataset/progress.json'
        if not progress.exists():complete=False;continue
        batch=json.loads(progress.read_text())
        complete&=(batch['status']=='complete' and batch['collected']>=target and batch['uploaded']>=target)
        for attempt in batch['attempts']:
            if not attempt.get('accepted'):continue
            folder=Path(attempt['path'])
            path=folder/'tactile.h5' if task=='arm_shadow_piano' else folder/'hdf5'/f"{attempt['seed']}.hdf5"
            key=f"{task}/{attempt['seed']:04d}"
            info=path.stat();signature=[info.st_size,info.st_mtime_ns]
            previous=state['episodes'].get(key)
            if previous and previous.get('file_signature')==signature:
                if not previous['valid']:raise ValueError('An unchanged recording failed audit: '+key)
                continue
            report=audit_recording(path)
            report.update(file_signature=signature,path=str(path.relative_to(root)))
            state['episodes'][key]=report
            save()
            print('RGB_AUDIT',key,'PASS' if report['valid'] else 'FAIL',report['decoded_images'],flush=True)
            if not report['valid']:raise ValueError(key+': '+str(report['errors']))
    state['counts']={task:sum(k.startswith(task+'/') and v['valid'] for k,v in state['episodes'].items()) for task in tasks}
    state['decoded_images']=sum(v['decoded_images'] for v in state['episodes'].values())
    if complete:
        assert all(v>=target for v in state['counts'].values())
    return bool(complete)


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'data/tactile_memory')
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--watch',action='store_true',help='Wait for the requested valid/uploaded count in every selected task')
    args=parser.parse_args()
    if args.count<1:parser.error('--count must be positive')
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.recording_audit.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=root/'recording_audit.json'
    signature=hashlib.sha256(Path(__file__).with_name('validate_tactile_recording.py').read_bytes()).hexdigest()
    state=json.loads(output.read_text()) if output.exists() else {}
    if state.get('audit_source_sha256')!=signature:state={}
    state.update(status='running',audit_source_sha256=signature)
    state.pop('error',None)
    state.setdefault('episodes',{})
    save=lambda:write_state(output,state)
    try:
        while True:
            complete=scan(root,args.tasks,state,args.count,save)
            state['status']='complete' if complete else 'running' if args.watch else 'partial'
            save()
            if complete or not args.watch:
                print('RGB_AUDIT_STATUS',state['status'],state['counts'],state['decoded_images'],flush=True)
                return
            time.sleep(30)
    except BaseException as error:
        state.update(status='stopped',error=repr(error));save()
        raise


if __name__=='__main__':main()
