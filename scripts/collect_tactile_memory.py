"""Collect and upload validated episodes, using one fresh native process per seed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.upload_tactile_memory import upload_episode

REPO=Path(__file__).resolve().parents[1]
TASKS=('tension_strap','play_piano','arm_shadow_piano')


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def strap_instruction(seed,maximum):
    """Include both endpoints, both orders, and varied interior forces/times."""
    # Keep prompt forces independent of the hidden modulus stream, which uses
    # default_rng(physics_seed) inside the original physical task.
    rng=np.random.default_rng(np.random.SeedSequence([seed,0x53545250]))
    if seed%10==0:forces=(0.,maximum)
    elif seed%10==1:forces=(maximum,0.)
    else:forces=tuple(np.round(rng.uniform(0,maximum,2),2))
    durations=np.round(rng.uniform(.75,2.,size=2),2)
    return [float(forces[0]),float(durations[0]),float(forces[1]),float(durations[1])]


def fingerprint(task):
    files=['scripts/collect_tactile_memory.py','scripts/collect_data.py',
           'scripts/upload_tactile_memory.py']
    if task=='tension_strap':
        files+=['envs/tension_strap.py','envs/_strap_sequence.py','envs/_force_task_utils.py',
                'task_config/tension_strap.yml',
                'scripts/validate_final_task_episode.py']
        config=yaml.safe_load((REPO/'task_config/tension_strap.yml').read_text())
        files.append(config['task_parameters']['calibration'])
    elif task=='play_piano':
        files+=['envs/play_piano.py','envs/_memory_task.py','envs/_base_task.py',
                'envs/_tactile_memory.py','envs/_piano_sequence.py','task_config/play_piano.yml',
                'envs/_force_task_utils.py','envs/_piano.py','envs/_piano_audio.py',
                'envs/utils/data.py','scripts/validate_piano_episode.py',
                'third_party/TacEx/source/tacex/tacex/gelsight_sensor.py',
                'third_party/TacEx/source/tacex/tacex/simulation_approaches/fem_based/mani_skill_sim.py',
                'third_party/TacEx/source/tacex/tacex/simulation_approaches/fem_based/mani_skill_sim_cfg.py',
                'third_party/TacEx/source/tacex/tacex/simulation_approaches/fem_based/sim/tactile_sensor_sapienipc_modified.py']
    else:
        files+=['env/tasks/arm_shadow_piano.py','env/tasks/shadow_piano.py',
                'env/tasks/single_finger_piano.py','env/tasks/_digit_tactile.py','env/tasks/_digit_renderer.py',
                'env/tasks/_single_key_score.py','env/tasks/_tactile_recording.py',
                'env/tasks/run_single_finger_piano.py','envs/_piano_sequence.py',
                'envs/_piano.py','scripts/validate_piano_episode.py']
    if task!='arm_shadow_piano':
        files+=['envs/_base_task.py','envs/robot/robot.py','envs/robot/robot_cfg.py',
                'envs/sensors/tactile.py','envs/sensors/camera.py']
    files=sorted(set(files))
    return {p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in files}


def native_command(args,seed,output):
    env=os.environ.copy()
    env.update(PYTHONEXE=str(args.native_python),PYTHONPATH=str(REPO),PYTHONUNBUFFERED='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=str(args.gpu))
    torch_lib=args.isaac_root/'exts/omni.isaac.ml_archive/pip_prebundle/torch/lib'
    env['LD_LIBRARY_PATH']=str(torch_lib)+':'+env.get('LD_LIBRARY_PATH','')
    for name in ('OPENTACBENCH_FORCE_WORKER','OPENTACBENCH_FORCE_OUTPUT','OPENTACBENCH_FORCE_RESULT'):
        env.pop(name,None)
    command=[str(args.isaac_root/'python.sh')]
    if args.task=='arm_shadow_piano':
        command+=['-m','env.tasks.run_single_finger_piano','--tactile','--record-actions',
            '--judge-dynamics','--song','random','--notes','6','--seed',str(seed),
            '--image-width','320','--output',str(output)]
    else:
        config=yaml.safe_load((REPO/'task_config'/f'{args.task}.yml').read_text())
        config.update(save_dir_exact=str(output),start_seed=seed,max_seed=seed,episode_num=1)
        if args.task=='tension_strap':
            maximum=float(config['task_parameters']['overload_N'])
            config['task_parameters']['force_time_sequence']=strap_instruction(seed,maximum)
            env.update(OPENTACBENCH_FORCE_WORKER='1',OPENTACBENCH_FORCE_OUTPUT=str(output),
                OPENTACBENCH_FORCE_RESULT=str(output/'result.json'))
        path=output/'config.yml'
        path.write_text(yaml.safe_dump(config))
        command+=['scripts/collect_data.py',args.task,str(path),'--episode_num','1',
                  '--start_seed',str(seed),'--max_seed',str(seed)]
    command+=['--headless','--enable_cameras','--rendering_mode','performance']
    return command,env


def collect_one(args,seed,output):
    output.mkdir(parents=True,exist_ok=False)
    command,env=native_command(args,seed,output)
    started=time.perf_counter()
    with (output/'log.txt').open('w') as log:
        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
        print(f'NATIVE_PID seed={seed} pid={child.pid}',flush=True)
        try:
            code=child.wait(timeout=args.episode_timeout)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            raise
    result_path=output/'result.json'
    if args.task=='tension_strap':
        result=json.loads(result_path.read_text()) if result_path.exists() else dict(result='error',error='worker produced no result')
    else:
        from scripts.validate_piano_episode import validate_piano
        if args.task=='arm_shadow_piano':
            result=json.loads(result_path.read_text()) if result_path.exists() else dict(success=False,error='worker produced no result')
        else:
            meta_path=output/'metadata.json'
            meta=json.loads(meta_path.read_text()).get(str(seed),{}) if meta_path.exists() else {}
            result=dict(result=meta.get('result','error'),piano_result=meta.get('piano_result',{}))
        if result.get('success') or result.get('result')=='success':
            result['validation']=validate_piano(output,'hand' if args.task=='arm_shadow_piano' else 'tool',
                seed,require_actions=args.task=='arm_shadow_piano',require_rich=args.task=='play_piano')
    result.update(seed=seed,task=args.task,worker_exit_code=code,collection_wall_seconds=time.perf_counter()-started,
                  source_commit=args.source_commit,source_sha256=args.source_sha256,
                  collection_workers=1)
    valid=(code==0 and (result.get('success') is True or result.get('result')=='success')
           and result.get('validation',{}).get('valid') is True)
    result['dataset_accepted']=valid
    if not valid and (result.get('success') or result.get('result')=='success'):
        result.update(success=False,result='error')
    write_json(result_path,result)
    return result


def upload_one(args,path,seed,roundtrip):
    for attempt in range(3):
        try:
            prefix=getattr(args,'path_in_repo',None) or f'{args.task}/episodes'
            return upload_episode(path,args.repo_id,f'{prefix}/{seed:04d}',roundtrip=roundtrip)
        except Exception:
            if attempt==2:raise
            time.sleep(3*(attempt+1))


def run(args):
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'.collection.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    args.source_sha256=fingerprint(args.task)
    prefix=getattr(args,'path_in_repo',None) or f'{args.task}/episodes'
    state_path=args.output/'progress.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if (state['source_sha256']!=args.source_sha256 or state['repo_id']!=args.repo_id
                or state.get('path_in_repo',f'{args.task}/episodes')!=prefix):
            raise ValueError('Resume requires identical task code and upload destination')
    else:
        state=dict(task=args.task,repo_id=args.repo_id,source_sha256=args.source_sha256,
            source_commit=args.source_commit,path_in_repo=prefix,attempts=[],status='running',target=args.count)
    state.update(status='running',target=args.count,workers=1)
    state.pop('error',None)
    accepted=[r for r in state['attempts'] if r.get('accepted')]
    pending=[]
    def save():
        state['collected']=sum(r.get('accepted',False) for r in state['attempts'])
        state['uploaded']=sum(r.get('uploaded',False) for r in state['attempts'])
        write_json(state_path,state)
    save()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for record in accepted:
                if not record.get('uploaded'):
                    pending.append((record,pool.submit(upload_one,args,Path(record['path']),record['seed'],not pending)))
            next_seed=max([r['seed'] for r in state['attempts']],default=args.start_seed-1)+1
            failures=0
            while len(accepted)<args.count and len(state['attempts'])<args.max_attempts:
                if fingerprint(args.task)!=args.source_sha256:
                    raise ValueError('Task code changed during the batch; refusing to mix versions')
                seed=next_seed
                path=args.output/'episodes'/f'{seed:04d}'
                print(f'COLLECT {args.task} seed={seed} accepted={len(accepted)}/{args.count}',flush=True)
                result=collect_one(args,seed,path)
                record=dict(seed=seed,path=str(path),accepted=result['dataset_accepted'],uploaded=False)
                state['attempts'].append(record)
                if record['accepted']:
                    accepted.append(record)
                    pending.append((record,pool.submit(upload_one,args,path,seed,len(accepted)==1)))
                    failures=0
                else:
                    failures+=1
                    record['failure']=(result.get('error') or result.get('validation',{}).get('errors')
                        or result.get('failure') or result.get('piano_result',{}).get('failure')
                        or result.get('terminal_reason') or result.get('result'))
                for r,future in pending:
                    if future.done() and not r.get('uploaded'):
                        r['upload']=future.result()
                        r['uploaded']=True
                save()
                if failures>=3:
                    raise RuntimeError('Three consecutive invalid episodes; inspect native evidence before resuming')
                next_seed+=1
            for record,future in pending:
                if not record.get('uploaded'):
                    record['upload']=future.result()
                    record['uploaded']=True
                    save()
        if len(accepted)<args.count:raise RuntimeError('Attempt budget exhausted before the requested valid count')
        state['status']='complete'
        save()
        print(f'DATASET_COMPLETE {args.task} collected={state["collected"]} uploaded={state["uploaded"]}',flush=True)
    except BaseException as error:
        state.update(status='stopped',error=repr(error))
        save()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--start-seed',type=int,default=0)
    parser.add_argument('--max-attempts',type=int,default=100)
    parser.add_argument('--episode-timeout',type=float,default=1500)
    parser.add_argument('--repo-id',required=True,help='Existing writable ModelScope dataset, OWNER/DATASET')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--path-in-repo',help='Dataset batch prefix; defaults to TASK/episodes')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--isaac-root',type=Path,default=Path(os.environ.get('ISAAC_SIM_ROOT','/isaac-sim')))
    parser.add_argument('--native-python',type=Path,default=REPO/'.venv-tactile/bin/python')
    args=parser.parse_args()
    if args.count<1 or args.max_attempts<args.count or args.start_seed<0 or args.episode_timeout<=0:
        parser.error('Positive episode count/timeout, nonnegative seed and sufficient attempt budget are required')
    args.output=(args.output or REPO/'data/tactile_memory'/args.task/'dataset').resolve()
    # Background shells can inherit SIGINT ignored. Restore interruption so
    # collect_one stops its native process group and saves the batch state.
    signal.signal(signal.SIGINT,signal.default_int_handler)
    signal.signal(signal.SIGTERM,signal.default_int_handler)
    run(args)


if __name__=='__main__':
    main()
