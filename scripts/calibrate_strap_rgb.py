"""Extend the original image-only force regressor using separate native probes.

This fits force-regression coefficients, not the sensor's optical calibration.
The public predictor and its raw RGB/marker feature contract are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_trace(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line]
    rows=[r for r in rows if 'image_features' in r and not r.get('image_error')]
    if not rows:raise ValueError('Trace has no valid image-derived features: '+str(path))
    x=np.asarray([r['image_features'] for r in rows],float)
    y=np.asarray([r['tension_N'] for r in rows],float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite calibration data')
    return np.concatenate([x,x*x],axis=1),y


def fit_extension(prior,x,y,alpha):
    if not np.isfinite(alpha) or alpha<=0:raise ValueError('Regularization must be positive')
    weights=np.asarray(prior['weights'])
    if x.shape[1:]!=weights.shape:raise ValueError('Original image feature contract differs')
    mean=x.mean(axis=0)
    scale=np.maximum(x.std(axis=0),1e-6)
    standardized=(x-mean)/scale
    residual=y-x@weights-prior['bias']
    offset=float(residual.mean())
    delta=np.linalg.solve(standardized.T@standardized+alpha*np.eye(x.shape[1]),
                          standardized.T@(residual-offset))/scale
    return (weights+delta).tolist(),float(prior['bias']+offset-mean@delta)


def metrics(model,x,y):
    error=x@np.asarray(model['weights'])+model['bias']-y
    return dict(samples=len(y),rmse_N=float(np.sqrt(np.mean(error**2))),
        p95_absolute_error_N=float(np.percentile(np.abs(error),95)),
        maximum_absolute_error_N=float(np.abs(error).max()),bias_N=float(error.mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--train',type=Path,nargs='+',required=True)
    parser.add_argument('--validate',type=Path,nargs='+',default=[])
    parser.add_argument('--alpha',type=float,default=10.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve()==args.prior.resolve():parser.error('Preserve the original force regressor')
    prior=json.loads(args.prior.read_text())
    training=[read_trace(p) for p in args.train]
    x=np.concatenate([v[0] for v in training]);y=np.concatenate([v[1] for v in training])
    # Do not carry the prior model's old PASS/selection statistics onto newly
    # fitted coefficients. Its own file remains unchanged and hash-addressed.
    model={k:prior[k] for k in ('target','feature_extractor','feature_transform',
           'physical_contract','input','excludes') if k in prior}
    model['weights'],model['bias']=fit_extension(prior,x,y,args.alpha)
    model['status']='Extended force regression fit; native image-only closed-loop validation required'
    model['source_calibration_sha256']=hashlib.sha256(args.prior.read_bytes()).hexdigest()
    model['development_samples']=len(y)
    model['extension']=dict(method='standardized ridge residual against the original frozen regressor',
        alpha=args.alpha,optical_calibration_changed=False,training=metrics(model,x,y),
        force_coverage_N=[float(y.min()),float(y.max())],
        provenance=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),samples=len(values[1]))
                    for p,values in zip(args.train,training)],
        validation=[dict(path=str(p),**metrics(model,*read_trace(p))) for p in args.validate])
    if args.output.exists():
        previous=json.loads(args.output.read_text())
        seen={r['path'] for r in previous.get('extension',{}).get('provenance',[])}
        unseen=[(p,values) for p,values in zip(args.train,training) if str(p) not in seen]
        if unseen:
            model['extension']['previous_model_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
            model['extension']['previous_unseen_probe_errors']=[
                dict(path=str(p),**metrics(previous,*values)) for p,values in unseen]
    for record,path in zip(model['extension']['provenance'],args.train):
        metadata_path=path.parents[2]/'metadata.json'
        if metadata_path.exists():
            meta=json.loads(metadata_path.read_text()).get(path.parent.name,{})
            parameters=meta.get('strap_parameters',{})
            record['physics_seed']=parameters.get('physics_seed')
            record['band_modulus_mpa']=parameters.get('modulus_mpa')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(model,indent=2))
    print(json.dumps(model['extension'],indent=2))


if __name__=='__main__':
    main()
