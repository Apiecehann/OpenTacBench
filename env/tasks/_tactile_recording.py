"""Record synchronous single-index tactile arrays and a labelled video panel."""
import json
import numpy as np


class TactileRecorder:
    def __init__(self,path,task):
        import h5py
        self.file=h5py.File(path,'w')
        self.task=task
        self.count=0
        self.datasets={}
        sensor=task.tactile
        self.file.attrs['metadata']=json.dumps(sensor.metadata())
        self.file.attrs['physics_dt_s']=task.cfg.physics_dt
        self.file.attrs['control_frequency_hz']=1/(task.cfg.physics_dt*task.cfg.action_repeat)
        self.file.create_dataset('mesh/rest_sensor_m',data=sensor.rest_sensor)
        self.file.create_dataset('mesh/tetrahedra',data=sensor.tets)
        self.file.create_dataset('mesh/front_indices',data=sensor.front)

    def append(self,observation):
        arrays={'physics_step':np.asarray(self.task.step_count,dtype=np.int64),
                'time_s':np.asarray(self.task.step_count*self.task.cfg.physics_dt),
                'state':observation['state']}
        arrays.update({'tactile/index/'+k:v for k,v in observation['tactile']['index'].items()})
        for name,value in arrays.items():
            value=np.asarray(value)
            if name not in self.datasets:
                self.datasets[name]=self.file.create_dataset(name,shape=(0,*value.shape),
                    maxshape=(None,*value.shape),chunks=(1,*value.shape),dtype=value.dtype,
                    compression='gzip',compression_opts=2,shuffle=True)
                if name.endswith(('depth','deformation','marker_displacement')):
                    self.datasets[name].attrs['units']='metre'
            dataset=self.datasets[name]
            dataset.resize(self.count+1,axis=0)
            dataset[self.count]=value
        self.count+=1
        if self.count % 30 == 0:
            self.file.flush()

    def close(self):
        self.file.attrs['metadata']=json.dumps(self.task.tactile.metadata())
        self.file.close()


def tactile_panel(scene,tactile):
    import cv2
    panel=np.zeros((640,320,3),dtype=np.uint8)
    views=[('INDEX - GelSight RGB',tactile['rgb'],240),
        ('Depth [19.9, 20.0] mm',cv2.applyColorMap(np.rint(np.clip(
            (tactile['depth']-.0199)/.0001,0,1)*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)[:,:,::-1],200),
        ('Deformation [0, 0.1] mm',cv2.applyColorMap(np.rint(np.clip(
            np.linalg.norm(tactile['deformation'],axis=-1)/.0001,0,1)*255).astype(np.uint8),
            cv2.COLORMAP_TURBO)[:,:,::-1],200)]
    top=0
    for title,value,height in views:
        panel[top:top+height]=cv2.resize(value,(320,height))
        cv2.rectangle(panel,(0,top),(320,top+27),(18,22,28),-1)
        cv2.putText(panel,title,(10,top+19),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,244,248),1,cv2.LINE_AA)
        top+=height
    return np.concatenate([scene,panel],axis=1)
