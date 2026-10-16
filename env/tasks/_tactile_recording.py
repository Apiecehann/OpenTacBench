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

    def append(self,observation):
        arrays={'physics_step':np.asarray(self.task.step_count,dtype=np.int64),
                'time_s':np.asarray(self.task.step_count*self.task.cfg.physics_dt),
                'state':observation['state']}
        arrays['tactile/index/rgb']=observation['tactile']['index']['rgb']
        for name,value in arrays.items():
            value=np.asarray(value)
            if name not in self.datasets:
                self.datasets[name]=self.file.create_dataset(name,shape=(0,*value.shape),
                    maxshape=(None,*value.shape),chunks=(1,*value.shape),dtype=value.dtype,
                    compression='gzip',compression_opts=2,shuffle=True)
                if name.endswith('/rgb'):
                    self.datasets[name].attrs['colour_order']='RGB'
            dataset=self.datasets[name]
            dataset.resize(self.count+1,axis=0)
            dataset[self.count]=value
        self.count+=1
        if self.count % 30 == 0:
            self.file.flush()

    def close(self):
        self.file.attrs['metadata']=json.dumps(self.task.tactile.metadata())
        self.file.close()


def tactile_panel(scene,tactile,title='GelSight Mini - index'):
    """Show the complete RGB frame at its aspect ratio, below a separate title."""
    import cv2
    panel=np.full((scene.shape[0],640,3),(18,22,28),dtype=np.uint8)
    cv2.putText(panel,title,(20,34),
        cv2.FONT_HERSHEY_SIMPLEX,.7,(240,244,248),1,cv2.LINE_AA)
    rgb=tactile['rgb']
    height,width=rgb.shape[:2]
    scale=min(panel.shape[1]/width,(panel.shape[0]-56)/height)
    shown=cv2.resize(rgb,(round(width*scale),round(height*scale))) if scale!=1 else rgb
    y=56+(panel.shape[0]-56-shown.shape[0])//2
    x=(panel.shape[1]-shown.shape[1])//2
    panel[y:y+shown.shape[0],x:x+shown.shape[1]]=shown
    return np.concatenate([scene,panel],axis=1)
