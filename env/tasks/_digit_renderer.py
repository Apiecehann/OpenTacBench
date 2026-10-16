"""Run the original TACTO renderer in its own NumPy/OpenGL environment."""
import json
import sys
from pathlib import Path


def main():
    output=sys.stdout.buffer
    sys.stdout=sys.stderr  # Reserve stdout for the image transport.
    import cv2
    import numpy as np
    from importlib.metadata import distribution
    import importlib.util
    import trimesh
    # Import the original rendering module directly. The package's top-level
    # import also loads its unrelated PyBullet/URDF scene integration.
    package=distribution('tacto')
    spec=importlib.util.spec_from_file_location('digit_tacto_renderer',package.locate_file('tacto/renderer.py'))
    original=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    config=str(package.locate_file('tacto/config_digit.yml'))
    root=Path(sys.argv[1])
    width=int(sys.argv[2]); height=width*3//4
    background=cv2.imread(str(root/'assets/examples/conf/bg_digit_240_320.jpg'))
    if background is None: raise RuntimeError('Original DIGIT reference image is missing')
    renderer=original.Renderer(width,height,background,config)
    case=trimesh.load(root/'assets/meshes/digit.STL')
    gel=renderer._generate_gel_trimesh()
    if not (.015<case.extents.min() and case.extents.max()<.065):
        raise RuntimeError(f'Unexpected DIGIT mesh units: {case.extents}')
    ready=dict(case_vertices=case.vertices.tolist(),case_faces=case.faces.tolist(),
        gel_vertices=gel.vertices.tolist(),gel_faces=gel.faces.tolist(),
        case_size_m=case.extents.tolist(),config_path=config)
    output.write(json.dumps(ready).encode()+b'\n'); output.flush()
    for line in sys.stdin:
        request=json.loads(line)
        if request['command']=='close': break
        if request['command']=='setup':
            for name,size in request['keys'].items():
                renderer.add_object(trimesh.creation.box(extents=size),name)
            output.write(b'{"ready":true}\n'); output.flush()
            continue
        renderer.update_camera_pose(*request['sensor'])
        poses={k:v[:2] for k,v in request['objects'].items()}
        forces={k:v[2] for k,v in request['objects'].items() if v[2]>.01}
        colors,depths=renderer.render(object_poses=poses,normal_forces=forces)
        rgb=np.ascontiguousarray(colors[0],dtype=np.uint8)
        valid=(renderer.depth0[0]>0.)&(depths[0]>0.)
        indentation=float(np.maximum(0.,renderer.depth0[0][valid]-depths[0][valid]).max(initial=0.))
        output.write(json.dumps(dict(bytes=rgb.nbytes,indentation_m=indentation)).encode()+b'\n')
        output.write(rgb.tobytes()); output.flush()
    renderer.r.delete()


if __name__=='__main__': main()
