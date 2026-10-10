"""Full 88-note keyboard layout (metres, MIDI A0 through C8).

Key dimensions follow the public RoboPianist piano model:
https://github.com/google-research/robopianist/tree/main/robopianist/models/piano
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PianoKey:
    midi: int
    black: bool
    position: tuple
    size: tuple

    @property
    def strike_position(self):
        x, y, z = self.position
        return (x - self.size[0] * .30, y, z + self.size[2])


def keyboard_layout():
    white_count = 0
    keys = []
    for midi in range(21, 109):
        black = midi % 12 in (1, 3, 6, 8, 10)
        if black:
            y = (white_count - 26) * .0235
            size = (.090, .010, .018)
            position = (.545, y, .067)
        else:
            y = (white_count - 25.5) * .0235
            size = (.150, .0225, .0225)
            position = (.515, y, .050)
            white_count += 1
        keys.append(PianoKey(midi, black, position, size))
    return tuple(keys)


MELODY = (60, 60, 67, 67, 69)  # Twinkle, Twinkle, Little Star: C4 C4 G4 G4 A4.


def press_delta(force_history, target, tolerance=.45):
    """Position servo in metres; reject a one-tick landing impulse.

    The three 120 Hz samples only filter the controller. Physical scoring and
    overload checks still consume every raw contact force without filtering.
    """
    import numpy as np
    force = float(np.median(force_history))
    delta = (-.000075 if force<target-tolerance
             else float(np.clip((force-target)*.00004,-.000075,.00010)))
    return delta, force


def striker_mesh():
    """Rounded tapping stylus, in metres; identical FEM and visible surface.

    Flat opposing grip faces remain 26 mm apart. The 7.8 mm compliant tip
    and 108 mm overall length keep the tool suitable for individual keys.
    Rings resolve the tip bevel, shaft, tapered ferrule and capped handle.
    """
    import numpy as np

    # z, half-width, corner radius. The lower 9 mm is the compliant tip.
    profile = [
        (0., .0035, .0007), (.0008, .0039, .0011),
        (.0045, .0039, .0011), (.009, .0039, .0011),
        (.010, .0038, .0012), (.018, .0038, .0012),
        (.027, .0038, .0012), (.036, .0038, .0012),
        (.045, .0038, .0012), (.054, .0038, .0012),
        (.056, .0043, .0015), (.060, .009, .0025),
        (.063, .0125, .003), (.064, .013, .0035),
        (.066, .013, .0035), (.067, .0128, .0035),
        (.071, .013, .0035), (.077, .013, .0035),
        (.084, .013, .0035), (.091, .013, .0035),
        (.098, .013, .0035), (.102, .013, .0035),
        (.103, .013, .0035), (.105, .013, .0035),
        (.107, .0124, .0033), (.108, .0114, .003),
    ]
    points = []
    for z, half, radius in profile:
        points.append((0., 0., z))
        for quadrant in range(4):
            a = quadrant*np.pi/2
            cx, cy = (half-radius)*np.array([
                1 if quadrant in (0,3) else -1,
                1 if quadrant in (0,1) else -1])
            for angle in np.linspace(a, a+np.pi/2, 7):
                points.append((cx+radius*np.cos(angle), cy+radius*np.sin(angle), z))
    points = np.ascontiguousarray(points, dtype=float)
    stride = 29
    tets = []
    for layer in range(len(profile)-1):
        base, top = layer*stride, (layer+1)*stride
        for i in range(28):
            j = (i+1)%28
            a,b,c,d,e,f = base,base+i+1,base+j+1,top,top+i+1,top+j+1
            tets.extend(((a,b,c,f),(a,b,e,f),(a,d,e,f)))
    tets = np.asarray(tets, dtype=np.int32)
    det = np.linalg.det(np.stack([points[tets[:,i]]-points[tets[:,0]] for i in (1,2,3)],axis=-1))
    flip = det<0
    tets[flip,1],tets[flip,2] = tets[flip,2].copy(),tets[flip,1].copy()
    return points, tets


def finish_striker(task):
    """Shade the actual deforming surface: soft tip, steel, brass and grip."""
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdGeom, UsdShade
    from ._force_task_utils import _material

    stage = omni.usd.get_context().get_stage()
    meshes = [prim for env in task.scene.env_prim_paths
              for prim in Usd.PrimRange(stage.GetPrimAtPath(env+'/striker'))
              if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError('Piano striker has no deforming render surface')
    for prim in meshes:
        path = str(prim.GetPath())
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(),float)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
        z = vertices[faces].mean(axis=1)[:,2]-vertices[:,2].min()
        groups = np.select([z<.009, z<.056, z<.067, z<.102],
                           ['SoftTip','Steel','Brass','Grip'], default='Brass')
        looks = {
            'SoftTip': ((.87,.82,.70),.88,0.),
            'Steel': ((.36,.40,.43),.28,.8),
            'Brass': ((.54,.33,.10),.28,.78),
            'Grip': ((.025,.09,.085),.62,0.),
        }
        for name,(color,roughness,metallic) in looks.items():
            material = _material(stage,str(path)+'/Looks/'+name,color,roughness,metallic)
            subset = UsdGeom.Subset.Define(stage,str(path)+'/'+name)
            subset.CreateElementTypeAttr('face')
            subset.CreateFamilyNameAttr('materialBind')
            subset.CreateIndicesAttr(np.flatnonzero(groups==name).tolist())
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)


def build_cabinet(task):
    """Presentation shell; the key bed and cabinet also have collision actors."""
    import omni.usd
    from pxr import UsdGeom
    from ._force_task_utils import _material, _rounded_box

    stage = omni.usd.get_context().get_stage()
    for env in task.scene.env_prim_paths:
        root = env + '/piano_cabinet'
        UsdGeom.Xform.Define(stage, root)
        ebony = _material(stage, root+'/Looks/Ebony', (.012, .015, .020), .23)
        trim = _material(stage, root+'/Looks/Brass', (.48, .30, .10), .25, .8)
        felt = _material(stage, root+'/Looks/Felt', (.28, .018, .025), .94)
        wood = _material(stage, root+'/Looks/Walnut', (.080, .036, .018), .38)
        def box(name, center, size, material=ebony, radius=.004):
            return _rounded_box(stage, root+'/'+name, center, size, radius, material)
        # Match the collision bed's 40 mm top, leaving the keys their physical
        # travel instead of letting a thick visual shell cover a depressed key.
        box('keybed', (.539,0,.020), (.252,1.315,.040))
        box('front_lip', (.423,0,.048), (.020,1.315,.027))
        for side in (-1,1):
            box('cheek_'+str(side).replace('-','n'), (.534,side*.636,.073), (.252,.043,.098))
        box('fallboard', (.625,0,.149), (.056,1.31,.18))
        box('upper_cabinet', (.690,0,.260), (.145,1.315,.400))
        box('top_lid', (.680,0,.467), (.198,1.345,.021))
        box('lid_seam', (.578,0,.456), (.0015,1.31,.002), trim, .0002)
        box('fallboard_trim', (.594,0,.090), (.003,1.255,.003), trim, .0004)
        box('red_key_felt', (.589,0,.078), (.006,1.225,.006), felt, .0007)
        box('lower_panel', (.614,0,.082), (.026,1.215,.12), wood)
        # Raised music desk, with a ledge and fine perimeter moulding.
        box('music_desk', (.591,0,.323), (.014,.43,.19))
        box('music_ledge', (.569,0,.229), (.052,.46,.012))
        for side in (-1,1):
            box('desk_side_'+str(side).replace('-','n'), (.581,side*.216,.323), (.006,.006,.19), trim, .001)
        box('desk_top', (.581,0,.417), (.006,.437,.006), trim, .001)
        # Small brass nameplate; no note labels reveal the hidden articulation.
        box('nameplate', (.594,0,.190), (.002,.100,.017), trim, .001)
        box('pedal_block', (.404,0,.013), (.042,.17,.021))
        for i,y in enumerate((-.048,0.,.048)):
            box('pedal_'+str(i), (.369,y,.012), (.060,.020,.012), trim, .008)
