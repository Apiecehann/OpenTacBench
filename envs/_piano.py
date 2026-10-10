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


MELODY = (64, 62, 60, 62, 64)  # E4 D4 C4 D4 E4; five adjacent notes.


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
