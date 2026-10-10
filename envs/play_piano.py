"""Play a short phrase on an 88-key piano after remembering a tactile cue."""
import numpy as np
from transforms3d.quaternions import mat2quat

from ._base_task import Pose, configclass
from ._memory_task import MemoryTask, MemoryTaskCfg
from ._tactile_memory import PianoScore, box_mesh
from ._piano import keyboard_layout, build_cabinet, MELODY

TASK_INSTRUCTION = ("Feel the cue: one rib starts short-light, two ribs starts long-firm. "
                    "Remember it, release the cue, and play E4 D4 C4 D4 E4 with alternating articulation.")


def _piano_cameras():
    cameras = MemoryTaskCfg().cameras
    eye, aim = np.array([-.80,-1.35,.95]), np.array([.55,0.,.19])
    z = (eye-aim)/np.linalg.norm(eye-aim)
    x = np.cross([0.,0.,1.],z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    cameras[0].offset.pos = tuple(eye)
    cameras[0].offset.rot = tuple(mat2quat(np.stack([x,y,z],axis=1)))
    cameras[0].offset.convention = 'opengl'
    for camera in cameras:
        camera.width, camera.height = 640, 480
    cameras[0].spawn.focal_length = 2.5
    cameras[0].spawn.horizontal_aperture = 2.4
    return cameras


@configclass
class TaskCfg(MemoryTaskCfg):
    memory_timeout_seconds = 30.0
    step_lim = 1800
    max_save_frames = 1900
    cameras = _piano_cameras()
    video_size = (1520, 480)


class Task(MemoryTask):
    layout = keyboard_layout()
    key_positions = tuple(key.position for key in layout)
    striker_position = (.34,.15,.003)
    striker_grasp_height = .085

    def __init__(self, cfg, **kwargs):
        self.score = PianoScore(0, key_sequence=tuple(m-21 for m in MELODY), key_count=88)
        self.key_forces = np.zeros(88)
        self.striker_released = False
        self._key_vertex_ids = None
        # A compliant probe tip makes force controllable at the 60 Hz action
        # cadence. Resolve deformations more finely than the default 0.4 mm.
        cfg.uipc_sim.newton.velocity_tol = .01
        super().__init__(cfg, **kwargs)

    def create_actors(self):
        self.create_cues()
        self.keybed = self.add_box("keyboard_base",(.24,1.315,.034),(.54,0,.006),
                                   fixed=True,color=(.012,.015,.020))
        self.add_box("piano_back",(.14,1.31,.40),(.69,0,.06),fixed=True,color=(.012,.015,.020))
        self.keys, self.key_tops, self.key_anchors = [], [], []
        for spec in self.layout:
            from uipc import view
            points,tets = box_mesh(spec.size, (10,2,2))
            self.key_tops.append(points[:,2] > spec.size[2] - 1e-6)
            self.key_anchors.append(points[:,0] > spec.size[0]/2 - 1e-6)
            key = self.add_mesh(f"key_{spec.midi}",points,tets,spec.position,
                fixed=True,modulus=100.,color=(.009,.010,.012) if spec.black else (.91,.89,.82))
            # A stiff key on a short elastic rear flexure. This uses the same
            # contact-coupled FEM solver as the tactile pads. libuipc 0.9's
            # revolute-joint/FEM combination fails during scene initialization.
            root = points[tets].mean(axis=1)[:,0] > spec.size[0]*.40
            ratio = .0055
            for attribute in ('mu','lambda'):
                values = view(key.uipc_meshes[0].tetrahedra().find(attribute))
                values.reshape(-1)[root] *= ratio
            # Pin the rear face exactly. Soft positional targets permit root
            # rotation under gravity and let the key settle onto the keybed.
            view(key.uipc_meshes[0].vertices().find('is_fixed')).reshape(-1)[self.key_anchors[-1]] = 1
            self.keys.append(key)
        # A slender tapping tool with a soft lower tip. A fully rigid steel tip
        # produced an impulsive >9 N first contact for a 0.2 mm command.
        points,tets = box_mesh((.026,.026,.108),(2,2,12))
        points[:,:2] *= np.where(points[:,2,None] < .063, .30, 1.)
        self.striker = self.add_mesh("striker",points,tets,self.striker_position,
                                    fixed=True,modulus=50.,color=(.20,.23,.27))
        tip = points[tets].mean(axis=1)[:,2] < .009
        for attribute in ('mu','lambda'):
            values = view(self.striker.uipc_meshes[0].tetrahedra().find(attribute))
            values.reshape(-1)[tip] *= .002
        contacts = self.uipc_sim.scene.contact_tabular()
        key_contact = contacts.create('piano_keys')
        resistance = self.cfg.uipc_sim.contact.default_contact_resistance * 1e9
        contacts.insert(key_contact,key_contact,friction_rate=.3,resistance=resistance,enable=False)
        for key in self.keys:
            for mesh in key.uipc_meshes:
                key_contact.apply_to(mesh)
        build_cabinet(self)

    def _reset_actors(self):
        self.reset_cue()
        self.score = PianoScore(self.cue_kind,self.cfg.sim.dt,
                               key_sequence=tuple(m-21 for m in MELODY), key_count=88)
        self.key_forces = np.zeros(88)
        self.striker_released = False
        self.striker.set_pose(Pose(self.striker_position,[1.,0.,0.,0.]))
        for key,position in zip(self.keys,self.key_positions):
            key.set_pose(Pose(position,[1.,0.,0.,0.]))

    def _release_reset_constraints(self):
        for key in self.keys:
            key.remove_animate(force=True)

    def pre_move(self):
        # Leave clearance above the full-height keyboard at the policy handoff.
        self.monitor = False
        center = self._robot_manager.get_gripper_center_pose().p
        self.move_center(center+[0,0,.08])
        super().pre_move()

    def observe_task(self):
        center = self._robot_manager.get_gripper_center_pose().p
        if not self.striker_released:
            near = np.linalg.norm(center-(np.asarray(self.striker_position)+[0,0,self.striker_grasp_height])) < .02
            if near:
                depth = self._tactile_manager.get_min_depth().detach().cpu().numpy()
                if np.all(depth < 27.9):
                    self.striker.remove_animate(force=True)
                    self.striker_released = True
        if self._key_vertex_ids is None:
            starts = [int(key.geo_slot_list[0].geometry().meta().find('global_vertex_offset').view()[0])
                      for key in self.keys]
            self._key_vertex_ids = np.full(max(s+len(m) for s,m in zip(starts,self.key_tops)), -1, np.int32)
            for i,(start,mask) in enumerate(zip(starts,self.key_tops)):
                self._key_vertex_ids[start+np.flatnonzero(mask)] = i
        # All 88 keys share one contact-gradient read. Reading each actor's full
        # deformed vertex array would synchronize the GPU 88 times per tick.
        indices,gradient = self.uipc_sim.get_contact_gradient()
        valid = (indices >= 0) & (indices < len(self._key_vertex_ids))
        labels = self._key_vertex_ids[indices[valid]]
        loads = gradient[valid,2] / self.cfg.sim.dt**2
        self.key_forces = np.maximum(0.,np.bincount(labels[labels>=0],
            weights=loads[labels>=0],minlength=88))
        previous = self.score.index
        self.score.advance(self.key_forces,memory_ready=self.memory.ready)
        if self.score.index != previous or self.score.failure:
            self.metadata.setdefault('piano_contact_events',[]).append(dict(
                step=self.step_count,previous_note=previous,next_note=self.score.index,
                failure=self.score.failure,forces_N={str(i):float(f) for i,f in enumerate(self.key_forces) if f>.01}))
        self.failure = self.failure or self.score.failure

    def _play_once(self):
        self.metadata['piano_pressure_trace'] = []
        self.metadata['piano_geometry'] = []
        if not self.touch_cue():
            return
        grasp = np.asarray(self.striker_position)+[0,0,self.striker_grasp_height]
        self.move_center(grasp+[0,0,.06])
        self.move_center(grasp)
        self.move(self.atom.close_gripper(0.,depth_threshold=27.4),tag="manipulate",delay=False)
        self.delay(12)
        if not self.striker_released:
            self.failure = "striker_grasp_missing"
            return
        self.move_center(grasp+[0,0,.08])
        for i, note in enumerate(self.score.notes):
            if self.failure or not self.plan_success:
                break
            # Use the actual free tip and resting key surface. Approaching from
            # fitted tool origins and nominal key tops wasted ~190 ticks/note.
            # These geometry reads belong only to the privileged demonstrator.
            tool = self.striker.vertices
            tip_z = tool[:,2].min()
            tip = np.r_[tool[tool[:,2] < tip_z+.003,:2].mean(axis=0),tip_z]
            offset = self._robot_manager.get_gripper_center_pose().p-tip
            key_index = self.score.key_sequence[i]
            top = np.asarray(self.layout[key_index].strike_position)
            surface = self.keys[key_index].vertices
            patch = (np.abs(surface[:,0]-top[0]) < .010) & (np.abs(surface[:,1]-top[1]) < .008)
            top[2] = surface[patch,2].max()
            geometry = dict(note=i,key=key_index,rest_key_height_m=float(top[2]),
                nominal_key_height_m=self.layout[key_index].strike_position[2],
                initial_tip_position_m=tip.tolist(),held_tool_offset_m=offset.tolist())
            self.metadata['piano_geometry'].append(geometry)
            position = top+offset+[0,0,.003]
            if i == 0:
                self.move_center(position)
            elif not self._line_to(position):
                break
            reached = False
            for _ in range(180):
                force = self.key_forces[key_index]
                error = note.force_N-force
                # Limit impact on landing; contact hysteresis handles the small
                # initial load transient independently of the note-force hold.
                # The first contact can be much stiffer than the settled tip.
                # Keep each 60 Hz descent below 75 microns so a low initial
                # force cannot trigger a 250-micron impact on the next tick.
                dz = -.000075 if force < .15 else float(np.clip(-error*.00004,-.000075,.00010))
                position[2] += dz
                continued = self.servo_center(position)
                self.metadata['piano_pressure_trace'].append(dict(step=self.step_count,note=i,
                    key=key_index,force_N=float(self.key_forces[key_index]),
                    max_force_N=float(self.key_forces.max()),max_key=int(self.key_forces.argmax()),
                    target_z=float(position[2])))
                if not continued:
                    break
                if self.score.stable_steps*self.cfg.sim.dt >= note.hold_s:
                    reached = True
                    break
            if not reached:
                self.failure = self.failure or "note_force_timeout"
                break
            surface = self.keys[key_index].vertices
            geometry['held_key_deflection_m'] = float(top[2]-surface[patch,2].max())
            # Lift immediately after the required hold. Release is scored too.
            for _ in range(12):
                # Clear the elastic key's return stroke in the first update;
                # a slower lift lets the key rebound into the tool a tick after
                # release, which is correctly detected as an extra key press.
                position[2] += .004
                if not self.servo_center(position):
                    break
                if self.score.index > i:
                    break
            # The strike points lie in front of the black keys. Stay just above
            # the white keys between adjacent notes, with a larger final lift.
            position[2] += .015 if i == len(self.score.notes)-1 else .004
            if not self._line_to(position, spacing=.003):
                break
        self.metadata["memory_result"] = dict(success=self.check_success(),failure=self.failure,
            cue_acquired=self.memory.acquired,notes=self.score.events,
            force_source="physical UIPC key-top contact resultant",
            policy_steps=self.step_count-self.policy_start_step)

    def check_success(self):
        return bool(self.score.success and self.memory.ready and not self.failure and self.plan_success)
