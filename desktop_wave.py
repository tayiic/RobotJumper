"""Native MuJoCo desktop idle demonstration. No hardware connections."""
import argparse
import ast
import time
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
src = root / "upstream"
constants = ast.parse((src / "tasks/jumper/common/constants.py").read_text(encoding="utf-8"))
home = next(ast.literal_eval(n.value) for n in constants.body
            if isinstance(n, ast.AnnAssign) and n.target.id == "HOME")
spec = mj.MjSpec.from_file(str(src / "assets/jumper/jumper.xml"))
spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE,
                        size=[1, 1, .02], rgba=[.45, .48, .52, 1])
spec.worldbody.add_light(pos=[0, 0, 1], dir=[0, 0, -1])
m = spec.compile()
d = mj.MjData(m)
joints = [m.joint(name) for name in home]
q = np.array([int(j.qposadr[0]) for j in joints])
v = np.array([int(j.dofadr[0]) for j in joints])
rest = np.array(list(home.values()))
d.qpos[2] = .10647
d.qpos[q] = rest
mj.mj_forward(m, d)
tip = m.site("LF").id
jac = np.zeros((3, m.nv))
mj.mj_jacSite(m, d, jac, None, tip)
arm = [i for i, name in enumerate(home) if name.startswith("LF_") and "J4" not in name]
lift = np.zeros_like(rest)
gradient = jac[2, v[arm]]
lift[arm] = .7 * gradient / np.linalg.norm(gradient)
wrist = list(home).index("LF_J3_joint")
limits = m.jnt_range[[j.id for j in joints]]


def advance(phase):
    envelope = np.clip(min(phase, 4 - phase), 0, 1)
    envelope = .5 - .5 * np.cos(np.pi * envelope)
    target = rest + envelope * lift
    target[wrist] += envelope * .28 * np.sin(2 * np.pi * np.clip(phase - 1, 0, 2))
    target = np.clip(target, limits[:, 0], limits[:, 1])
    d.qfrc_applied[v] = np.clip(10 * (target - d.qpos[q]) - .5 * d.qvel[v], -1.7464, 1.7464)
    mj.mj_step(m, d)


def camera(cam):
    cam.lookat[:] = [0, 0, .09]
    cam.distance, cam.azimuth, cam.elevation = .77, 38, -23


if args.check:
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    frames = []
    max_drift = 0.
    peaks = np.zeros(10)
    cam = mj.MjvCamera()
    camera(cam)
    with mj.Renderer(m, 320, 480) as renderer:
        for step in range(50000):
            phase = step % 5000 * m.opt.timestep
            advance(phase)
            max_drift = max(max_drift, np.linalg.norm(d.qpos[:2]))
            peaks[step // 5000] = max(peaks[step // 5000], d.site_xpos[tip, 2])
            if step < 5000 and step % 125 == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(Image.fromarray(renderer.render()))
            assert np.isfinite(d.qpos).all() and d.qpos[2] > .075 and abs(d.qpos[3]) > .96
    print(f"cycles=10 drift_max_m={max_drift:.5f} minimum_peak_z_m={min(peaks):.5f}", flush=True)
    assert max_drift < .02 and min(peaks) > .045
    frames[0].save(out / "desktop-wave.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        camera(viewer.cam)
        while viewer.is_running():
            tick = time.monotonic()
            for _ in range(10):
                advance(d.time % 40)
            viewer.sync()
            time.sleep(max(0., 10 * m.opt.timestep - (time.monotonic() - tick)))
