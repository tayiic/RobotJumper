"""Load the official Jumper model in MuJoCo or record a bounded environment check."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
model = mujoco.MjModel.from_xml_path(str(ROOT / "upstream/assets/jumper/jumper.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

if args.check:
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    with mujoco.Renderer(model, height=600, width=800) as renderer:
        renderer.update_scene(data)
        Image.fromarray(renderer.render()).save(output / "jumper-loaded.png")
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.time > 0
    result = dict(mujoco=mujoco.__version__, joints=model.njnt,
                  actuators=model.nu, steps=1000, simulation_seconds=data.time,
                  finite_state=True, note="Uncontrolled physics check; no trained policy")
    (output / "simulation-check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
else:
    import mujoco.viewer
    mujoco.viewer.launch(model=model, data=data)
