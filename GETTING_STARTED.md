# Jumper simulation

Windows setup with Python 3.11 and Git:

```powershell
git clone https://github.com/KingKongRobotics/jumper.git upstream
git -C upstream checkout 198686eac0d188815d0943e9fee50de415edc80a
python -m venv .venv
.venv/Scripts/python.exe -m pip install mujoco==3.11.0 numpy pillow
.venv/Scripts/python.exe desktop_wave.py --check
```

Run `桌面招手.cmd` for the interactive viewer. The script uses the official model and HOME pose, a flat floor, and a simplified PD controller. It is a simulation prototype and has not been validated on a physical robot. The official training and deployment environment is separate and requires the upstream setup procedure.
