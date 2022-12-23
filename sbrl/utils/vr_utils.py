"""README
1) place the headset exactly in the same orientation as the robot base;
2) headset coordinate system w.r.t to the robot base coordinate system:
oculus => robot:
x_robot = 0 * x + 0 * y - 1 * z
y_robot = -1 * x + 0 * y + 0 * z
z_robot = 0 * x + 1 * y + 0 * z
3) robot coordinate system w.r.t global (only rotation)
robot => global:
x_global = 0 * x + -1 * y + 0 * z
y_global = 1 * x + 0 * y + 0 * z
z_global = 0 * x + 0 * y + 1 * z
"""

controller_off_message = '\n\nATTENTION: Controller is off! Press enter when you have turned it back on :)'
