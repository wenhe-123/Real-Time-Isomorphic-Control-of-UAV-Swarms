import numpy as np
from collections import deque

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line


def generate_waypoints(n_drones, times, init_pos):
    waypoints = {}

    # intial position
    current = init_pos.copy()
    waypoints[times[0]] = current.copy()


    for t in times:
        # noise with mean value and standard deviation
        noise = np.random.normal(scale=0.1, size=(n_drones, 3))
        current = current + noise

        # bounding box constraint
        current = np.clip(current, [-0.3, -0.3, 0.3], [0.3, 0.3, 0.5])

        waypoints[t] = current.copy()

    return waypoints

# move step by step
def interpolate(t, times, waypoints, init_pos):
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return (1 - alpha) * waypoints[t0] + alpha * waypoints[t1]
    return waypoints[times[-1]]


def main():
    n_worlds = 1
    n_drones = 3

    # 创建仿真器对象
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        control=Control.state
    )

    sim.reset()

    init_pos = np.array([
        [0.0, 0.0, 0.5],
        [0.3, 0.0, 0.5],
        [-0.3, 0.0, 0.5],
    ])

    # debug
    # print("DEBUG control mode:", sim.data.controls.mode)

    duration = 15.0
    fps = 60

    times = [0, 5, 10, 15]
    waypoints = generate_waypoints(n_drones, times, init_pos)

    pos_buffer = deque(maxlen=50) # save recorded positions
    smooth_target = None
    alpha_smooth = 0.08


    for i in range(int(duration * sim.control_freq)):
        t = i / sim.control_freq

        
        raw_target = interpolate(t, times, waypoints, init_pos)

        if smooth_target is None:
            smooth_target = raw_target.copy()
        else:
            # smooth_target
            smooth_target = alpha_smooth * raw_target + (1 - alpha_smooth) * smooth_target


        #define state 
        #    [x, y, z,
        #   vx, vy, vz,
        #   qx, qy, qz, qw,      quaternion
        #   wx, wy, wz ]

        cmd = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)
        cmd[..., :3] = smooth_target  
        cmd[..., 6:10] = [0, 0, 0, 1]

        # control
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)

        # record the trace
        if i % 5 == 0:
            pos_buffer.append(sim.data.states.pos[0].copy())

        # draw
        if ((i * fps) % sim.control_freq) < fps:
            if len(pos_buffer) > 1:
                lines = np.array(pos_buffer)
                for d in range(n_drones):
                    draw_line(
                        sim,
                        lines[:, d, :],
                        rgba=[1.0, 0.0, 0.0, 1.0],
                        start_size=0.5,
                        end_size=2.0,
                    )
            sim.render()

    sim.close()


if __name__ == "__main__":
    main()
