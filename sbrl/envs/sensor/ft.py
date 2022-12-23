import numpy as np

import sbrl.utils.ros_utils as ru
from sbrl.envs.sensor.sensors import Sensor
from sbrl.experiments import logger
from sbrl.utils.python_utils import AttrDict

ru.ros_import_open()
from geometry_msgs.msg import WrenchStamped

ru.ros_import_close()


class FTSensor(Sensor):
    def _init_params_to_attrs(self, params: AttrDict):
        self.ft_sensor_topic = params.get("ft_sensor_topic", "/wireless_ft/wrench_3")
        self.mass_on_sensor = params.get("ft_mass_on_sensor", 0.045)  # kg

        # transform xy plane counter clockwise by:
        self.theta = 0  # 5. / 6 * np.pi  # y up, x left
        self.frame_rot = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                                   [np.sin(self.theta), np.cos(self.theta), 0],
                                   [0, 0, 1]])

        # self.calibration = np.array([[30.8075377269387, 1.2893008157985, 813.569688708002, -7254.89791700611, -787.595279064185, 7104.39474007346],
        #                             [-871.782540620028, 8324.75261230435, 539.573226128255, -4190.17786942629, 305.143608475913, -4103.53338191159],
        #                             [10394.2105970546, 387.126558109331, 10287.5435837209, 545.14516888845, 9983.94635516981, 648.901079941133],
        #                             [-7.09817753535296, 58.629315710162, -157.956380580148, -37.7281831894694, 161.827405736171, -19.0197000158304],
        #                             [195.846997398986, 6.4510344807947, -101.436603140553, 46.4263739657068, -89.6654792534856, -55.793052288557],
        #                             [12.089642172249, -106.169056621114, 9.8490093249083, -107.484272257137, 12.0232903778477, -104.766830359676]])
        # self.gauge_offsets = np.array([30344, 32009, 31090, 30444, 30748, 31980])
        # self.gauge_gains = np.array([774, 806, 802, 798, 806, 810])
        self.baseline_ft_steps = params.get("baseline_ft_steps", 0)  # 0 means no baseline, raw values
        logger.debug("Baseline steps: %d" % self.baseline_ft_steps)
        self.ft_buffer = np.zeros((self.baseline_ft_steps, 6))
        self.baseline_ft_value = 0
        self.latest_ft = None
        self.latest_ft_time = 0
        self.ft_step = 0

    def _init_setup(self, **kwargs):
        self._ft_subscriber = ru.bind_subscriber(ru.RosTopic(self.ft_sensor_topic, WrenchStamped),
                                                 self.ft_sensor_callback)
        logger.debug("Using FT sensor on ros topic: %s. Waiting for callbacks..." % self.ft_sensor_topic)
        while self.latest_ft is None:
            pass
        logger.debug("FT sensor found.")

    def read_state(self, **kwargs):
        assert self.latest_ft is not None
        if self.ft_step < self.baseline_ft_steps:
            latest_ft = np.zeros(6)  # zero until baseline gets set
        else:
            latest_ft = self.latest_ft - self.baseline_ft_value
        # print(latest_ft)
        return AttrDict(force=latest_ft[:3], torque=latest_ft[3:], time=self.latest_ft_time)

    def ft_sensor_callback(self, msg):
        force = self.frame_rot @ np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        torque = self.frame_rot @ np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        self.latest_ft = np.concatenate([force, torque])

        self.latest_ft_time = msg.header.stamp.to_sec()
        if self.ft_step < self.baseline_ft_steps:
            self.ft_buffer[self.ft_step] = self.latest_ft.copy()
        # average of the buffer is our baseline
        if self.ft_step == self.baseline_ft_steps - 1:
            self.baseline_ft_value = self.ft_buffer.mean(0)
        self.ft_step += 1

    def reset(self, **kwargs):
        self.ft_buffer = np.zeros((self.baseline_ft_steps, 6))
        self.latest_ft = np.zeros(6)
        self.latest_ft_time = 0
        self.ft_step = 0
        self.baseline_ft_value = 0
