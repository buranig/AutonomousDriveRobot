"""
2D Controller Class to be used for waypoint following demo.
"""

import numpy as np
from collections import deque
import math

# sampling time
WB = 2.9  # [m] Wheel base of vehicle
Lf = 20  # [m] look-ahead distance


class Controller2D(object):
    def __init__(self, waypoints):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 9
        self._current_frame = 0
        self._current_timestamp = 0
        self.throttle = 0
        self.brake = 0
        self.steer = 0

        self._waypoints = waypoints
        self.idx = 1
        self.target = self._waypoints[self.idx]

        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi
        self.e_buffer = deque(maxlen=20)
        self._e = 0

        # parameters for pid speed controller
        '''self.K_P = 1
        self.K_D = 0.001
        self.K_I = 0.3'''
        self.K_P = 10
        self.K_D = 0.001
        self.K_I = 0.3

    def update_values(self, x, y, yaw, speed):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed

    def update_desired_speed(self):
        min_idx = 0
        min_dist = float("inf")
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - self._current_x,
                self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        # next lines allows us to define a desired speed at every waypoint
        if min_idx < len(self._waypoints) - 1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        if desired_speed <= 9:
            self._desired_speed = desired_speed
        else:
            self._desired_speed = desired_speed - 9

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def update_controls(self):
        # update status
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        # if we have the chance of defining the desired speed at each waypoint uncomment next two lines
        # self.update_desired_speed()
        v_desired = 8  # self._desired_speed
        waypoints = self._waypoints

        # ==================================
        # LONGITUDINAL CONTROLLER, using PID speed controller
        # ==================================
        self._e = v_desired - v  # v_desired
        self.e_buffer.append(self._e)

        if len(self.e_buffer) >= 2:
            _de = (self.e_buffer[-1] - self.e_buffer[-2]) / dt
            _ie = sum(self.e_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        self.throttle = np.clip((self.K_P * self._e) + (self.K_D * _de / dt) + (self.K_I * _ie * dt), -1.0, 1.0)

        # ==================================
        # LATERAL CONTROLLER, using stanley steering controller for lateral control.
        # ==================================
        k_e = 0.3
        k_v = 20

        # 1. calculate heading error
        yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1], waypoints[-1][0] - waypoints[0][0])
        yaw_diff = yaw_path - yaw
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        if yaw_diff < - np.pi:
            yaw_diff += 2 * np.pi

        # 2. calculate crosstrack error
        current_xy = np.array([x, y])
        crosstrack_error = np.min(np.sum((current_xy - np.array(waypoints)[:, :2]) ** 2, axis=1))

        yaw_cross_track = np.arctan2(y - waypoints[0][1], x - waypoints[0][0])
        yaw_path2ct = yaw_path - yaw_cross_track
        if yaw_path2ct > np.pi:
            yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < - np.pi:
            yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + v))

        # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)

        # 3. control low
        steer_expect = yaw_diff + yaw_diff_crosstrack
        if steer_expect > np.pi:
            steer_expect -= 2 * np.pi
        if steer_expect < - np.pi:
            steer_expect += 2 * np.pi
        steer_expect = min(1.22, steer_expect)
        steer_expect = max(-1.22, steer_expect)

        # 4. update
        steer_output = steer_expect

        # Convert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * steer_output
        # Clamp the steering command to valid bounds
        self.steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)

        print(f'Steer: {self.steer}, Throttle: {self.throttle}')

    def pure_pursuit_steer_control(self):
        if self.dist(point1=(self._current_x, self._current_y), point2=self.target) < Lf and self.idx < len(
                self._waypoints) - 1:
            self.idx += 1
            self.target = self._waypoints[self.idx]

        if self.idx < len(self._waypoints) and self.dist((self._current_x, self._current_y), self._waypoints[-1]) < 10:
            self._desired_speed = 0
            self.steer = 0

        alpha = math.atan2(self.target[1] - self._current_y, self.target[0] - self._current_x) - self._current_yaw

        self.steer = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

        # decreasing the desired speed when turning
        if self.steer > math.radians(10) or self.steer < -math.radians(10):
            self._desired_speed = 5
        else:
            self._desired_speed = 9

        self.throttle = self.proportional_control()

    def proportional_control(self):
        a = 3 * (self._desired_speed-self._current_speed)
        return a

    @staticmethod
    def dist(point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)

        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance
