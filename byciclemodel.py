import numpy as np
import math
import pygame

max_steer = np.radians(30.0)  # [rad] max steering angle
L = 2.9  # [m] Wheel base of vehicle
# dt = 0.1
Lr = L / 2.0  # [m]
Lf = L - Lr
Cf = 1600.0 * 2.0  # N/rad
Cr = 1700.0 * 2.0  # N/rad
Iz = 2250.0  # kg/m2
m = 1500.0  # kg


# non-linear lateral bicycle model
class NonLinearBicycleModel():
    def __init__(self, x, y, yaw=0.0, vx=0.01, vy=0, omega=0.0):
        self.m2p = 3779.52  # meters to pixels

        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        self.grey = (70, 70, 70)

        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega
        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01
        # position and speed for simulation
        self.x_screen = x
        self.y_screen = y
        self.v_screen = np.sqrt(vx ** 2 + vy ** 2)

        self.rectangle = Rectangle((self.x, self.y), yaw, 20, 20)

    def draw(self, map):
        self.rectangle.rotation((self.x, self.y), self.yaw)
        points = [self.rectangle.buf_p1, self.rectangle.buf_p2, self.rectangle.buf_p3, self.rectangle.buf_p4,
                  self.rectangle.buf_p1]
        pygame.draw.lines(map, self.red, False, points, 1)

    def update(self, throttle, delta, dt):
        # applying the control inputs
        delta = np.clip(delta, -max_steer, max_steer)
        Ffy = -Cf * math.atan2(((self.vy + Lf * self.omega) / self.vx - delta), 1.0)
        Fry = -Cr * math.atan2((self.vy - Lr * self.omega) / self.vx, 1.0)
        R_x = self.c_r1 * abs(self.vx)
        F_aero = self.c_a * self.vx ** 2
        F_load = F_aero + R_x
        self.vx = self.vx + (throttle - Ffy * math.sin(delta) / m - F_load / m + self.vy * self.omega) * dt
        self.vy = self.vy + (Fry / m + Ffy * math.cos(delta) / m - self.vx * self.omega) * dt
        self.omega = self.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt

        self.yaw = self.yaw + self.omega * dt
        self.yaw = normalize_angle(self.yaw)

        self.x = self.x + self.vx * math.cos(self.yaw) * dt - self.vy * math.sin(self.yaw) * dt
        self.y = self.y + self.vx * math.sin(self.yaw) * dt + self.vy * math.cos(self.yaw) * dt

        # updating screen variables
        self.v_screen = np.sqrt(self.vx ** 2 + self.vy ** 2) * self.m2p
        self.x_screen = self.x + self.vx * self.m2p * math.cos(self.yaw) * dt - self.vy * self.m2p * math.sin(
            self.yaw) * dt
        self.y_screen = self.y + self.vx * self.m2p * math.sin(self.yaw) * dt + self.vy * self.m2p * math.cos(
            self.yaw) * dt * self.m2p


# model:
# Gao, Feng, Qiuxia Hu, Jie Ma, and Xiangyu Han. 2021.
# "A Simplified Vehicle Dynamics Model for Motion Planner Designed by Nonlinear Model Predictive Control"
# Applied Sciences 11, no. 21: 9887. https://doi.org/10.3390/app11219887

# reference: https://www.coursera.org/lecture/intro-self-driving-cars/lesson-2-the-kinematic-bicycle-model-Bi8yE,
# we used the "Rear Alex Bicycle model" as mentioned in that tutorial. TODO: update Read.me
class LinearBicycleModel(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, throttle, delta):
        """
        Update the state of the vehicle.
        Stanley Control uses bicycle model.
        :param a: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += throttle * dt


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


class Rectangle():
    def __init__(self, center, yaw, width, length):
        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        self.grey = (70, 70, 70)

        self.center_x, self.center_y = center
        self.p1 = (self.center_x - length / 2, self.center_y - width / 2)
        self.p2 = (self.center_x - length / 2, self.center_y + width / 2)
        self.p3 = (self.center_x + length / 2, self.center_y + width / 2)
        self.p4 = (self.center_x + length / 2, self.center_y - width / 2)

        self.buf_p1 = self.p1
        self.buf_p2 = self.p2
        self.buf_p3 = self.p3
        self.buf_p4 = self.p4

        self.yaw = yaw

        self.rotation(center, yaw)

    def rotation(self, current_pos, yaw):
        self.yaw = yaw
        vec = np.array([current_pos[0] - self.center_x, current_pos[1] - self.center_y])

        # Traslating the corners
        self.p1 = self.p1 + vec
        self.p2 = self.p2 + vec
        self.p3 = self.p3 + vec
        self.p4 = self.p4 + vec

        # Rotating the corners about the car center
        self.center_x, self.center_y = current_pos
        self.buf_p1 = np.array(
            [self.center_x + (self.p1[0] - self.center_x) * math.cos(self.yaw) - (
                    self.p1[1] - self.center_y) * math.sin(self.yaw),
             self.center_y + (self.p1[0] - self.center_x) * math.sin(self.yaw) + (
                     self.p1[1] - self.center_y) * math.cos(self.yaw)])
        self.buf_p2 = np.array(
            [self.center_x + (self.p2[0] - self.center_x) * math.cos(self.yaw) - (
                    self.p2[1] - self.center_y) * math.sin(self.yaw),
             self.center_y + (self.p2[0] - self.center_x) * math.sin(self.yaw) + (
                     self.p2[1] - self.center_y) * math.cos(self.yaw)])
        self.buf_p3 = np.array(
            [self.center_x + (self.p3[0] - self.center_x) * math.cos(self.yaw) - (
                    self.p3[1] - self.center_y) * math.sin(self.yaw),
             self.center_y + (self.p3[0] - self.center_x) * math.sin(self.yaw) + (
                     self.p3[1] - self.center_y) * math.cos(self.yaw)])
        self.buf_p4 = np.array(
            [self.center_x + (self.p4[0] - self.center_x) * math.cos(self.yaw) - (
                    self.p4[1] - self.center_y) * math.sin(self.yaw),
             self.center_y + (self.p4[0] - self.center_x) * math.sin(self.yaw) + (
                     self.p4[1] - self.center_y) * math.cos(self.yaw)])
