import math

import numpy as np
import pygame

import DWmodified
from difdrive import RRTGraph, Envir
import time
from byciclemodel import NonLinearBicycleModel
from controller import Controller2D
from DWmodified import *


def main():
    # initialization
    pygame.init()

    # start position
    start = (200, 200)
    goal = (600, 200)

    # dimentions
    dims = (600, 1200)
    obsdim = 30
    obsnum = 30

    # running or not
    running = True

    # the envir
    environment = Envir(start, goal, dims, obsdim, obsnum)
    environment.trail((start[0], start[1]))
    graph = RRTGraph(start, goal, dims, obsdim, obsnum, car_lat_dim=0.005 * 3779.52)

    # obstacles
    obstacles = graph.makeobs()
    environment.drawMap(obstacles)

    iteration = 0

    t1 = time.time()
    t0 = t1

    # print(self.static_obstacles[13].shape.intersects(Point(-26,29).buffer(1.7)))
    # print(Point(-26,29).buffer(1.7).intersects(self.static_obstacles[13].shape))

    while not graph.goalFlag:
        if iteration % 20 == 0:
            X, Y, Parent = graph.bias(goal)
            pygame.draw.circle(environment.map, environment.grey, (X[-1], Y[-1]), environment.nodeRad + 2, 0)
            pygame.draw.line(environment.map, environment.blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]),
                             environment.edgeThickness)
        else:
            X, Y, Parent = graph.expand()
            pygame.draw.circle(environment.map, environment.grey, (X[-1], Y[-1]), environment.nodeRad + 2, 0)
            pygame.draw.line(environment.map, environment.blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]),
                             environment.edgeThickness)
        if iteration % 5 == 0:
            pygame.display.update()

        # restarting tree if search takes too long
        if not graph.goalFlag and iteration > 8000:
            (x, y) = start
            graph.goalFlag = False
            graph.x = []
            graph.y = []
            graph.parent = []

            # initialize the tree
            graph.x.append(x)
            graph.y.append(y)
            graph.parent.append(0)

            # optimization
            graph.collision_count = {}  # allows us to turn off some nodes that are in bad positions
            graph.collision_count[0] = 0
            graph.iteration = 0

            # path
            graph.goalstate = None
            graph.path = []

            iteration = 0
            print('Reset')

        iteration += 1

    graph.path_to_goal()

    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)
    environment.drawPath(graph.getPathCoords())
    pygame.display.update()

    coord_path, coord_path_x, coord_path_y = graph.waypoints2path()
    # print(f'Path found, Execution time: {time.time() - t0} sec. Iterations {iteration}\n')

    # the robot
    robot = NonLinearBicycleModel(start[0], start[1])

    # configuration for the Dynamic window approach
    config = DWmodified.Config(obstacles)
    u, predicted_trajectory = dwa_control([robot.x, robot.y, robot.yaw, np.sqrt(robot.vx ** 2 + robot.vy ** 2),
                                           robot.omega], config, np.array(goal), config.ob)
    robot.draw_dynamic_window(environment.map, predicted_trajectory)

    # controller
    controller = Controller2D(coord_path)
    robot.draw(environment.map)

    # dt
    dt = 0
    last_time = pygame.time.get_ticks()

    # simulation loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # robot.move(dt=dt, event=event)

        # simulation
        dt = (pygame.time.get_ticks() - last_time) / 1000
        last_time = pygame.time.get_ticks()

        # Dynamic Window Approach
        u, predicted_trajectory = dwa_control([robot.x, robot.y, robot.yaw, np.sqrt(robot.vx ** 2 + robot.vy ** 2),
                                               robot.omega], config, np.array(goal), config.ob)
        robot.draw_dynamic_window(environment.map, predicted_trajectory)

        # updating position
        pygame.display.update()
        environment.map.fill(environment.white)
        controller.update_values(robot.x, robot.y, robot.yaw, np.sqrt(robot.vx ** 2 + robot.vy ** 2))

        # updating the control inputs
        controller.pure_pursuit_steer_control()

        # applying control inputs
        robot.update(controller.throttle, controller.steer, dt)

        # drawing the new situation
        robot.draw(environment.map)  # draw the rectangle
        environment.trail((robot.x, robot.y))
        # print(math.degrees(controller.steer))
        environment.robot_frame((robot.x, robot.y), robot.yaw, controller.steer)
        environment.target = controller.target
        environment.drawMap(obstacles)
        environment.drawPath(graph.getPathCoords())
        environment.write_info(round(robot.x, 2), round(robot.y, 2), round(np.sqrt(robot.vx ** 2 + robot.vy ** 2), 2),
                               robot.yaw, round(controller.throttle, 2))
        # check reaching goal
        dist_to_goal = math.hypot(robot.x - goal[0], robot.y - goal[1])
        if dist_to_goal <= graph.car_lat_dim:
            print("Goal!!")
            pygame.display.update()
            pygame.event.clear()
            pygame.event.wait(0)
            break


if __name__ == '__main__':
    main()
