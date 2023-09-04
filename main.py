import numpy as np
import pygame
from difdrive import RRTGraph, Robot, Envir
import time
from byciclemodel import NonLinearBicycleModel
from controller import Controller2D


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
    environment.drawMap(obstacles, )

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
            graph.collision_count = {}
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
    # print(list(zip(coord_path_x, coord_path_y)))
    # print(f'Path = {coord_path}')
    # print(f'Path found, Execution time: {time.time() - t0} sec. Iterations {iteration}\n')

    # the robot
    # robot = Robot(start, r"C:\Users\Giacomo Burani\PycharmProjects\DifDriveRobot\DDR.png",
    #               0.03 * 3779.52, coord_path=coord_path)
    robot = NonLinearBicycleModel(start[0], start[1])

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
        dt = (pygame.time.get_ticks() - last_time) / 1000
        last_time = pygame.time.get_ticks()
        pygame.display.update()
        environment.map.fill(environment.white)
        # robot.move(dt=dt)
        # print(f"Robot speed, vl: {robot.vl}, vr: {robot.vr}")
        controller.update_values(robot.x, robot.y, robot.yaw, np.sqrt(robot.vx ** 2 + robot.vy ** 2))
        controller.pure_pursuit_steer_control()
        # controller.update_desired_speed()
        robot.update(controller.throttle, controller.steer, dt)
        robot.draw(environment.map)
        environment.trail((robot.x, robot.y))
        environment.robot_frame((robot.x, robot.y), robot.yaw)
        # environment.robot_frame((robot.x, robot.y), robot.theta)
        environment.target = controller.target
        environment.drawMap(obstacles)
        environment.drawPath(graph.getPathCoords())
        environment.write_info(round(robot.x, 2), round(robot.y, 2), round(np.sqrt(robot.vx ** 2 + robot.vy ** 2), 2),
                               robot.yaw)


if __name__ == '__main__':
    main()
