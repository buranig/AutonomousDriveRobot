import pygame
import math
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from shapely.ops import nearest_points
import random
import numpy as np

"""
This file contains the classes for generating the simulation environment and the RRT path finding algorithm
"""


class Envir:
    def __init__(self, start, goal, dimentions, obsdim, obsnum):
        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        self.grey = (70, 70, 70)

        # map dimensions
        self.height = dimentions[0]
        self.width = dimentions[1]
        self.start = start
        self.goal = goal
        self.target = start

        # windows settings
        pygame.display.set_caption("Differential drive robot")
        self.map = pygame.display.set_mode((self.width, self.height))
        self.map.fill((255, 255, 255))
        self.nodeRad = 0
        self.nodeThickness = 5
        self.edgeThickness = 1

        self.obstacles = []
        self.obsdim = obsdim
        self.obsNumber = obsnum

        # text variables
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.text = self.font.render('default', True, self.white, self.black)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dimentions[1] - 600, dimentions[0] - 100)

        # trail
        self.trail_set = []

    def write_info(self, x, y, v, yaw, throttle):
        txt = f"X: {x}, Y: {y}, V: {v}, Yaw: {int(math.degrees(yaw))}°, Throttle: {throttle}"
        # txt = f"Vl: {vl}, Vr: {vr}, theta: {int(math.degrees(theta))}"
        self.text = self.font.render(txt, True, self.white, self.black)
        self.map.blit(self.text, self.textRect)

    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.yellow, (self.trail_set[i][0], self.trail_set[i][1]),
                             (self.trail_set[i + 1][0], self.trail_set[i + 1][1]))

        if self.trail_set.__sizeof__() > 30000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)

    def robot_frame(self, pos, yaw, delta):
        n = 80

        centerx, centery = pos
        x_axis = (centerx + n * math.cos(yaw), centery + n * math.sin(yaw))
        y_axis = (centerx + n * math.cos(yaw + math.pi / 2), centery + n * math.sin(yaw + math.pi / 2))
        arrow = (centerx + n / 2 * math.cos(yaw + delta), centery + n / 2 * math.sin(yaw + delta))
        pygame.draw.line(self.map, self.red, (centerx, centery), x_axis, 3)
        pygame.draw.line(self.map, self.green, (centerx, centery), y_axis, 3)
        pygame.draw.line(self.map, self.black, (centerx, centery), arrow, 2)

    def drawMap(self, obstacles):
        pygame.draw.circle(self.map, self.green, self.start, self.nodeRad + 5, 0)
        pygame.draw.circle(self.map, self.red, self.goal, self.nodeRad + 20, 1)
        pygame.draw.circle(self.map, self.blue, self.target, self.nodeRad + 5, 0)
        self.drawObs(obstacles)

    def drawPath(self, path):
        for node in path:
            pygame.draw.circle(self.map, self.red, node, 3, 0)

    def drawObs(self, obstacles):
        obstaclesList = obstacles.copy()
        while (len(obstaclesList) > 0):
            obstacle = obstaclesList.pop(0)
            pygame.draw.rect(self.map, self.grey, obstacle)


class RRTGraph:
    def __init__(self, start, goal, MapDimensions, obsdim, obsnum, car_lat_dim):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.maph, self.mapw = MapDimensions
        self.x = []
        self.y = []
        self.parent = []

        # initialize the tree
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)

        # optimization
        self.collision_count = {}
        self.collision_count[0] = 0

        # the obstacles
        self.obstacles = []
        # Lanelet Network
        self.obsNum = obsnum
        self.obsDim = obsdim

        # path
        self.goalstate = None
        self.path = []
        self.dmax = 50

        # Car dimension
        self.car_lat_dim = car_lat_dim * 1.5

    def makeRandomRect(self):
        uppercornerx = int(random.uniform(0, self.mapw - self.obsDim))
        uppercornery = int(random.uniform(0, self.maph - self.obsDim))

        return (uppercornerx, uppercornery)

    def makeobs(self):
        obs = []
        for i in range(0, self.obsNum):
            rectang = None
            startgoalcol = True
            while startgoalcol:
                upper = self.makeRandomRect()
                rectang = pygame.Rect(upper, (self.obsDim, self.obsDim))

                rectang_buf = self.pygame2shapley(rectang)  # converting to shapely geometry
                if rectang_buf.intersects((Point(self.start).buffer(self.car_lat_dim)) or rectang_buf.intersects(
                        Point(self.goal).buffer(self.car_lat_dim))):
                    startgoalcol = True
                else:
                    startgoalcol = False
            obs.append(rectang)
        self.obstacles = obs.copy()
        return obs

    def add_node(self, n, x, y):
        # self.x.insert(n, x)
        self.x.append(x)
        self.y.append(y)
        if n not in self.collision_count:
            self.collision_count[n] = 0

    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)

    def add_edge(self, parent, child):
        self.parent.insert(child, parent)

    def remove_edge(self, n):
        self.parent.pop(n)

    def number_of_nodes(self):
        return len(self.x)

    def distance(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        px = (float(x1) - float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2
        return (px + py) ** (0.5)

    # cambiare con intervalli non scritti cosi a cazzo
    def sample_envir(self):
        x = int(random.uniform(0, self.mapw))
        y = int(random.uniform(0, self.maph))
        return x, y

    def nearest(self, n):
        dmin = self.distance(0, n)
        nnear = 0
        for i in range(0, n):
            if self.collision_count[i] < 4:
                if self.distance(i, n) < dmin:
                    dmin = self.distance(i, n)
                    nnear = i
            # else: print('NOT CONSIDERING')
        return nnear

    def isFree(self):
        n = self.number_of_nodes() - 1
        (x, y) = (self.x[n], self.y[n])
        '''if self.lanelet.find_lanelet_by_position([np.array([x, y])])[0] == []:
            self.remove_node(n)
            return False'''

        # print(RRTGraph.circle_polygon_collision(self.obstacles[7], Circle(Point(3,-7), 1)))
        obs = self.obstacles.copy()

        point = Point(x, y).buffer(self.car_lat_dim)
        # cambiare con il mio collision checker circle-plygon collision
        for _obs in obs:
            # if RRTGraph.circle_polygon_collision(_obs, Circle(point, 1.5)):
            obst = self.pygame2shapley(_obs)
            if obst.intersects(point):
                # print(f'Point collide = {point}')
                self.remove_node(n)
                # print('Point ({x,y}) not free')
                return False
        # print(f'Node ({x,y}) is free')
        return True

    # This function checks wether the an edge collide with obstacle
    def crossObstacle(self, x1, x2, y1, y2):
        obs = self.obstacles.copy()

        # print(f'index = {idx}')
        point1 = [x1, y1]
        point2 = [x2, y2]
        if point1 == point2: return True

        # segment = LineString([point1, point2])
        vec = np.array([point2[0] - point1[0], point2[1] - point1[1]])
        vec_p = np.array([-vec[1], vec[0]])
        vec_p = vec_p / np.linalg.norm(vec_p)
        # Creating two parallel segments at distance r from the center segment
        p_0 = np.array([point1[0], point1[1]])
        p_1 = np.array([point2[0], point2[1]])
        p_up_0 = p_0 + self.car_lat_dim * vec_p
        p_up_1 = p_1 + self.car_lat_dim * vec_p
        p_up_0 = [p_up_0[0], p_up_0[1]]
        p_up_1 = [p_up_1[0], p_up_1[1]]

        p_down_0 = p_0 - self.car_lat_dim * vec_p
        p_down_0 = [p_down_0[0], p_down_0[1]]
        p_down_1 = p_1 - self.car_lat_dim * vec_p
        p_down_1 = [p_down_1[0], p_down_1[1]]

        # Idea instead of checking three line, check rectangle created with p_up and p_down

        _polygon = Polygon([p_up_0, p_up_1, p_down_1, p_down_0, p_up_0])

        for obstacle in obs:
            # print(f'obstacle = {obstacle}')
            obst = self.pygame2shapley(obstacle)
            if obst.intersects(_polygon):
                # print(f'Segment = {segment}, up = {segment_up}, down = {segment_down}')
                # print(f'Rectangle = {_polygon}')
                return True
        return False
        # or Point(x1,y1).buffer(1.7).intersects(obstacle.shape) or

    def pygame2shapley(self, obstacle):
        obs = Polygon([(obstacle.left, obstacle.top), (obstacle.left + obstacle.width, obstacle.top),
                       (obstacle.left, obstacle.top + obstacle.height),
                       (obstacle.left + obstacle.width, obstacle.top + obstacle.height),
                       (obstacle.left, obstacle.top)])
        return obs

    def connect(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        # print(f'Trying to connect ({x1,y1}) to ({x2,y2}), Parent node = {n1}')

        if n1 not in self.collision_count: self.collision_count[n1] = 0

        # Cross obstacle isn't working correctly -> doesn't account for car width
        # print(self.crossObstacle(3,4,-7,-2))

        if self.crossObstacle(x1, x2, y1, y2):
            # print(f'Collision connecting ({x1, y1}) to ({x2, y2})')
            self.remove_node(n2)
            self.goalFlag = False

            self.collision_count[n1] += 1

            return False
        else:
            # print(f'Connecting ({x1, y1}) to ({x2, y2})\n')
            self.add_edge(n1, n2)
            return True

    def step(self, nnear, nrand):
        # print('step')
        d = self.distance(nnear, nrand)
        # step farlo sono in direzioni in cui collegando non c'è collisione
        if d > self.dmax:
            u = self.dmax / d
            (xnear, ynear) = (self.x[nnear], self.y[nnear])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py, px)
            (x, y) = (int(xnear + self.dmax * math.cos(theta)),
                      int(ynear + self.dmax * math.sin(theta)))
            self.remove_node(nrand)
            # print(f'Trying to connect ({xnear, ynear}) with ({xrand, yrand})')
            if abs(x - self.goal[0]) <= self.dmax and abs(y - self.goal[1]) <= self.dmax:
                self.add_node(nrand, self.goal[0], self.goal[1])
                self.goalstate = nrand
                self.goalFlag = True
                # print(f'Distance ok, Connecting ({xnear, ynear}) with ({xrand, yrand})')
            else:
                # print(f'Distance too much, Connecting ({xnear, ynear}) with ({x, y}) instead')
                self.add_node(nrand, x, y)

    def bias(self, ngoal):
        # print('bias')
        n = self.number_of_nodes()
        self.add_node(n, ngoal[0], ngoal[1])
        nnear = self.nearest(n)
        self.step(nnear, n)
        self.connect(nnear, n)
        return self.x, self.y, self.parent

    def expand(self):
        n = self.number_of_nodes()
        x, y = self.sample_envir()
        # print(f'trying to expand ({x,y})')
        self.add_node(n, x, y)
        if self.isFree():
            xnearest = self.nearest(n)
            self.step(xnearest, n)
            self.connect(xnearest, n)
        return self.x, self.y, self.parent

    def path_to_goal(self):
        if self.goalFlag:
            # print(f'Goal Found')
            self.path = []
            self.path.append(self.goalstate)
            newpos = self.parent[self.goalstate]
            while (newpos != 0):
                self.path.append(newpos)
                newpos = self.parent[newpos]
            self.path.append(0)
        return self.goalFlag

    def getPathCoords(self):
        pathCoords = []
        for node in self.path:
            x, y = (self.x[node], self.y[node])
            pathCoords.insert(0, (x, y))
        return pathCoords

    def getPathCoords_xy(self):
        pathCoords_x = []
        pathCoords_y = []
        for node in self.path:
            x, y = (self.x[node], self.y[node])
            pathCoords_x.insert(0, x)
            pathCoords_y.insert(0, y)
        return pathCoords_x, pathCoords_y

    def cost(self, n):
        ninit = 0
        n = n
        parent = self.parent[n]
        c = 0
        while n is not ninit:
            c = c + self.distance(n, parent)
            n = parent
            if n is not ninit:
                parent = self.parent[n]
        return c

    def getTrueObs(self, obs):
        TOBS = []
        for ob in obs:
            TOBS.append(ob.inflate(-50, -50))
        return TOBS

    def waypoints2path(self):
        oldpath = self.getPathCoords()
        path = []
        pathx = []
        pathy = []
        waypoint_num = 3
        for i in range(0, len(self.path) - 1):
            # print(i)
            if i >= len(self.path):
                break
            x1, y1 = oldpath[i]
            x2, y2 = oldpath[i + 1]
            # print('---------')
            # print((x1, y1), (x2, y2))
            for i in range(0, waypoint_num):
                u = i / waypoint_num
                x = int(x2 * u + x1 * (1 - u))
                y = int(y2 * u + y1 * (1 - u))
                path.append((x, y))
                pathx.append(x)
                pathy.append(y)
                # print((x, y))
        path.append((oldpath[-1][0], oldpath[-1][1]))
        pathx.append(oldpath[-1][0])
        pathy.append(oldpath[-1][1])
        return path, pathx, pathy
