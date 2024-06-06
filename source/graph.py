import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import model


class graph:
    def __init__(self, params: dict = {}):
        
        self.name = params['name'] if 'name' in params else '2D Vehicle Animation'
        self.xmin = params['xmin'] if 'xmin' in params else -20
        self.ymin = params['ymin'] if 'ymin' in params else -20
        self.xmax = params['xmax'] if 'xmax' in params else 20
        self.ymax = params['ymax'] if 'ymax' in params else 20
        self.vehicle_length = params['vehicle_length'] if 'vehicle_length' in params else 2
        self.vehicle_width = params['vehicle_width'] if 'vehicle_width' in params else 1

        self.path = np.array([])
        self.path_thickness = 0.6
        self.aux = []

    def add_path(self, path: np.array):
        self.path = path
        
    def add_state_path(self, states: list):
        path = []
        for each in states:
            path.append([each.x, each.y, each.heading])
        self.path = np.array(path)

    def add_aux(self, aux: np.array, line_type: str):
        self.aux.append((aux, line_type))

    def compression(self, degree: int):
        if self.path.size == 0:
            raise Exception("There is no path to compress.")
        for each in self.aux:
            if len(each[0]) != len(self.path):
                raise Exception("The aux charts must have the same length")
        result = []
        counter = 0
        for i in range(0, len(self.path)):
            if i % degree == 0:
                result.append(self.path[i])
                counter += 1
        self.path = np.array(result)
        result_aux = []
        for each in self.aux:
            result_local = []
            counter = 0
            for i in range(0, len(each[0])):
                if i % degree == 0:
                    result_local.append(each[0][i])
                    counter += 1
            result_aux.append((np.array(result_local), each[1]))
        self.aux = result_aux


    def generate_gif(self, name: str = 'vehicle_animation.gif'):
        if self.path.size == 0:
            raise Exception("There is no path to plot.")
        fig, ax = plt.subplots()

        for each in self.aux:
            if len(each[0]) != len(self.path):
                raise Exception("The aux charts must have the same length")

        def plot_vehicle(ax, x, y, theta):
            # Define vehicle points
            vehicle_points = np.array([[-self.vehicle_length/2,
                                        -self.vehicle_width/2],
                                       [self.vehicle_length/2,
                                        -self.vehicle_width/2],
                                       [self.vehicle_length/2,
                                        self.vehicle_width/2],
                                       [-self.vehicle_length/2,
                                        self.vehicle_width/2],
                                       [-self.vehicle_length/2,
                                        -self.vehicle_width/2]])

            # Rotation matrix
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])

            # Rotate vehicle points
            rotated_points = np.dot(vehicle_points, rotation_matrix.T)

            # Translate vehicle points to given position
            translated_points = rotated_points + np.array([x, y])

            # Plot vehicle
            ax.plot(translated_points[:, 0], translated_points[:, 1], 'b-')

        def plot_path(ax, path, frame_end):
            if frame_end == 0:
                return
            path_transposed = path.T.tolist()
            ax.plot(path_transposed[0][0:frame_end],
                    path_transposed[1][0:frame_end], linewidth=self.path_thickness)
            
        def plot_aux(ax, aux, line_type):
            aux_transposed = aux.T.tolist()
            ax.plot(aux_transposed[0],
                    aux_transposed[1], line_type)

        def update(frame):
            ax.clear()
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(self.name)
            ax.grid()
            plot_vehicle(ax, *self.path[frame])
            plot_path(ax, self.path, frame)
            for each in self.aux:
                plot_aux(ax, each[0][frame], each[1])

        ani = FuncAnimation(fig, update, frames=len(self.path),
                            interval=0, repeat=False)
        ani.save(name, writer='pillow', fps=24)
        
class graph_compete(graph):
    def __init__(self, params: dict = {}):
        super().__init__(params)
        self.path = []
        self.landscape = []

    def __add_path(self, path: np.array, color: str, shift: int = 0, name: str = 'Default'):
        self.path.append((path, color, shift, name))
        
    def add_state_path(self, states: list, color: str, shift: int = 0, name: str = 'Default'):
        path = []
        for each in states:
            path.append([each.x, each.y, each.heading])
        self.path.append((np.array(path), color, shift, name))
        
    def add_state_landscape(self, states: list, line_type: str):
        path = []
        for each in states:
            path.append([each.x, each.y])
        self.landscape.append((np.array(path), line_type))

    def __add_landscape(self, path: np.array, line_type: str):
        self.landscape.append((path, line_type))

    def compression(self, degree: int):
        if self.path[0][0].size == 0:
            raise Exception("There is no path to compress.")
        for each in self.aux:
            if len(each[0]) != len(self.path[0][0]):
                raise Exception("The aux charts must have the same length")
        
        for j in range(0, len(self.path)):
            result = []
            counter = 0
            for i in range(0, len(self.path[j][0])):
                if i % degree == 0:
                    result.append(self.path[j][0][i])
                    counter += 1
            self.path[j] = (np.array(result), self.path[j][1], self.path[j][2], self.path[j][3])
        result_aux = []
        for each in self.aux:
            result_local = []
            counter = 0
            for i in range(0, len(each[0])):
                if i % degree == 0:
                    result_local.append(each[0][i])
                    counter += 1
            result_aux.append((np.array(result_local), each[1], each[2]))
        self.aux = result_aux

    def __add_aux(self, aux: np.array, line_type: str, shift: int = 0):
        self.aux.append((aux, line_type, shift))
        
    def transpose(self):
        for i in range(len(self.path)):
            temp = self.path[i][0].T[0].copy()
            self.path[i][0].T[0] = self.path[i][0].T[1].copy()
            self.path[i][0].T[1] = temp
            
        temp = self.landscape[0][0].T[0].copy()
        self.landscape[0][0].T[0] = self.landscape[0][0].T[1].copy()
        self.landscape[0][0].T[1] = temp
        
    def __generate_graphics(self, name: str = 'picture_at.png', animation: bool = False, moment: float = 100.0):
        if len(self.path) == 0:
            raise Exception("There is no path to plot.")
        fig, ax = plt.subplots()
        for each in self.aux:
            if len(each[0]) != len(self.path[0][0]):
                raise Exception("The aux charts must have the same length")
        
        def plot_vehicle(ax, x, y, theta, color, name):
            # Define vehicle points
            vehicle_points = np.array([[-self.vehicle_length/2,
                                        -self.vehicle_width/2],
                                       [self.vehicle_length/2,
                                        -self.vehicle_width/2],
                                       [self.vehicle_length/2,
                                        self.vehicle_width/2],
                                       [-self.vehicle_length/2,
                                        self.vehicle_width/2],
                                       [-self.vehicle_length/2,
                                        -self.vehicle_width/2]])

            # Rotation matrix
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])

            # Rotate vehicle points
            rotated_points = np.dot(vehicle_points, rotation_matrix.T)

            # Translate vehicle points to given position
            translated_points = rotated_points + np.array([x, y])

            # Plot vehicle
            ax.plot(translated_points[:, 0], translated_points[:, 1], color, label=name)

        def plot_path(ax, path, frame_end, color):
            if frame_end == 0:
                return
            path_transposed = path.T.tolist()
            ax.plot(path_transposed[0][0:frame_end],
                    path_transposed[1][0:frame_end], color, linewidth=self.path_thickness)
            
        def plot_aux(ax, aux, line_type):
            aux_transposed = aux.T.tolist()
            ax.plot(aux_transposed[0],
                    aux_transposed[1], line_type, linewidth=self.path_thickness)

        def update(frame):
            ax.clear()
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)
            plt.axis('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(self.name)
            ax.grid()
            for each in self.path:
                if frame > each[2]:
                    if frame < len(each[0]):
                        plot_vehicle(ax, *each[0][frame-each[2]], each[1], each[3])
                        plot_path(ax, each[0], frame-each[2], each[1])
                    else:
                        plot_vehicle(ax, *each[0][-1], each[1], each[3])
                        plot_path(ax, each[0], len(each[0])-1, each[1])
            for each in self.aux:
                if frame > each[2]:
                    if frame < len(each[0]):
                        plot_aux(ax, each[0][frame-each[2]], each[1])
                    else:
                        plot_aux(ax, each[0][-1], each[1])
            for each in self.landscape:
                plot_aux(ax, each[0], each[1])
            ax.legend()
        frames = len(self.path[0][0])
        for each in self.path:
            frames += each[2]
        if animation:
            ani = FuncAnimation(fig, update, frames=frames,
                                interval=0, repeat=False)
            ani.save(name, writer='pillow', fps=24)
        else:
            moment_at = int(float(len(self.path[0][0])) * moment / 100.0)
            update(moment_at)
            plt.savefig(name)
        
    def generate_gif(self, name: str = 'vehicle_animation.gif'):
        self.__generate_graphics(name, True)
        
    def generate_pic_at(self, name: str = 'picture_at.png', moment: float = 100.0):
        self.__generate_graphics(name, False, moment)
