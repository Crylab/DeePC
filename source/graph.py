import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import model


class graph:
    def __init__(self, params: dict = {}):
        if 'name' in params.keys():
            self.name = params['name']
        else:
            self.name = '2D Vehicle Animation'

        if 'xmin' in params.keys():
            self.xmin = params['xmin']
        else:
            self.xmin = -20

        if 'ymin' in params.keys():
            self.ymin = params['ymin']
        else:
            self.ymin = -20

        if 'xmax' in params.keys():
            self.xmax = params['xmax']
        else:
            self.xmax = 20

        if 'ymax' in params.keys():
            self.ymax = params['ymax']
        else:
            self.ymax = 20

        if 'vehicle_length' in params.keys():
            self.vehicle_length = params['vehicle_length']
        else:
            self.vehicle_length = 2

        if 'vehicle_width' in params.keys():
            self.vehicle_width = params['vehicle_width']
        else:
            self.vehicle_width = 1

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