import numpy as np

class TrafficMetrics:
    def __init__(self, road_length, v_max):
        self.road_length = road_length
        self.v_max = v_max
        self.history = []

    def compute(self, cars, entry_nodes):
        if not cars:
            density = 0
            avg_v = 0
            flow = 0
        else:
            density = len(cars) / self.road_length
            avg_v = sum(car.v for car in cars) / len(cars)
            flow = density * avg_v

        # congestion definition
        #congested = avg_v < 0.5 * self.v_max if cars else False

        # queue info (for traffic lights analysis)
        if entry_nodes:
            avg_queue = sum(len(n.waiting_queue) for n in entry_nodes) / len(entry_nodes)
            max_queue = max(len(n.waiting_queue) for n in entry_nodes)
        else:
            avg_queue = 0
            max_queue = 0

        congested = False
        if cars:
            if avg_v < 0.7 * self.v_max:    #35
                congested = True
        if avg_queue > 20 or max_queue > 30:
                congested = True

        self.history.append({
            "density": density,
            "velocity": avg_v,
            "flow": flow,
            "congested": congested,
            "cars": len(cars),
            "queue": avg_queue,
            "max_queue": max_queue
        })

    def save(self, filename="traffic_data.npz"):
        dtype = np.dtype([
            ("density", np.float64),
            ("velocity", np.float64),
            ("flow", np.float64),
            ("congested", np.bool_),
            ("cars", np.int64),
            ("queue", np.float64),
            ("max_queue", np.int64),
        ])
        arr = np.array(
            [(h["density"], h["velocity"], h["flow"], h["congested"], h["cars"], h["queue"], h["max_queue"])
             for h in self.history],
            dtype=dtype,
        )
        np.savez_compressed(filename, traffic=arr)
    
    @staticmethod
    def load(filename="traffic_data.npz"):
        arr = np.load(filename)
        return arr

    def first_congestion_step(self):
        for i, h in enumerate(self.history):
            if h["congested"]:
                return i
        return None