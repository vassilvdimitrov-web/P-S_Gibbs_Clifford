import csv

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
        congested = avg_v < 0.5 * self.v_max if cars else False

        # queue info (for traffic lights analysis)
        if entry_nodes:
            avg_queue = sum(len(n.waiting_queue) for n in entry_nodes) / len(entry_nodes)
        else:
            avg_queue = 0

        self.history.append({
            "density": density,
            "velocity": avg_v,
            "flow": flow,
            "congested": congested,
            "cars": len(cars),
            "queue": avg_queue
        })

    def save(self, filename="traffic_data.csv"):
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "density", "velocity", "flow", "congested", "cars", "queue"
            ])
            writer.writeheader()
            writer.writerows(self.history)

    def first_congestion_step(self):
        for i, h in enumerate(self.history):
            if h["congested"]:
                return i
        return None