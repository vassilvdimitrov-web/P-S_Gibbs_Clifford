import random

class Car:
    def __init__(self, edge, s, velocity, length, reaction_speed, exit_node=None, is_bad_driver=False):
        self.edge = edge
        self.s = s          #the position of the particular segment
        self.v = velocity
        self.l = length
        self.reac = reaction_speed
        self.exit_node = exit_node
        self.is_bad_driver = is_bad_driver
        self.waiting_at_node = False
        self.finished = False           #when the car exits the system

    def __repr__(self):
        return f"Car(edge={id(self.edge)}, s={self.s:.2f}, v={self.v:.2f})"
    def __repr__(self):
        return f"Car(edge={id(self.edge)}, bad={self.is_bad_driver}, s={self.s:.2f}, v={self.v:.2f})"

class RoadEdge:             #a road object
    def __init__(self, bezier_edge, speed_limit=50.0):
        self.bezier_edge = bezier_edge
        self.node0 = bezier_edge.node0
        self.node1 = bezier_edge.node1
        self.speed_limit = speed_limit
        self.length = None   

    def update_length(self, nodes):
        self.length = self.bezier_edge.total_length(nodes)

def cars_on_edge(cars,edge):
    return [car for car in cars if car.edge is edge and not car.finished]

def distance_to_same_car_on_the_same_edge(car,next_car):
    return next_car.s - car.s

def update_velocities_on_edge(edge, cars, safe_distance=10.0, acceleration=2.0, brake_factor=5.0):
    edge_cars = [car for car in cars if car.edge is edge and not car.finished and not car.waiting_at_node]
    edge_cars.sort(key=lambda c: c.s)
    for i, car in enumerate(edge_cars):
        if car.is_bad_driver:
            safe_dist = safe_distance * 1.5
            accel = acceleration*0.7
            brake = brake_factor * 1.5
        else :
            safe_dist = safe_distance
            accel = acceleration
            brake = brake_factor
            
        if i < len(edge_cars)-1:
            next_car =edge_cars[i+1]
            gap = distance_to_same_car_on_the_same_edge(car, next_car) - next_car.l
        else:
            gap = float("inf")  #the last car on the edge (no car in front)

        if gap < safe_dist:     #reduce velocity
            car.v = max(0.0, car.v - brake * car.reac)
        else:                   # speed up
            car.v = min (edge.speed_limit, car.v + accel)

def update_positions_on_edge(edge, cars, dt=0.5):       #when the car reaches the end, it stops and waits for node logic
    for car in cars:
        if car.edge is not edge:
            continue
        if car.finished or car.waiting_at_node:
            continue
        car.s += car.v*dt
        if car.s >= edge.length:
            car.s = edge.length
            car.v = 0
            car.waiting_at_node= True

def update_all_edges(edges, cars, safe_distance=10.0, dt=0.05):
    for edge in edges:          #update velocities
        update_velocities_on_edge(edge, cars, safe_distance=safe_distance)
    for edge in edges:          #update positions
        update_positions_on_edge(edge, cars, dt=dt)


def add_car_on_edge(edge, exit_node=None, v=5.0, bad_driver_prob=0.2):
    return Car(
        edge=edge,
        s=0.0,
        velocity=v,
        length=random.uniform(8.0, 14.0),
        reaction_speed=random.uniform(0.5, 1.5),
        exit_node=exit_node,
        is_bad_driver=(random.random() < bad_driver_prob))




"""
road_length = 2000
v_max = 50
entry_points = [100,400, 500, 900, 1300]    # k "in-points" 
exit_points = [250, 500, 850, 1000, 1600]         # n "out-points"

safe_distance = 10
entry_rate = 0.05
dt = 0.5                                    # how much time passes between updates, time step
n_cars = 20                                 # start with n cars already on the road

class Car:
    def __init__(self, position, velocity, length, reaction_speed, exit_target):
        self.x = position
        self.v = velocity
        self.l = length
        self.reac = reaction_speed
        self.exit = exit_target


# make certain parts of the road with different local speed limits
def local_speed_limit(x):
    
    if 300 <= x < 500:
        return 20
    elif 900 <= x < 1100:
        return 30
    else:
    
        return v_max


def distance_to_next_car(car, next_car):
    d = next_car.x - car.x
    if d < 0:
        d += road_length
    return d

def car_relative_position(car):
    return car.x / road_length

# change speed depending on the the distance to the car in front
def update_velocities(cars):
    cars.sort(key=lambda car: car.x)                # sort the cars with respect to position
    for i in range(len(cars)):
        car = cars[i]
        next_car = cars[(i + 1) % len(cars)]
        gap = distance_to_next_car(car, next_car) - next_car.l
        limit = local_speed_limit(car.x)

        if gap < safe_distance:
            car.v = max(0, car.v - 5 * car.reac)
        else:
            car.v = min(limit, car.v + 1)

# x(t + dt) = x(t) + v*dt
def update_positions(cars):
    for car in cars:
        car.x = (car.x + car.v * dt) % road_length  # stay on the road (reapeat the circle)

def remove_exiting_cars(cars):
    remaining = []
    for car in cars:
        d = abs(car.x - car.exit)                   # distance to exit
        d = min(d, road_length - d)                 # minimal distance (beacuse of the circle 
                                                    # we can pass the 0 point when we have are 
                                                    # at the "end" of the road
        if d > 5:                                   # if car is close (≤5) to exit remove it
            remaining.append(car)
    return remaining

def try_add_cars(cars):
    for point in entry_points:
        if random.random() < entry_rate:
            too_close = False
            for car in cars:
                d = abs(car.x - point)
                d = min(d, road_length - d)
                if d < safe_distance:               # if there is no space -> can't enter
                    too_close = True
                    break

            if not too_close:
                cars.append(
                    Car(
                        position=point,
                        velocity=5,
                        length=random.uniform(2, 4),
                        reaction_speed=random.uniform(0.5, 1.5),
                        exit_target=random.choice(exit_points)
                    )
                )
        
## just for debugging
def is_car_at_end_of_road(car):
    return abs(road_length - car.x) < 10

if __name__ == "__main__":
    #add cars to the simulation
    cars = []
    for _ in range(n_cars):                        
        x = random.uniform(0, road_length)
        v = random.uniform(5, v_max)
        l = random.uniform(2, 4)
        reac = random.uniform(0.30, 2.0)
        exit = random.choice(exit_points)
        cars.append(Car(x, v, l, reac, exit))

    for step in range(200):
        update_velocities(cars)
        update_positions(cars)
        cars = remove_exiting_cars(cars)
        try_add_cars(cars)
        print(step, len(cars), round(sum(car.v for car in cars) / len(cars), 2))
"""