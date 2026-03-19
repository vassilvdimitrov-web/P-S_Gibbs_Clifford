import random


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
    """
    if 300 <= x < 500:
        return 20
    elif 900 <= x < 1100:
        return 30
    else:
    """
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