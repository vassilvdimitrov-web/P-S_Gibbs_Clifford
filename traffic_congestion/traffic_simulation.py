import random


road_length = 1500
v_max = 50
entry_points = [100,400, 500, 900, 1300]    #k "in-points" 
exit_points = [250, 500, 850, 1000]         #n "out-points"

safe_distance = 10

class Car:
    def __init__(self, position, velocity, length, reaction_speed, wants_to_exit_at):
        self.x = position
        self.v = velocity
        self.l = length
        self.reac = reaction_speed
        self.exit = wants_to_exit_at


# make certain parts of the road with different lokal speed limits
def local_speed_limit(x):
    if 300 <= x < 500:
        return 20
    elif 900 <= x < 1100:
        return 30
    else:
        return v_max
    
 
#add cars to the simulation
cars = []
for _ in range(100):                        
    x = random.uniform(0, road_length)
    v = random.uniform(5, v_max)
    l = random.uniform(2, 4)
    reac = random.uniform(0.30, 2.0)
    exit = random.choise(exit_points)
    cars.append(Car(x, v, l, reac, exit))


def distance_to_next_car(car, next_car):
    d = next_car.x - car.x
    if d < 0:
        d += road_length
    return d


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

entry_rate = 0.05



