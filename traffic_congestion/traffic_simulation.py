import random


road_length = 1500
v_max = 40
#entry_points = [100,400, 500, 900, 1300]    #k "in-points" 
#exit_points = [250, 500, 850, 1000]         #n "out-points"

safe_distance = 10

class Car:
    def __init__(self, position, velocity, length, reaction_speed, enter):
        self.x = position
        self.v = velocity
        self.l = length
        self.reac = reaction_speed
        #self.go_in = enter                  #is the car goint to enter

cars = []
for _ in range(100):
    x = random.uniform(0, road_length)
    v = random.uniform(5, v_max)
    l = random.uniform(2, 4)
    reac = random.uniform()
    #go_in = random.choise(0,1)
    cars.append(Car(x, v, l, reac))

cars.sort(key=lambda car: car.x)            # sort the cars with respect to position


def distance_to_next_car(car, nextCar):


for i in range(len(cars)):
    car = cars[i]



