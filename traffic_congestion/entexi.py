

#suggestions
class EntryNode:
    def __init__(self, linear_pos, node_id):
        self.linear_pos = linear_pos
        self.node_id = node_id
        self.waiting_queue = [] # List of Car objects waiting to spawn

class Car:
    def __init__(self, position, velocity, reaction_speed, exit_node_index):
        self.x = position
        self.v = velocity
        self.reac = reaction_speed
        self.exit_node = exit_node_index
        self.is_waiting_to_enter = False

def is_safe_to_merge(entry_pos, active_cars, road_length, safe_buffer=10):
    """
    Checks if a car can enter at entry_pos without hitting the 'Lag Car'.
    """
    # 1. Find the 'Lag Car' (the car approaching the entry point)
    lag_car = None
    min_lag_dist = float('inf')
        
    def update_velocities_with_exits(cars, nodes_linear_pos):
        for i, car in enumerate(cars):
            # ... your existing gap logic ...
            
            # New Exit Logic:
            dist_to_exit = (car.exit - car.x) % road_length
            if dist_to_exit < 100: # If within 100 units of exit
                # Smoothly reduce speed to an "exit speed" (e.g., 10)
                car.v = max(10, car.v - (2 * car.reac))

                
def can_safely_enter(entry_pos, cars, road_length, safe_distance):
    lag_car = None
    min_lag_dist = float('inf')
    
    lead_car = None
    min_lead_dist = float('inf')

    for car in cars:
        # 1. Check behind (Lag)
        dist_behind = (entry_pos - car.x) % road_length
        if dist_behind < min_lag_dist:
            min_lag_dist = dist_behind
            lag_car = car
            
        # 2. Check ahead (Lead) - We want to know if there's a car right in front of us that we might hit
        dist_ahead = (car.x - entry_pos) % road_length
        if dist_ahead < min_lead_dist:
            min_lead_dist = dist_ahead
            lead_car = car

    # Logic: Safe if Lag car is far enough AND Lead car isn't right on top of us
    if lag_car:
        required_lag_gap = (lag_car.v * lag_car.reac) + safe_distance
        if min_lag_dist < required_lag_gap:
            return False
            
    if lead_car:
        if min_lead_dist < 15:  # Constant small buffer for the car in front
            return False

    return True
    
def is_gap_safe(approaching_car, entry_node_pos, road_length, safe_buffer=10):
    """
    Logic: The approaching car needs time to see the new car and brake.
    Required Distance = (Velocity * Reaction Time) + Physical Buffer
    """
    dist_to_node = (entry_node_pos - approaching_car.x) % road_length
    
    # The 'reac' parameter from your Traffic Sim (0.3 to 2.0)
    # Higher reaction speed value = slower response = needs more distance
    required_dist = (approaching_car.v * approaching_car.reac) + safe_buffer
    
    return dist_to_node > required_dist

def process_node_entries(entry_nodes, active_cars, road_length):
    """
    Checks all entry nodes. If a car is waiting and the road is safe,
    it moves the car from the queue to the active road.
    """
    for node in entry_nodes:
        if node.waiting_queue:
            # Check the car at the front of the line
            next_car = node.waiting_queue[0]
            
            if can_safely_enter(node.linear_pos, active_cars, road_length, safe_distance=15):
                # Remove from queue and mark as active
                entering_car = node.waiting_queue.pop(0)
                entering_car.is_waiting_to_enter = False
                entering_car.x = node.linear_pos
                active_cars.append(entering_car)

def handle_exits(active_cars, road_length, exit_threshold=5):
    """
    Removes cars from the active list if they are close enough to their exit node.
    """
    # We iterate backwards to safely remove items from the list while looping
    for i in range(len(active_cars) - 1, -1, -1):
        car = active_cars[i]
        
        # Calculate circular distance to target
        dist = abs(car.x - car.exit)
        dist = min(dist, road_length - dist)
        
        if dist < exit_threshold:
            active_cars.pop(i)
            # You could add a 'score' or 'counter' here for your group stats
            
def generate_entry_demand(entry_nodes, exit_points, probability=0.05):
    """
    Randomly adds new cars to the waiting queues of entry nodes.
    This simulates people 'arriving' at the intersection.
    """
    import random
    for node in entry_nodes:
        if len(node.waiting_queue) < max_q:
            if random.random() < probability:
                new_car = Car(
                position=node.linear_pos, 
                velocity=0, # Starts at 0 while waiting
                reaction_speed=random.uniform(0.5, 1.5),
                exit_node_index=random.choice(exit_points)
            )
            new_car.is_waiting_to_enter = True
            node.waiting_queue.append(new_car)