import numpy as np
import random as rng
import math

TIME_SCALE = 5
VEL_SCALE = 6400
MUZZLE_VEL_SCALE = 39370
DRAG_COEF_SCALE = 2e-5
POS_SCALE = 32000

length = np.linalg.norm

# Normalize a vector
def normalize(vec):
    return vec / np.linalg.norm(vec)

def random_time():
    return rng.uniform(0, TIME_SCALE)

# Generate a random position above the origin
def random_dir():
    rand = normalize(np.random.rand(3)-0.5)
    if rand[2] < 0:
        rand[2] = -rand[2]
    return rand

# Random velocity
def random_vel():
    return (np.random.rand(3)-0.5)*2 * VEL_SCALE

# Random muzzle velocity from 400 to 1000 m/s
def random_muzzle_vel():
    return rng.uniform(MUZZLE_VEL_SCALE*0.3, MUZZLE_VEL_SCALE)

# Random drag coefficient
def random_drag_coef():
    return rng.uniform(DRAG_COEF_SCALE*0.1, DRAG_COEF_SCALE)

class Ballistics:
    delta_time = 0.015      # 15ms delta time
    gravity = np.array([0,0,-600])
    correction = 0.2
    threshold = 50
    def __init__(self):
        self.time = random_time()
        self.dir = random_dir()
        self.vel = random_vel()
        self.muzzle_vel = random_muzzle_vel()
        self.drag_coef = random_drag_coef()
        print("Aiming at", self.dir, "with flight time of", self.time, "muzzle vel of", self.muzzle_vel/39, "m/s and drag coef of", self.drag_coef)

    def solve(self):
        # Do the bullet flight
        self.bullet_pos = np.array([0,0,0])
        cur_time = 0
        self.bullet_vel = self.dir * self.muzzle_vel
        while cur_time < self.time:
            vel_length = length(self.bullet_vel)
            vel_normalized = normalize(self.bullet_vel)
            vel_sq = vel_length**2
            drag = vel_normalized * self.drag_coef * vel_sq

            self.bullet_pos = self.bullet_pos + self.bullet_vel * Ballistics.delta_time
            self.bullet_vel += (Ballistics.gravity - drag) * Ballistics.delta_time
            cur_time += Ballistics.delta_time
        # After a point in the path is reached (after self.time)
        # Calculate where the target was at firing
        self.pos = self.bullet_pos - self.vel*self.time


samples = int(input("Number of data samples: "))

# File has the following data per line
# INPUTS (normalized) - target position, velocity, muzzle velocity, drag coefficient
# OUTPUTS (normalized) - aim direction, fuze time
f = open("training_data.txt", "a+")

i = 0
while i < samples:
    data = Ballistics()
    data.solve()
    sample = [data.pos/POS_SCALE, data.vel/VEL_SCALE, data.muzzle_vel/MUZZLE_VEL_SCALE, data.drag_coef, data.dir, data.time/TIME_SCALE]

    string = ""
    for x in sample:
        string += str(x) + " "

    f.write(string + "\n")
    #f.write(str(data.pos) + " " + str(data.vel) + " " + str(data.muzzle_vel) + " " + str(data.drag_coef) + " " + str(data.dir) + " "  str(data.time) + "\n")
    i += 1