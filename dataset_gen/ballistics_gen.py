import numpy as np
import random as rng
import math

POS_SCALE = 32000
VEL_SCALE = 6400
MUZZLE_VEL_SCALE = 39370
DRAG_COEF_SCALE = 2e-5

# Generate a random position above the origin
def random_pos():
    rand = (np.random.rand(3)-0.5)*2 * POS_SCALE
    if rand[2] < 0:
        rand[2] = -rand[2]
    return rand

# Random velocity
def random_vel():
    rand = (np.random.rand(3)-0.5)*2 * VEL_SCALE
    return rand

# Random muzzle velocity from 200 to 1000 m/s
def random_muzzle_vel():
    return rng.uniform(MUZZLE_VEL_SCALE*0.2, MUZZLE_VEL_SCALE)

# Random drag coefficient
def random_drag_coef():
    return rng.uniform(DRAG_COEF_SCALE*0.1, DRAG_COEF_SCALE)

class Ballistics:
    delta_time = 0.015      # 15ms delta time
    gravity = np.array([0,0,-600])
    correction = 0.2
    threshold = 50
    def __init__(self):
        self.pos = random_pos()
        self.vel = random_vel()
        self.muzzle_vel = random_muzzle_vel()
        self.drag_coef = random_drag_coef()

    def flight(self, dir):
        self.bullet_pos = np.array([0,0,0])
        self.time = 0
        self.bullet_vel = dir * self.muzzle_vel
        delta_dist = -1
        last_dist = np.linalg.norm(self.pos)
        while delta_dist < 0:
            vel_length = np.linalg.norm(self.bullet_vel)
            vel_normalized = self.bullet_vel / vel_length
            vel_sq = vel_length**2
            drag = vel_normalized * self.drag_coef * vel_sq

            next_pos = self.bullet_pos + self.bullet_vel * Ballistics.delta_time
            self.bullet_vel += (Ballistics.gravity - drag) * Ballistics.delta_time

            self.dist = np.linalg.norm(self.pos - next_pos)
            delta_dist = self.dist - last_dist
            last_dist = self.dist
            self.bullet_pos = next_pos
            self.time += Ballistics.delta_time

    def solve(self):
        aim_pos = self.pos 
        direction = self.pos / np.linalg.norm(self.pos)
        self.flight(direction)
        # Get the error of the last iteration
        error_vec = self.pos - self.bullet_pos
        error = np.linalg.norm(error_vec)
        while error > Ballistics.threshold:
            # Compensate the direction to account for the error
            aim_pos += error_vec * Ballistics.correction
            direction = aim_pos / np.linalg.norm(aim_pos)
            self.flight(direction)
            error_vec = self.pos - self.bullet_pos
            error = np.linalg.norm(error_vec)
            print(error_vec)
    

bal = Ballistics()
bal.solve()
print(bal.dist)