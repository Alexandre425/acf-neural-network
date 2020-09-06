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

class SphericalCoord:
    """
    Angles expressed in radians
    """
 
    def __init__(self, theta, phi, distance):
        self.theta = theta
        self.phi = phi
        self.distance = distance
 
    def __str__(self):
        return f"{self.theta} {self.phi} {self.distance}"
 
    @classmethod
    def from_cartesian(cls, x: float, y: float, z: float):
        xy_len = math.sqrt(x*x + y*y)
        xyz_len = math.sqrt(x*x + y*y + z*z)
        theta = math.acos(x / xy_len)
        phi = math.asin(z / xyz_len)
        distance = xyz_len
        return cls(theta, phi, distance)
 
    @classmethod
    def from_nparray(cls, arr):
        return cls.from_cartesian(arr[0], arr[1], arr[2])
    
    def normalize(self, scale):
        self.theta /= 2*np.pi
        self.phi /= 2*np.pi
        self.distance /= scale
        return self

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


def generate_samples():
    samples = int(input("Number of data samples: "))
    X = np.zeros([samples, 8])
    Y = np.zeros([samples, 3])

    i = 0
    while i < samples:
        data = Ballistics()
        data.solve()

        data.pos = SphericalCoord.from_nparray(data.pos).normalize(POS_SCALE)
        data.vel = SphericalCoord.from_nparray(data.vel).normalize(VEL_SCALE)
        data.dir = SphericalCoord.from_nparray(data.dir).normalize(1)

        X[i] = np.array([data.pos.theta, data.pos.phi, data.pos.distance, data.vel.theta, data.vel.phi, data.vel.distance, data.muzzle_vel/MUZZLE_VEL_SCALE, data.drag_coef])
        Y[i] = np.array([data.dir.theta - data.pos.theta, data.dir.phi - data.pos.phi, data.time/TIME_SCALE])
        i += 1
        if (i%(samples*0.01) == 0):
            print(f"Progress: {int(100*i/samples)}%", end='\r')
    
    return X, Y