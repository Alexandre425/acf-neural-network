import numpy as np
import random as rng
import math

TIME_SCALE = 3
VEL_SCALE = 6400
MUZZLE_VEL = 400.16194283808 * 39.37
DRAG_COEF = 5.8502517407202e-06
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
    return 0
    #return rng.uniform(MUZZLE_VEL_SCALE*0.3, MUZZLE_VEL_SCALE)

# Random drag coefficient
def random_drag_coef():
    return 0
    #return rng.uniform(DRAG_COEF_SCALE*0.1, DRAG_COEF_SCALE)

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
    def __init__(self):
        self.time = random_time()
        self.dir = random_dir()
        self.vel = random_vel()
        self.muzzle_vel = MUZZLE_VEL
        self.drag_coef = DRAG_COEF

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


def generate_samples(samples):
    X = np.zeros([samples, 6])
    Y = np.zeros([samples, 4])

    i = 0
    while i < samples:
        d = Ballistics()
        d.solve()

        # Determine the intersection of the direction vector with the vertical plane aligned with the target's velocity
        # Hopefully this will make it easier for the NN to digest
        plane_normal = normalize(np.cross(np.array([0,0,1]), d.vel))        # Get the plane normal from the vel and up vector
        length = np.dot(d.pos, plane_normal) / np.dot(d.dir, plane_normal)  # Get the length of the trace from the barrel
        aim_pos = d.dir * length                                            # End pos of the trace is the aim pos
        rel_aim_pos = aim_pos - d.pos

        X[i] = np.array([d.pos[0]/POS_SCALE, d.pos[1]/POS_SCALE, d.pos[2]/POS_SCALE, d.vel[0]/VEL_SCALE, d.vel[1]/VEL_SCALE, d.vel[2]/VEL_SCALE])
        Y[i] = np.array([rel_aim_pos[0]/VEL_SCALE, rel_aim_pos[1]/VEL_SCALE, rel_aim_pos[2]/VEL_SCALE, d.time/TIME_SCALE])
        i += 1
        if (i%(samples*0.01) == 0):
            print(f"Progress: {int(100*i/samples)}%", end='\r')
    
    return X, Y