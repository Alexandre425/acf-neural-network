import numpy as np
import random as rng
import math
import time
from multiprocessing import Process, Queue

THREAD_COUNT = 4
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

def random_sample():
    d = Ballistics()
    d.solve()

    # Simplify the position into two components
    hori_dist = math.sqrt(d.pos[0]**2 + d.pos[1]**2)
    vert_dist = d.pos[2] 
    # Transform the velocity to a local axis
    flat_to_target = np.copy(d.pos)
    flat_to_target[2] = 0
    flat_to_target = normalize(flat_to_target)
    flat_perpendicular = np.array([flat_to_target[1], -flat_to_target[0], 0])   # Flip x and y and invert one to make a perpendicular vector
    rel_vel = np.array([np.dot(flat_to_target, d.vel), np.dot(flat_perpendicular, d.vel), d.vel[2]])    # Localize the velocity (x is towards target, y is towards the right)

    # Determine the intersection of the direction vector with the vertical plane aligned with the target's velocity
    # Hopefully this will make it easier for the NN to digest
    plane_normal = normalize(np.cross(np.array([0,0,1]), d.vel))        # Get the plane normal from the vel and up vector
    length = np.dot(d.pos, plane_normal) / np.dot(d.dir, plane_normal)  # Get the length of the trace from the barrel
    aim_pos = d.dir * length                                            # End pos of the trace is the aim pos
    z_comp = aim_pos[2] - d.bullet_pos[2]                               # Compensation in the z axis relative to impact point

    # Make a list of lists with X and Y
    smp = [[hori_dist/POS_SCALE, vert_dist/POS_SCALE] + (rel_vel/VEL_SCALE).tolist(), [z_comp/VEL_SCALE, d.time/TIME_SCALE]]
    return smp
        

def generate_samples(samples):
    smp_per_thread = int(samples / THREAD_COUNT)
    processes = []
    queue = Queue(THREAD_COUNT)

    # Multiprocess the samples
    processes = [Process(target=lambda smp,q: q.put([random_sample() for _ in range(smp)]), args=(smp_per_thread, queue)).start() for _ in range(THREAD_COUNT)]
    # Get the data
    Data = [queue.get() for _ in range(THREAD_COUNT)]
    unfolded = [sample for sublist in Data for sample in sublist]
    X = np.array([sample[0] for sample in unfolded])
    Y = np.array([sample[1] for sample in unfolded])
    
    return X, Y


if __name__ == '__main__':
    print(random_sample())