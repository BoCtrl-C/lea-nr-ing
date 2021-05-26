"""The script is a modified version of dataset_generation.py from the
dnn_bouncing_balls submodule capable of generating sequences characterized by
different numerosities.
"""


from dnn_bouncing_balls.dataset_generation import (
    SIZE,
    matricize,
    new_speeds,
    norm
)

import argparse
import os
import random

import numpy as np


#----------
# Constants
#----------

ALLW_NUMS = range(1, 6) # allowed numerosities


#----------
# Functions
#----------

def bounce_n(T=128, n=2, r=None, m=None):
    if r is None:
        r = np.array([1.2]*n)
    if m is None:
        m = np.array([1]*n)
    
    X = np.zeros((T, n, 2), dtype='float')
    v = np.random.randn(n, 2)
    v = v/norm(v)*.5
    good_config = False
    while not good_config:
        x = 2 + np.random.rand(n, 2)*8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0: good_config = False
                if x[i][z] + r[i] > SIZE: good_config = False

        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False
    
    eps = .5
    for t in range(T):
        for i in range(n):
            X[t, i] = x[i]
            
        for mu in range(int(1/eps)):
            for i in range(n):
                x[i] += eps*v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0: v[i][z] =  abs(v[i][z])
                    if x[i][z] + r[i] > SIZE: v[i][z] = - abs(v[i][z])

            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r[i] + r[j]:
                        w = x[i] - x[j]
                        w = w/norm(w)

                        v_i = np.dot(w.transpose(), v[i])
                        v_j = np.dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)
                        
                        v[i] += w*(new_v_i - v_i)
                        v[j] += w*(new_v_j - v_j)

    return X

def bounce_vec(res, T=128, r=None, m=None, add_num=False):
    n = random.choice(ALLW_NUMS) # uniform numerosity sampling

    if r is None:
        r = 1.2
    r = np.array([r]*n)
        
    x = bounce_n(T, n, r, m)
    V = matricize(x, res, r)

    seq = V.reshape(T, res**2)
    if add_num:
        # add the numerosity feature to each frame
        seq = np.concatenate((n*np.ones((T, 1)), seq), axis=1)
    return seq


#-----
# Main
#-----

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--res', default=60, type=int)
    parser.add_argument('--frames', default=40, type=int)
    parser.add_argument('--samples', default=6000, type=int)
    parser.add_argument('--r', default=1.2, type=float)
    parser.add_argument('--add_num', action='store_true')
    args = parser.parse_args()

    data_path = args.dir
    res = args.res
    T = args.frames
    N = args.samples
    r = args.r
    add_num = args.add_num

    digits = len(str(N))
    file_name_template = 'sequence_{:0' + str(digits) + 'd}.npy'

    training_path   = os.path.join(data_path, 'training')
    validation_path = os.path.join(data_path, 'validation')
    testing_path    = os.path.join(data_path, 'testing')

    if not os.path.exists(data_path): os.mkdir(data_path)
    if not os.path.exists(training_path): os.mkdir(training_path)
    if not os.path.exists(validation_path): os.mkdir(validation_path)
    if not os.path.exists(testing_path): os.mkdir(testing_path)

    # training set generation
    for i in range(N):
        if i % 200 == 0:
            print('Training set sample {}'.format(i))
        sequence = bounce_vec(res=res, T=T, r=r, add_num=add_num)
        file_path = os.path.join(training_path, file_name_template.format(i))
        np.save(file_path, sequence.astype(np.float32))

    # validation set generation
    for i in range(int(0.2*N)):
        if i % 200 == 0:
            print('Validation set sample {}'.format(i))
        sequence = bounce_vec(res=res, T=T, r=r, add_num=add_num)
        file_path = os.path.join(validation_path, file_name_template.format(i))
        np.save(file_path, sequence.astype(np.float32))

    # testing set generation
    for i in range(int(0.2*N)):
        if i % 200 == 0:
            print('Testing set sample {}'.format(i))
        sequence = bounce_vec(res=res, T=T, r=r, add_num=add_num)
        file_path = os.path.join(testing_path, file_name_template.format(i))
        np.save(file_path, sequence.astype(np.float32))