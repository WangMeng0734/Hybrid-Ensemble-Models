import numpy as np
import opfunu
from matplotlib import pyplot as plt
from matplotlib import cm
import math
import time
import pandas as pd

# initialization
def initialization(N, Dim, UB, LB):
    B_no = len(UB)
    X = np.zeros((N, Dim))

    if B_no == 1:
        X = np.random.rand(N, Dim) * (UB - LB) + LB
    elif B_no > 1:
        for i in range(Dim):
            Ub_i = UB[i]
            Lb_i = LB[i]
            X[:, i] = np.random.rand(N) * (Ub_i - Lb_i) + Lb_i

    return X

def HOA(N, T, LB, UB, Dim, F_obj, identifier):
    
    Best_P = np.zeros(Dim)
    Best_FF = float('inf')
    X = initialization(N, Dim, UB, LB)
    X_new = X.copy()
    Ffun = np.zeros(N)
    Ffun_new = np.zeros(N)
    t = 1
    conv = []


    # Find a team leader
    for i in range(N):
        X_new[i, :] = X[i, :]
        Ffun_new[i] = F_obj(X_new[i, :])
        Ffun[i] = Ffun_new[i]
        if Ffun[i] < Best_FF:
            Best_FF = Ffun[i]
            Best_P = X[i, :]

    while t < T + 1:
        start_time = time.time()  
        for i in range(N):      
            # Tilt Angle
            theta = np.random.randint(low=0, high=50, size=1)
            # slope
            s = math.tan(theta)
            # Scan Factor
            SF = np.random.uniform(low=1.0, high=2.0, size=1)
            # Initial Speed
            Vel = 6*math.exp( -3.5*abs(s+0.05) )
            # Update speed
            newVel = X_new[i, :].copy()
            newVel = Vel + np.random.randn(1, Dim)*(Best_P - SF*X_new[i, :])

            # Update Location
            X_new[i, :] = X_new[i, :] + newVel

            # Scope Specification
            F_UB = X_new[i, :] > UB
            F_LB = X_new[i, :] < LB
            X_new[i, :] = (X_new[i, :] * ~(F_UB + F_LB)) + UB * F_UB + LB * F_LB

            Ffun_new[i] = F_obj(X_new[i, :])
            if Ffun_new[i] < Ffun[i]:
                X[i, :] = X_new[i, :]
                Ffun[i] = Ffun_new[i]
            if Ffun[i] < Best_FF:
                Best_FF = Ffun[i]
                Best_P = X[i, :]
            
        end_time = time.time()  
        iteration_time = end_time - start_time  
        total_time += iteration_time  
        average_time = total_time / t  

        iteration_times.append(iteration_time)
        GbestScores.append(Best_FF)
        GbestPositions.append(Best_P.tolist())

        if t % 1 == 0:
            print('At iteration', t, 'the best solution fitness is {:.8f}'.format(Best_FF))
        conv.append(Best_FF)
        
        print(f"{t}th iteration time: {iteration_time:.4f}秒")  
        print(f"Average iteration time: {average_time:.4f}秒")  

        t += 1

    return Best_FF, Best_P, conv


