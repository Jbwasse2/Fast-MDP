import sys
import mdptoolbox.example
import mdptoolbox
import numpy as np
from operator import mul
from functools import reduce

P, R = mdptoolbox.example.forest()
n=2
r_Ps = [P for i in range(n)]
r_Rs = [R for i in range(n)]
big_P = reduce(np.kron, r_Ps)
big_R = reduce(np.kron, r_Rs)
pi = mdptoolbox.mdp.PolicyIteration(big_P, big_R, 0.9)
pi.run()
print(pi.policy)
print(pi.V)



Ps = [P,P]
Rs = []
#pi_kron = mdptoolbox.kronPolicyMDP.KronPolicyMDP(P, R, 0.9)
#pi_kron.run()

