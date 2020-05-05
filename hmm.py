# HMM (Hidden Markov Model) using the Viterbi Algorithm to predict a sequence.
import random
import numpy as np
# the set of states include two items : {sunny , rainy}
# the observables include a person's mood : {happy , grim}

# Transition Probabilities between the states
p_ss = 0.8
p_sr = 0.2
p_rs = 0.4
p_rr = 0.6

# Emission Probabilities 
p_sh = 0.8
p_sg = 0.2
p_rh = 0.4
p_rg = 0.6

# Initial Probabilities
def initial():
    a = np.array([[1-p_ss,-p_rs], [1,1]])
    b = np.array([0,1])
    x = np.linalg.solve(a, b)
    return (x[0],x[1])

p_s,p_r=initial()

# a helper function which would tell the probability of the current day being sunny or rainy based on person's mood using bayes theorem.
def bayes(current_mood=1):
    if(current_mood==1):
        prob=(p_sh*p_s)/((p_sh*p_s)+(p_rh*p_r))
        #print((p_sh*p_s))
        print('Chances of sunny day when he is happy are:',round(prob,2))
        
    else:
        prob=(p_sg*p_s)/((p_sg*p_s)+(p_rg*p_r))
        print('Chances of sunny day when he is sad are:',round(prob,2))
        
bayes(1)
bayes(0)

# using the viterbi to predict the latent variable using the observables. Here we only use the best possible path for the possible value current states based on past state values. 
def viterbi(moods):
    probabilities = []
    weather = []
    if moods[0] == 'H':
        probabilities.append((p_s*p_sh, p_r*p_rh))
    else:
        probabilities.append((p_s*p_sg, p_r*p_rg))

    for i in range(1,len(moods)):
        yesterday_sunny, yesterday_rainy = probabilities[-1]
        if moods[i] == 'H':
            today_sunny = max(yesterday_sunny*p_ss*p_sh, yesterday_rainy*p_rs*p_sh)
            today_rainy = max(yesterday_sunny*p_sr*p_rh, yesterday_rainy*p_rr*p_rh)
            probabilities.append((today_sunny, today_rainy))
        else:
            today_sunny = max(yesterday_sunny*p_ss*p_sg, yesterday_rainy*p_rs*p_sg)
            today_rainy = max(yesterday_sunny*p_sr*p_rg, yesterday_rainy*p_rr*p_rg)
            probabilities.append((today_sunny, today_rainy))

    for p in probabilities:
        if p[0] > p[1]:
            weather.append('S')
        else:
            weather.append('R')
    print('The predicted sequence:')
    print(weather)
    print('A list of tuples showing the probabilities of the day being sunny and rainy respectively.')
    print(probabilities)        
moods = ['H', 'H', 'G', 'G', 'G', 'H']

viterbi(moods)
