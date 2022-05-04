import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import json
#%matplotlib inline

graph = {}
node_set = []
filenames = ['Facebook_dataset2.txt']
rw_steps = 20000
num_of_run = 1000

def preprocess(filename):
    global graph,node_set
    node_list = []
    data = pd.read_csv(filename,sep = ' ',index_col = False)
    data = data[data['nodeID1'] != data['nodeID2']]
    for i in range(len(data)):
        edge = data.iloc[i,:]
        node1 = int(edge['nodeID1'])
        node2 = int(edge['nodeID2'])
        layer = int(edge['layerID'])-1
        if node1 not in graph:
            graph[node1] = [[],[]]
        if node2 not in graph[node1][layer]:
            graph[node1][layer].append(node2)
        if node2 not in graph:
            graph[node2] = [[],[]]
        if node1 not in graph[node2][layer]:
            graph[node2][layer].append(node1)
        node_list.append(node1)
        node_list.append(node2)
    node_set = list(set(node_list))

def get_initial_state_1():
    u = random.choice(node_set)
    while len(graph[u][0]) == 0:
        u = random.choice(node_set)
    v = random.choice(graph[u][0])
    w = random.choice(graph[v][0])
    return [u,v,w,1]

def get_next_state_rw1(state):
    u,v,w,state_type = state
    if state_type == 1:
        b = len(graph[w][0])
        r = len(graph[w][1])
        if random.random() < b/(r+b):
            next_node = random.choice(graph[w][0])
            return [v,w,next_node,1]
        else:
            next_node = random.choice(graph[w][1])
            return [v,w,next_node,2]
    else:
        next_node = random.choice(graph[v][0])
        return [u,v,next_node,1]

def induce(state):
    u = state[0]
    v = state[1]
    w = state[2]
    if len(set([u,v,w])) != 3:
        return 0
    edge_num_l1 = 0
    edge_num_l2 = 0
    public_edge_num = 0
    if v in graph[u][0]:
        edge_num_l1 = edge_num_l1 + 1
    if w in graph[u][0]:
        edge_num_l1 = edge_num_l1 + 1
    if v in graph[w][0]:
        edge_num_l1 = edge_num_l1 + 1
    if v in graph[u][1]:
        edge_num_l2 = edge_num_l2 + 1
    if w in graph[u][1]:
        edge_num_l2 = edge_num_l2 + 1
    if v in graph[w][1]:
        edge_num_l2 = edge_num_l2 + 1

    if v in graph[u][0] and v in graph[u][1]:
        public_edge_num = public_edge_num + 1
    if w in graph[u][0] and w in graph[u][1]:
        public_edge_num = public_edge_num + 1
    if v in graph[w][0] and v in graph[w][1]:
        public_edge_num = public_edge_num + 1

    edge_num = edge_num_l1 + edge_num_l2

    if edge_num < 2:
        return 0
    if edge_num == 2:
        if public_edge_num == 1:
            return 0
        else:
            assert public_edge_num == 0
            if edge_num_l1 == 2:
                return 1
            elif edge_num_l2 == 2:
                return 15
            else:
                assert edge_num_l1 == 1
                return 2

    elif edge_num == 3:
        if public_edge_num == 1:
            if edge_num_l1 == 2:
                return 3
            else:
                return 4
        else:  # triangle
            assert public_edge_num == 0
            if edge_num_l1 == 3:
                return 6
            if edge_num_l1 == 2:
                return 7
            if edge_num_l1 == 1:
                return 8
            else:
                return 16

    elif edge_num == 4:
        if public_edge_num == 2:
            return 5
        else:
            assert public_edge_num == 1
            if edge_num_l1 == 3:
                return 9
            elif edge_num_l2 == 3:
                return 10
            else:
                assert edge_num_l1 == 2 and edge_num_l2 == 2
                return 11

    elif edge_num == 5:
        if edge_num_l1 == 3:
            return 12
        else:
            return 13

    else:
        assert edge_num == 6
        return 14


def stationary_distribution_rw1(state):
    u,v,w,state_type = state
    if state_type == 1:
        return 1/len(graph[v][0])
    else:
        return 1/(len(graph[v][0]) + len(graph[v][1]))

def alpha_rw1(index):
    alpha = [1,2,1,3,1,4,6,4,2,8,2,5,10,6,12,1,1]
    return alpha[index]

def rw1(rw_steps):
    C = np.zeros(17)
    state = get_initial_state_1()
    t = 1
    while t <= rw_steps:
        i = induce(state)
        C[i] += 1/(alpha_rw1(i)*stationary_distribution_rw1(state))
        state = get_next_state_rw1(state)
        t += 1
    C = C[1:15]
    C = [C[0],C[1],C[2],C[3],C[4],C[5],sum(C[6:])]
    d = C / sum(C)
    return d

def get_next_state_rw1s2(state):
    u,v,w,state_type = state
    if state_type == 1:
        b = len(graph[w][0])
        r = len(graph[w][1])
        if random.random() < b/(r+b):
            next_node = random.choice(graph[w][0])
            return [v,w,next_node,1]
        else:
            next_node = random.choice(graph[w][1])
            return [v,w,next_node,2]
    elif state_type == 2:
        next_node = random.choice(graph[w][1])
        return [v,w,next_node,3]
    else:
        left_node = random.choice(graph[u][0])
        right_node = random.choice(graph[u][0])
        return [left_node,u,right_node,1]

def stationary_distribution_rw1s2(state):
    u,v,w,state_type = state
    if state_type == 1:
        return 1/len(graph[v][0])
    elif state_type == 2:
        return 1/(len(graph[v][0]) + len(graph[v][1]))
    else:
        return len(graph[u][0]) / ((len(graph[u][0])+len(graph[u][1]))*len(graph[v][1]))

def alpha_rw1s2(index):
    alpha = [1, 2, 1, 3, 3, 6, 6, 4, 4, 8, 8, 7, 12, 12, 18, 1, 1]
    return alpha[index]

def rw1s2(rw_steps):
    C = np.zeros(17)
    state = get_initial_state_1()
    t = 1
    while t <= rw_steps:
        i = induce(state)
        C[i] += 1/(alpha_rw1s2(i)*stationary_distribution_rw1s2(state))
        state = get_next_state_rw1s2(state)
        t += 1
    C = C[1:15]
    C = [C[0],C[1],C[2],C[3],C[4],C[5],sum(C[6:])]
    d = C / sum(C)
    return d

def get_next_state_rw1m(state):
    u,v,w,state_type = state
    if state_type == 1:
        b = len(graph[w][0])
        r = len(graph[w][1])
        if random.random() < b/(r+b):
            next_node = random.choice(graph[w][0])
            return [v,w,next_node,1]
        else:
            next_node = random.choice(graph[w][1])
            return [v,w,next_node,2]
    elif state_type == 2:
        b = len(graph[v][0])
        ry = len(graph[w][1])
        if random.random() < ry / (ry + b):
            next_node = random.choice(graph[w][1])
            return [v,w,next_node,3]
        else:
            next_node = random.choice(graph[v][0])
            return [u,v,next_node,1]
    else:
        left_node = random.choice(graph[u][0])
        right_node = random.choice(graph[u][0])
        return [left_node,u,right_node,1]

def stationary_distribution_rw1m(state):
    u,v,w,state_type = state
    if state_type == 1:
        return 1/len(graph[v][0])
    elif state_type == 2:
        return 1/(len(graph[v][0]) + len(graph[v][1]))
    else:
        return len(graph[u][0]) / ((len(graph[u][0])+len(graph[u][1]))*(len(graph[v][1])+len(graph[u][0])))

def alpha_rw1m(index):
    alpha = [1, 2, 1, 3, 3, 6, 6, 4, 4, 8, 8, 7, 12, 12, 18, 1, 1]
    return alpha[index]

def rw1m(rw_steps):
    C = np.zeros(17)
    state = get_initial_state_1()
    t = 1
    while t <= rw_steps:
        i = induce(state)
        C[i] += 1/(alpha_rw1m(i)*stationary_distribution_rw1m(state))
        state = get_next_state_rw1m(state)
        t += 1
    C = C[1:15]
    C = [C[0],C[1],C[2],C[3],C[4],C[5],sum(C[6:])]
    d = C / sum(C)
    return d

def get_initial_state_rw1nr():
    u = random.choice(node_set)
    num_neighbor = len(graph[u][0]) + len(graph[u][1])
    while num_neighbor == 0:
        u = random.choice(node_set)
        num_neighbor = len(graph[u][0]) + len(graph[u][1])
    neighbor_u = graph[u][0] + graph[u][1]
    v = random.choice(neighbor_u)
    neighbor_v = graph[v][0] + graph[v][1]
    w = random.choice(neighbor_v)
    return [u,v,w]

def get_next_state_rw1nr(state):
    u,v,w = state
    neighor_w = graph[w][0] + graph[w][1]
    next_node = random.choice(neighor_w)
    return [v,w,next_node]

def stationary_distribution_rw1nr(state):
    u,v,w = state
    num_neighbor_v = len(graph[v][0]) + len(graph[v][1])
    return 1/num_neighbor_v

def alpha_rw1nr(index):
    alpha = [1,2,2,4,4,8,6,6,6,10,10,10,16,16,24,1,1]
    return alpha[index]

def get_exact_count(filename):
    name = filename.split('.')[0] + '_exact_counts.txt'
    data = pd.read_csv(name,'\t')
    exact_count = list(data.sort_values('Mapping')['Frequency'])[:14]
    exact_count = np.array([exact_count[0],exact_count[1],exact_count[2],exact_count[3],exact_count[4],exact_count[5],sum(exact_count[6:])])
    return np.array(exact_count / sum(exact_count))

def rw1nr(rw_steps):
    C = np.zeros(17)
    state = get_initial_state_rw1nr()
    t = 1
    while t <= rw_steps:
        i = induce(state)
        C[i] += 1/(alpha_rw1nr(i)*stationary_distribution_rw1nr(state))
        state = get_next_state_rw1nr(state)
        t += 1
    C = C[1:15]
    C = [C[0],C[1],C[2],C[3],C[4],C[5],sum(C[6:])]
    d = C / sum(C)
    return d

def get_initial_state_2():
    u = random.choice(node_set)
    while len(graph[u][0]) == 0:
        u = random.choice(node_set)
    v = random.choice(graph[u][0])
    w = random.choice(graph[v][0])
    return [(u,v),(v,w),1]

def get_next_state_rw2(state):
    edge_1,edge_2, state_type = state
    v = [x for x in edge_1 if x in edge_2][0]
    u = [x for x in edge_1 if x != v][0]
    w = [x for x in edge_2 if x != v][0]
    if state_type == 1:
        blue_candidate = [(v,x) for x in graph[v][0] if x != w] + [(w,x) for x in graph[w][0] if x != v]
        red_candidate = [(v,x) for x in graph[v][1] if x != w] + [(w,x) for x in graph[w][1] if x != v]
        if w in graph[v][1]:
            red_candidate.append((v,w))
        b = len(blue_candidate)
        r = len(red_candidate)
        if random.random() < b/(r+b):
            next_edge = random.choice(blue_candidate)
            return [(v,w),next_edge,1]
        else:
            next_edge = random.choice(red_candidate)
            return [(v,w),next_edge,2]
    else:
        candidate = [(u,x) for x in graph[u][0] if x != v] + [(v,x) for x in graph[v][0] if x != u]
        next_edge = random.choice(candidate)
        return [(u,v),next_edge,1]

def induce_2(state):
    edge_1,edge_2,state_type = state
    v = [x for x in edge_1 if x in edge_2][0]
    u = [x for x in edge_1 if x != v][0]
    w = [x for x in edge_2 if x != v][0]

    if len(set([u,v,w])) != 3:
        return 0
    edge_num_l1 = 0
    edge_num_l2 = 0
    public_edge_num = 0
    if v in graph[u][0]:
        edge_num_l1 = edge_num_l1 + 1
    if w in graph[u][0]:
        edge_num_l1 = edge_num_l1 + 1
    if v in graph[w][0]:
        edge_num_l1 = edge_num_l1 + 1
    if v in graph[u][1]:
        edge_num_l2 = edge_num_l2 + 1
    if w in graph[u][1]:
        edge_num_l2 = edge_num_l2 + 1
    if v in graph[w][1]:
        edge_num_l2 = edge_num_l2 + 1

    if v in graph[u][0] and v in graph[u][1]:
        public_edge_num = public_edge_num + 1
    if w in graph[u][0] and w in graph[u][1]:
        public_edge_num = public_edge_num + 1
    if v in graph[w][0] and v in graph[w][1]:
        public_edge_num = public_edge_num + 1

    edge_num = edge_num_l1 + edge_num_l2

    if edge_num < 2:
        return 0
    if edge_num == 2:
        if public_edge_num == 1:
            return 0
        else:
            assert public_edge_num == 0
            if edge_num_l1 == 2:
                return 1
            elif edge_num_l2 == 2:
                return 15
            else:
                assert edge_num_l1 == 1
                return 2

    elif edge_num == 3:
        if public_edge_num == 1:
            if edge_num_l1 == 2:
                return 3
            else:
                return 4
        else:  # triangle
            assert public_edge_num == 0
            if edge_num_l1 == 3:
                return 6
            if edge_num_l1 == 2:
                return 7
            if edge_num_l1 == 1:
                return 8
            else:
                return 16

    elif edge_num == 4:
        if public_edge_num == 2:
            return 5
        else:
            assert public_edge_num == 1
            if edge_num_l1 == 3:
                return 9
            elif edge_num_l2 == 3:
                return 10
            else:
                assert edge_num_l1 == 2 and edge_num_l2 == 2
                return 11

    elif edge_num == 5:
        if edge_num_l1 == 3:
            return 12
        else:
            return 13

    else:
        assert edge_num == 6
        return 14

def stationary_distribution_rw2(state):
    edge_1,edge_2,state_type = state
    v = [x for x in edge_1 if x in edge_2][0]
    u = [x for x in edge_1 if x != v][0]
    w = [x for x in edge_2 if x != v][0]

    if state_type == 1:
        return 1
    else:
        blue_neighbor = [(u,x) for x in graph[u][0] if x != v] + [(v,x) for x in graph[v][0] if x != u]
        red_neighbor = [(u,x) for x in graph[u][1] if x != v] + [(v,x) for x in graph[v][1] if x != u]
        if u in graph[v][1]:
            red_neighbor.append((u,v))
        b = len(blue_neighbor)
        r = len(red_neighbor)
        return b/(b + r)
    
def alpha_rw2(index):
    alpha = [1,2,1,3,1,4,6,4,2,8,2,5,10,6,12,1,1]
    return alpha[index]


def rw2(rw_steps):
    C = np.zeros(17)
    state = get_initial_state_2()
    t = 1
    while t <= rw_steps:
        i = induce_2(state)
        C[i] += 1/(alpha_rw2(i)*stationary_distribution_rw2(state))
        state = get_next_state_rw2(state)
        t += 1
    C = C[1:15]
    C = [C[0],C[1],C[2],C[3],C[4],C[5],sum(C[6:])]
    d = C / sum(C)
    return d



def random_walk(name_alg, num_of_run, rw_steps,exact_count,filename):
        eas_curve = []
        estimate_sum = np.zeros(7)
        estimate_list = []
        re = []
        for num in range(1,num_of_run+1):
            estimate_concentration = name_alg(rw_steps)
            estimate_list.append(estimate_concentration)
            print('{} Round:'.format(name_alg.__name__), num)
            print('estimate_concentration_{}:'.format(name_alg.__name__),estimate_concentration)
            estimate_sum += estimate_concentration
            eas = np.abs(estimate_sum / num - exact_count)/exact_count
            print('eas:', eas)
            eas_curve.append(eas)
            re.append(np.abs(estimate_concentration - exact_count) / exact_count)
        mre_1000 = [np.mean([x[i] for x in re]) for i in range(7)]
        
        nrmse = []
        for rw_step in range(2000,rw_steps + 1,1000):
            temp = []
            for num in range(1,num_of_run + 1):
                estimate_concentration = name_alg(rw_step)
                print('Round {} {}'.format(rw_step, num))
                temp.append(estimate_concentration)
            nrmse_temp = np.zeros(7)
            for i in range(7):
                estimate_i = np.array([x[i] for x in temp])
                var_i = np.var(estimate_i)
                mean_i = np.mean(estimate_i)
                nrmse_temp[i] = np.sqrt(var_i + (mean_i - exact_count[i])**2) / exact_count[i]
            nrmse.append(nrmse_temp)
        
        
        plt.subplot(1,3,1)   
        for j in range(7):
            plt.plot(range(100,num_of_run+1),[x[j] for x in eas_curve[99:]])
            plt.title('{} {}'.format(name_alg.__name__,filename))
            plt.ylabel('EAS')
            plt.xlabel('num_runs')
        
        plt.subplot(1,3,2)
        x = np.array(range(1,8))
        tick_label = list(range(1,8))
        bar_width = 0.5
        plt.bar(x, mre_1000,bar_width)
        plt.xticks(x+bar_width/2, tick_label)
        plt.ylabel('MRE')
        plt.title('{} {}'.format(name_alg.__name__,filename))
        
        plt.subplot(1,3,3)
        plt.plot(range(2000,rw_steps+1,1000),nrmse)
        plt.ylabel('NRMSE')
        plt.xlabel('rw_steps')
        plt.title('{} {}'.format(name_alg.__name__,filename))
        
        experiment_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        store = [list(eas_curve),list(mre_1000),list(nrmse),list(estimate_list)]
        
        filename_2 = filename.split('/')[-1]
        file_name = 'result/{}_{}_{}.npy'.format(filename_2,name_alg.__name__,experiment_time)
        
        np.save(file_name, store)
            
            
            
def main():
    for filename in filenames:
        exact_count = get_exact_count(filename)
        print('exact_count',exact_count)
        
        print('Preprocess begins')
        preprocess(filename)
        print('Preprocess finishes')
        
        plt.figure(1)
        random_walk(rw1,num_of_run,rw_steps,exact_count,filename)
        
        plt.figure(2)
        random_walk(rw1s2,num_of_run,rw_steps,exact_count,filename)
        
        
        plt.figure(3)
        random_walk(rw1m,num_of_run,rw_steps,exact_count,filename)
        
        plt.figure(4)
        random_walk(rw2,num_of_run,rw_steps,exact_count,filename)
        
        plt.figure(5)
        random_walk(rw1nr,num_of_run,rw_steps,exact_count,filename)