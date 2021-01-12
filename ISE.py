import time
import sys
import argparse
import numpy as np
def IC_sampling(graph:list,seed_set:list)->int:
    count=len(seed_set)
    activated=np.empty([len(graph)],dtype=np.bool_)
    for i in range(len(graph)):
        activated[i]=False
    activity_set=seed_set.copy()
    for seed in seed_set:
        activated[seed]=True
    while len(activity_set)>0:
        new_activity_set=[]
        for seed in activity_set:
            for neighbor in graph[seed]:
                if not activated[neighbor[0]] and neighbor[1]>=np.random.random_sample():
                    activated[neighbor[0]]=True
                    new_activity_set.append(neighbor[0])
        count+=len(new_activity_set)
        activity_set=new_activity_set
    return count
def LT_sampling(graph:list,seed_set:list):
    if not hasattr(LT_sampling,'reverse_graph'):
        LT_sampling.reverse_graph=[]
        for _ in range(len(graph)):
            LT_sampling.reverse_graph.append([])
        for i in range(len(graph)):
            for neighbor in graph[i]:
                LT_sampling.reverse_graph[neighbor[0]].append((i,neighbor[1]))
    count=len(seed_set)
    activated=np.empty([len(graph)],dtype=np.bool_)
    for i in range(len(graph)):
        activated[i]=False
    thresholds=np.empty([len(graph)],dtype=np.float64)
    for i in range(len(graph)):
        thresholds[i]=np.random.random_sample()
    activity_set=seed_set.copy()
    for seed in seed_set:
        activated[seed]=True
    while len(activity_set)>0:
        new_activity_set=[]
        #print(activity_set)
        for seed in activity_set:
            for neighbor in graph[seed]:
                if not activated[neighbor[0]]:
                    w_total=0.0
                    for n_neighbor in LT_sampling.reverse_graph[neighbor[0]]:
                        if activated[n_neighbor[0]]:
                            w_total+=n_neighbor[1]
                    if w_total>=thresholds[neighbor[0]]:
                        activated[neighbor[0]]=True
                        new_activity_set.append(neighbor[0])
        count+=len(new_activity_set)
        activity_set=new_activity_set
    return count
if __name__ == '__main__':
    start_time=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--file_name',type=str,default='network.txt')
    parser.add_argument('-s','--seed',type=str,default='seeds.txt')
    parser.add_argument('-m','--model',type=str,default='IC')
    parser.add_argument('-t','--time_limit',type=int,default=60)
    args=parser.parse_args()
    file_name=args.file_name
    seed=args.seed
    model=args.model
    time_limit=args.time_limit
    model_function=IC_sampling if model=='IC' else LT_sampling
    file_input=open(file_name)
    content=file_input.readlines()
    first_line=content[0].split(' ')
    n=int(first_line[0])
    m=int(first_line[1])
    graph=[]
    seed_set=[]
    for _ in range(n+1):
        graph.append([])
    for i in range(1,m+1):
        line=content[i].split(' ')
        u=int(line[0])
        v=int(line[1])
        w=float(line[2])
        graph[u].append((v,w))
    file_input=open(seed)
    content=file_input.readlines()
    for seed in content:
        u=int(seed)
        seed_set.append(u)
    total_times=0
    total_num=0
    begin_time=time.time()
    for _ in range(100):
        total_num+=model_function(graph,seed_set)
        total_times+=1
    average_time=(time.time()-begin_time)/100
    search_times=int((start_time+time_limit-time.time()-1)/average_time)
    #print(graph,seed_set)
    for _ in range(search_times):
        total_num+=model_function(graph,seed_set)
        total_times+=1
    #print(time.time()-start_time)
    #print(total_num,total_times)
    print(total_num/total_times)
    sys.stdout.flush()