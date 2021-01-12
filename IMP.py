import time,sys,os,argparse,math,random,heapq,multiprocessing
'''
class Processor(multiprocessing.Process):
    def __init__(self,graph,node_num,rr_function):
        super(Processor,self).__init__(target=self.start)
        self.input_queue=multiprocessing.Queue()
        self.output_queue=multiprocessing.Queue()
        self.R=[]
        self.graph=graph
        self.node_num=node_num
        self.rr_function=rr_function
    def run(self):
        while True:
            theta=self.input_queue.get()
            count=0
            while count<theta:
                v=random.randint(1,self.node_num)
                rr=self.rr_function(self.graph,v)
                self.R.append(rr)
                count+=1
            self.output_queue.put(self.R)
            self.R=[]
processors=[]
processor_num=8
'''
def rr_ic(graph:list,node:int)->list:
    activity_set=[node]
    activity_nodes={node}
    while len(activity_set)>0:
        new_activity_set=[]
        for seed in activity_set:
            if seed in graph:
                for neighbor in graph[seed]:
                    if neighbor[0] not in activity_nodes and random.random()<=neighbor[1]:
                        activity_nodes.add(neighbor[0])
                        new_activity_set.append(neighbor[0])
        activity_set=new_activity_set
    activity_nodes=list(activity_nodes)
    return activity_nodes
def rr_lt(graph:list,node:int)->list:
    activity_nodes={node}
    activity_set=node
    while activity_set>0:
        new_activity_set=0
        if activity_set not in graph:
            break
        neighbors=graph[activity_set]
        new_node=random.sample(neighbors,1)[0][0]
        if new_node not in activity_nodes:
            activity_nodes.add(new_node)
            new_activity_set=new_node
        activity_set=new_activity_set
    activity_nodes=list(activity_nodes)
    return activity_nodes
def IMM(graph:list,k:int,eps,l)->list:
    global node_num
    l=l*(1+ math.log(2)/math.log(node_num))
    R=sampling(graph,k,eps,l)
    Sk,_=node_selection(graph,R,k)
    return Sk
def node_selection(graph:list,R:list,k:int):
    global node_num
    node_rr_set=[set() for _ in range(node_num+1)]
    index=0
    for rr in R:
        for rr_node in rr:
            node_rr_set[rr_node].add(index)
        index+=1
    max_heap=[]
    for i in range(node_num+1):
        max_heap.append([-len(node_rr_set[i]),i])
    heapq.heapify(max_heap)
    visited_rr=set()
    Sk=[]
    while len(Sk)<k:
        top=heapq.heappop(max_heap)
        tmp=node_rr_set[top[1]]-visited_rr
        if len(tmp)==len(node_rr_set[top[1]]):
            visited_rr|=tmp
            Sk.append(top[1])
        else:
            node_rr_set[top[1]]=tmp
            top[0]=-len(tmp)
            heapq.heappush(max_heap,top)
    return Sk,len(visited_rr)/len(R)
def log_c(n:int,k:int):
    res=0
    for i in range(n-k+1,n+1):
        res+=math.log(i)
    for i in range(1,k+1):
        res-=math.log(i)
    return res
def sampling(graph:list,k:int,eps,l)->list:
    '''
    global rr_function,node_num,processors,processor_num
    R=[]
    LB=1
    eps_p=eps*math.sqrt(2)
    n=node_num
    lambda_p=((2+2*eps_p/3)*(log_c(n,k)+l*math.log(n)+math.log(math.log2(n)))*n)/(eps_p**2)
    for i in range(1,int(math.log2(n))):
        x=n/(2**i)
        theta=lambda_p/x
        #for _ in range(int(theta+1)):
        #    v=random.randint(1,n)
        #    R.append(rr_function(graph,v))
        for processor in processors:
            processor.input_queue.put((theta-len(R))/processor_num)
        for processor in processors:
            R+=processor.output_queue.get()
        Si,F=node_selection(graph,R,k)
        if n*F>=(1+eps_p)*x:
            LB=n*F/(1+eps_p)
            break
    alpha=math.sqrt(l*math.log(n)+math.log(2))
    beta=math.sqrt((1-1/math.e)*(log_c(n,k)+l*math.log(n)+math.log(2)))
    lambda_star=2*n*(((1-1/math.e)*alpha+beta)**2)*(eps**-2)
    theta=lambda_star/LB
    while theta>=len(R):
        v=random.randint(1,n)
        R.append(rr_function(graph,v))
    return R
    '''
    global rr_function,node_num,start_time,time_limit
    R=[]
    if rr_function==rr_ic:
        fenmu=2
        fenzi=1
    else:
        fenmu=3
        fenzi=2
    while (time.time()-start_time)*fenmu<=time_limit*fenzi:
        v=random.randint(1,node_num)
        R.append(rr_function(graph,v))
    return R
    
def ISE_test(network_file_path, S_k_star, model, time_budget):
    seeds_file = 'caiyiwen.txt'
    with open(seeds_file, 'w') as f:
        for seed in S_k_star:
            f.write('{}\n'.format(seed))
    print('ISE test start')
    start_time = time.time()
    print('ISE:', end=' ')
    os.system('python {} -i {} -s {} -m {} -t {}'.format('./ISE.py', network_file_path,
                                                          seeds_file, model, time_budget))
    end_time = time.time()
    print('Total time of ISE:', end_time - start_time)
if __name__ == '__main__':
    start_time=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--file_name',type=str,default='network.txt')
    parser.add_argument('-k','--k',type=int,default=5)
    parser.add_argument('-m','--model',type=str,default='IC')
    parser.add_argument('-t','--time_limit',type=int,default=60)
    args=parser.parse_args()
    file_name=args.file_name
    k=args.k
    model=args.model
    time_limit=args.time_limit
    rr_function=rr_ic if model=='IC' else rr_lt
    file_input=open(file_name)
    content=file_input.readlines()
    first_line=content[0].split(' ')
    n=int(first_line[0])
    m=int(first_line[1])
    node_num=n
    graph={}
    for i in range(1,m+1):
        line=content[i].split(' ')
        u=int(line[0])
        v=int(line[1])
        w=float(line[2])
        if v not in graph:
            graph[v]=[]
        graph[v].append((u,w))  # Reversed Graph
    '''
    for i in range(processor_num):
        processors.append(Processor(graph,node_num,rr_function))
        processors[i].start()
    '''
    seeds=IMM(graph,k,0.1,1)
    for seed in seeds:
        print(seed)
        #pass
    '''
    end_time=time.time()
    print(end_time-start_time)
    ISE_test(file_name,seeds,model,time_limit)
    '''
    sys.stdout.flush()
    #os._exit(0)