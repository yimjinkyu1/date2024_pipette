import random
import math
import time
import json
import copy
import numpy as np
import pandas as pd

    
class MetaHeuristics():
    """ Metaheuristics Module for Simulated Annealing"""
    def __init__(self, read_file, output_file, node_size, pp_group, dp_group, inter_dp_size, inner_dp_size, pp_msg_size, dp_msg_size, initial_temp, alpha, time_limit):
        np.random.seed(0)
        random.seed(0)

        
        # Read nccl data
        with open(read_file, "r") as f:
            data = json.load(f)

        self.node_data = data["gpus"]
        self.node_size = len(self.node_data)
        self.node_index = {i:self.node_data[i] for i in range(self.node_size)}


        float_array = []
        for row in data["bandwidth"]:
            float_row =[float(element) for element in row]
            float_array.append(float_row)
        #self.sendrecv_bw_data = np.array(float(data["bandwidth"]))
        self.sendrecv_bw_data = np.array(float_array)
        
        self.read_file = read_file
        self.output_file = output_file
        self.node_size = node_size
        self.pp_group = pp_group
        self.dp_group = dp_group
        self.inter_dp_size = inter_dp_size
        self.inner_dp_size = inner_dp_size
        self.pp_msg_size = pp_msg_size #GB
        self.dp_msg_size = dp_msg_size #GB
        
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.time_limit = time_limit
        self.n_top_low = 5

        ### TEST params ###
        # min_val = 5
        # max_val = 200
        # self.sendrecv_bw_data = np.random.randint(low=min_val, high=max_val, size=(self.node_size, self.node_size)) #GB/s
        
        
        self.node_index = {i:f'agpu{i}' for i in range(self.node_size)}
        
        
        # Variable scaling factors
        self.millisec_scaling = 10**3 # scaling factor (seconds to milliseconds)

        self.latency = 1 / self.sendrecv_bw_data         


    """ Utils for sorting the solution """
    def two_opt(self, state):
        "Inverses the order of cities in a route between node one and node two"
        idx_i, idx_j = random.sample(range(self.node_size), 2)
        state[min(idx_i, idx_j):max(idx_i, idx_j)] = state[min(idx_i, idx_j):max(idx_i, idx_j)][::-1]
        return state

    def insert(self, state):
        "Insert city at node j before node i"
        idx_i, idx_j = random.sample(range(self.node_size), 2)
        node_j = state[idx_j]
        state.remove(node_j)
        state.insert(idx_i, node_j)
        return state

    def swap(self, state):
        "Swap cities at positions i and j with each other"
        idx_i, idx_j = random.sample(range(self.node_size), 2)
        state[idx_i], state[idx_j] = state[idx_j], state[idx_i]
        return state

    def insert_route(self, state):
        "Select a subroute from a to b and insert it at another postion in the route"
        idx_i = random.choice(range(self.node_size))
        idx_j = random.choice(range(self.node_size))
        subroute = state[min(idx_i, idx_j):max(idx_i, idx_j)]
        del state[min(idx_i, idx_j):max(idx_i, idx_j)]
        insert_pos = random.choice(range(self.node_size))
        for i in subroute[::-1]:
            state.insert(insert_pos, i)
        return state

    def insert_reversed(self, state):
        "Select a subroute from a to b and insert it reversedly at another position in the route"
        idx_i = random.choice(range(self.node_size))
        idx_j = random.choice(range(self.node_size))
        subroute = state[min(idx_i, idx_j):max(idx_i, idx_j)]
        del state[min(idx_i, idx_j):max(idx_i, idx_j)]
        insert_pos = random.choice(range(self.node_size))
        for i in subroute:
            state.insert(insert_pos, i)
        return state

    def print_best(self, state):
        """ Print function """
        print(f'Best iteration: {self.best_iter} / Best cost: {self.best_cost} / Time: {self.wall_time}')
        print(np.array(state).reshape(self.pp_group, self.dp_group).tolist(), "\n")

    """ Objective Function """
    def objective(self, state):
        """ Objective function for metaheuristic soltuions """
        state = np.array(state).reshape(self.pp_group, self.dp_group)
        
        ## 1) calcurate pp_cost            
        pp_cost_list = []
        for pipeline in state:
            pp_cost = 0
            for i in range(len(pipeline)-1):
                n1, n2 = pipeline[i], pipeline[i+1]
                latency = self.latency[n1][n2]
                #pp_cost += 2 * (self.pp_msg_size / latency) * self.millisec_scaling
                pp_cost += 2 * (self.pp_msg_size * latency) * self.millisec_scaling
            pp_cost_list.append(pp_cost)

        cost_inter_pp = max(pp_cost_list)

        ## 2) calcurate dp_const 
        dp_nodes = [x[0] for x in state]
        inter_latency_list = []
        inner_latency_list = []
        for i in range(len(dp_nodes)):
            if i == len(dp_nodes) - 1:
                n1, n2 = dp_nodes[-1], dp_nodes[0]
            else:
                n1, n2 = dp_nodes[i], dp_nodes[i+1]
            inter_latency_list.append(self.latency[n1][n2])
            inner_latency_list.append(self.latency[dp_nodes[i]][dp_nodes[i]])
            
        self.inter_max_latency = max(inter_latency_list)
        self.inner_max_latency = max(inner_latency_list)

        # dp_const = 2 * (inter_dp_size - 1) / (inter_dp_size * min_bw)
        cost_inter_dp = 2 * (self.inter_dp_size - 1) * self.dp_msg_size * self.inter_max_latency / self.inter_dp_size #seconds
        cost_inter_dp = cost_inter_dp * self.millisec_scaling # miiliseconds
        
        cost_inner_dp = 2 * (self.inner_dp_size - 1) * self.dp_msg_size * self.inner_max_latency / self.inner_dp_size
        cost_inner_dp = cost_inner_dp * self.millisec_scaling # miiliseconds
        ## 3) calcurate total parallel cost 
        total_parallel_cost = cost_inter_pp + cost_inter_dp + cost_inner_dp # miiliseconds
        
        return pp_cost_list, cost_inter_pp, cost_inter_dp, cost_inner_dp, total_parallel_cost       
       
    
    def update_lists(self, state, cost, wall_time):
        low_min = self.low_list[0][1] 
        top_max = self.top_list[0][1]

        low_cost_set = set([x[1] for x in self.low_list])
        top_cost_set = set([x[1] for x in self.top_list])

        if cost > low_min and len({cost} & low_cost_set) == 0:
            self.low_list.pop(0)
            self.low_list.append((state, cost, wall_time))
            
        if cost < top_max and len({cost} & top_cost_set) == 0:
            self.top_list.pop(0)
            self.top_list.append((state, cost, wall_time))
        
        self.low_list = sorted(self.low_list, key=lambda x:x[1], reverse=False)
        self.top_list = sorted(self.top_list, key=lambda x:x[1], reverse=True)

    def make_nodes(self, state):
        return np.array([self.node_index[x] for x in state]).reshape(self.pp_group, self.dp_group).tolist()

    def get_neighbors(self, state):
        """ Returns neighbor of your solution."""
        state = copy.deepcopy(state)
            
        func = self.local_search[random.choice(np.arange(len(self.local_search)))]
        state = func(state)

        return state 

    """ Simulated Annealing """
    def annealing(self, initial_state):
        """Peforms simulated annealing to find a solution"""
        
        # Initialize the parameters
        self.start, self.wall_time, self.best_time = time.time(), 0, 0
        current_temp = copy.deepcopy(self.initial_temp)
        iter, self.best_iter = 0, 0

        self.best = initial_state
        self.pp_cost_list, self.cost_inter_pp, self.cost_inter_dp, self.cost_inner_dp, self.best_cost = self.objective(self.best)
        self.low_list, self.top_list = [(np.arange(self.node_size), -1, -1) for _ in range(self.n_top_low)], [(np.arange(self.node_size), self.best_cost * (10**2), -1) for _ in range(self.n_top_low)]  
        self.local_search = [self.two_opt, self.insert, self.swap, self.insert_route, self.insert_reversed]
        
        # Start by initializing the current state with the initial state
        solution, solution_cost = initial_state, self.best_cost

        # for iter in range(self.n_iter):
        while self.wall_time < self.time_limit:
            neighbor = self.get_neighbors(solution)

            # Check if neighbor is best so far
            _, _, _, _, neighbor_cost = self.objective(neighbor)
            cost_diff = neighbor_cost - solution_cost

            # if the new solution is better, accept it
            if cost_diff <= 0:
                solution = neighbor

            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                scaled_cost_diff = cost_diff  / (10 ** (int(math.log10(cost_diff))+1))
                probability = math.exp(float(-scaled_cost_diff) / float(current_temp))
                
                if random.uniform(0, 1) <= probability:
                    solution = neighbor
                    
            # decrement the temperature
            current_temp = current_temp * self.alpha
            pp_cost_list, cost_inter_pp, cost_inter_dp, cost_inner_dp, solution_cost = self.objective(solution)
                        
            ### Update top list and low list 
            self.wall_time = time.time() - self.start 
            self.update_lists(solution, solution_cost, self.wall_time)

            if solution_cost < self.best_cost:
                self.best = copy.deepcopy(solution)
                self.best_arr = np.array(self.best).reshape(self.pp_group, self.dp_group).tolist()
                self.best_time = self.wall_time

                self.cost_inter_pp, self.cost_inter_dp, self.cost_inner_dp, self.best_cost = cost_inter_pp, cost_inter_dp, cost_inner_dp, solution_cost
                self.best_iter = iter

                self.print_best(self.best)

            iter += 1
        
        self.best_nodes = self.make_nodes(self.best)
        self.low_nodes = [self.make_nodes(x[0]) for x in self.low_list]
        self.top_nodes = [self.make_nodes(x[0]) for x in self.top_list]
        
        return self.best, self.best_cost

    def sa_run(self):
        """ Simulated Annealing main function """
        initial = random.sample(range(self.node_size), self.node_size)
        
        self.annealing(initial)

        # # write output file
        # result_dict = {}
        # result_dict['best_cost'] = self.best_cost
        # result_dict['best_iter'] = self.best_iter
        # result_dict['initial_temp'] = self.initial_temp
        # result_dict['alpha'] = self.alpha
        # result_dict['wall_time'] = self.best_time
        
        # with open(self.output_file, 'w') as f:
        #     json.dump([self.best_nodes, result_dict], f, indent=2)
            
        # make top & low list
        self.low_list = sorted(self.low_list, key=lambda x: x[1], reverse=False)
        self.top_list = sorted(self.top_list, key=lambda x: x[1], reverse=False)
        
        top_low = []
        for tl_list in [self.top_list, self.low_list]:
            for x in tl_list:
                pp_cost_list, cost_inter_pp, cost_inter_dp, cost_inner_dp, total_parallel_cost = self.objective(x[0])
                
                result_dict = {}
                result_dict["pipeline_cost_list"] = pp_cost_list
                result_dict["pipeline_cost"] = cost_inter_pp
                result_dict["inter_data_parallel_cost"] = cost_inter_dp
                result_dict["inner_data_parallel_cost"] = cost_inner_dp
                result_dict["total_parallel_cost"] = total_parallel_cost
                result_dict["wall_time"] = x[-1]
                
                top_low.append([np.array(x[0]).reshape(self.pp_group, self.dp_group).tolist(), result_dict])


        with open(self.output_file, 'w') as f:
            json.dump(top_low, f, indent=2)
        
        return self.cost_inter_pp, self.cost_inter_dp, self.cost_inner_dp, top_low

    

