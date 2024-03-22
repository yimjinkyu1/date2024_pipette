import json
import copy
import random
import argparse
import numpy as np

from ortools.sat.python import cp_model


class CPSolver:
    def __init__(self, read_file, output_file, node_size, pp_group, dp_group, inter_dp_size, inner_dp_size, pp_msg_size, dp_msg_size, time_limit):
        np.random.seed(0)
        random.seed(0)
        
        # Read nccl data
        with open(read_file, "r") as f:
            data = json.load(f)

        self.node_data = data["gpus"]
        self.node_size = len(self.node_data)
        self.node_index = {i:self.node_data[i] for i in range(self.node_size)}

        self.sendrecv_bw_data = np.array(data["bandwidth"])
        
        self.read_file = read_file
        self.output_file = output_file
        self.node_size = node_size
        self.pp_group = pp_group
        self.dp_group = dp_group
        self.inter_dp_size = inter_dp_size
        self.inner_dp_size = inner_dp_size
        self.pp_msg_size = pp_msg_size #GB
        self.dp_msg_size = dp_msg_size #GB
        
        self.time_limit = time_limit

        # ### TEST params ###
        # min_val = 5
        # max_val = 200
        # self.sendrecv_bw_data = np.random.randint(low=min_val, high=max_val, size=(self.node_size, self.node_size)) #GB/s
    
        self.node_index = {i:f'agpu{i}' for i in range(self.node_size)}
        # Variable scaling factors
        self.millisec_scaling = 10**3 # scaling factor (seconds to milliseconds)
        self.latency_scaling = 10**10 # scaling factor for latency

        self.total_latency = 1 / self.sendrecv_bw_data * self.latency_scaling  

        if self.pp_group > 1 and self.dp_group > 1:
            self.two_dim = True
        else:
            self.two_dim = False

        self.infinity = 10**17
        self.flattened_latency = [int(x) for x in np.matrix(self.total_latency).flatten().tolist()[0]]

    def make_2d_model(self) : 
        """ 
        Make CP-SAT solver model using Google ORTools
        https://developers.google.com/optimization/cp/cp_solver?hl=ko 
        """
        
        self.model = cp_model.CpModel()

        # variable for node
        self.nodes = []
        for i in range(self.node_size):
            var = self.model.NewIntVar(0, self.node_size-1, f'node_{i}')
            self.nodes.append(var)

        # Constraints: No duplicated variables
        self.model.AddAllDifferent(self.nodes)

        # 0) Objectives: Pipeline Parallel cost + Data Parallel cost
        self.nodes = np.array(self.nodes).reshape((self.pp_group, self.dp_group))
      
        self.cost_inter_pp = self.model.NewIntVar(0, self.infinity, 'cost_inter_pp') 
        self.pp_cost_list = [self.model.NewIntVar(0, self.infinity, f'sum_pp_cost_{p}') for p in range(self.pp_group)]
        
        for p, pipeline in enumerate(self.nodes):
            self.pp_var_list = [self.model.NewIntVar(0, self.infinity, f'pp_var_{p}_{i}') for i in range(self.dp_group - 1)]
            
            for idx in range(self.dp_group - 1):
                n1, n2 = pipeline[idx], pipeline[idx+1]
                
                bw = self.model.NewIntVar(0, self.infinity, 'bw')
                
                # index for the flattened latency matrix
                n_idx = self.model.NewIntVar(0, self.node_size*self.node_size, 'idx')
                self.model.Add(n_idx == cp_model.LinearExpr.Term(n1, self.node_size) + n2)
                self.model.AddElement(n_idx, self.flattened_latency, bw)
                
                self.model.AddMultiplicationEquality(self.pp_var_list[idx], [bw, self.pp_msg_size])

            self.model.Add(self.pp_cost_list[p] == sum(self.pp_var_list))
                           
        self.model.AddMaxEquality(self.cost_inter_pp, self.pp_cost_list)

        # 2) calculate dp_cost for the first data parallel group                
        self.dp_nodes = [x[0] for x in self.nodes]
        
        self.max_inter_dp = self.model.NewIntVar(0, self.infinity, 'max_inter_dp')
        self.inter_dp_list = [self.model.NewIntVar(0, self.infinity, f'dp_cost_{i}') for i in range(self.inter_dp_size)]
        self.max_inner_dp = self.model.NewIntVar(0, self.infinity, 'max_inner_dp')
        self.inner_dp_list = [self.model.NewIntVar(0, self.infinity, f'dp_cost_{i}') for i in range(self.inner_dp_size)]
        
        for idx in range(self.inter_dp_size):
            
            if idx == self.inter_dp_size - 1:
                n1, n2 = self.dp_nodes[-1], self.dp_nodes[0]
            else:
                n1, n2 = self.dp_nodes[idx], self.dp_nodes[idx+1]
            
            # index for the flattened latency matrix
            inter_idx = self.model.NewIntVar(0, self.node_size*self.node_size, 'inter_idx')
            self.model.Add(inter_idx == cp_model.LinearExpr.Term(n1, self.node_size) + n2)
            self.model.AddElement(inter_idx, self.flattened_latency, self.inter_dp_list[idx])
            
            inner_idx = self.model.NewIntVar(0, self.node_size*self.node_size, 'inner_idx')
            self.model.Add(inner_idx == cp_model.LinearExpr.Term(n1, self.node_size) + n1)
            self.model.AddElement(inner_idx, self.flattened_latency, self.inner_dp_list[idx])

        self.model.AddMaxEquality(self.max_inter_dp, self.inter_dp_list)
        self.model.AddMaxEquality(self.max_inner_dp, self.inner_dp_list)
        
        # 2-1) Inter DP cost
        self.cost_inter_dp = self.model.NewIntVar(0, self.infinity, 'cost_inter_dp')
        inter_num = self.model.NewIntVar(0, self.infinity, 'inter_num')
        self.model.AddMultiplicationEquality(inter_num, [2 * (self.inter_dp_size - 1) * self.dp_msg_size, self.max_inter_dp])
        
        inter_denom = self.model.NewIntVar(1, self.infinity, 'inter_denom')
        self.model.Add(inter_denom == self.inter_dp_size)

        self.model.AddDivisionEquality(self.cost_inter_dp, inter_num, inter_denom)
                    
        # 2-2) Inner DP cost     
        self.cost_inner_dp = self.model.NewIntVar(0, self.infinity, 'cost_inner_dp')
        inner_num = self.model.NewIntVar(0, self.infinity, 'inner_num')
        self.model.AddMultiplicationEquality(inner_num, [2 * (self.inner_dp_size - 1) * self.dp_msg_size, self.max_inner_dp])
        
        inner_denom = self.model.NewIntVar(1, self.infinity, 'inner_denom')
        self.model.Add(inner_denom == self.inter_dp_size)

        self.model.AddDivisionEquality(self.cost_inner_dp, inner_num, inner_denom)
        

        ## 3) calculate total parallel cost 
        self.makespan = self.model.NewIntVar(0, self.infinity, 'makespan')
        self.model.Add(self.makespan == self.cost_inter_pp + self.cost_inter_dp + self.cost_inner_dp)

        self.model.Minimize(self.makespan)


    def make_1d_model(self) : 
        """ 
        Make CP-SAT solver model using Google ORTools
        https://developers.google.com/optimization/cp/cp_solver?hl=ko 
        """
        # Preprocess the latency matrix
        self.total_latency = np.concatenate((np.zeros((self.node_size, 1)), self.total_latency), axis=1)
        self.total_latency = np.concatenate((np.zeros((1, self.node_size+1)), self.total_latency), axis=0)
        # print(self.total_latency)
        # Make the model
        self.model = cp_model.CpModel()

        obj_vars = []
        obj_coeffs = []

        # Create the circuit constraint.
        arcs = []
        self.arc_literals = {}
        for i in range(self.node_size + 1):
            for j in range(self.node_size + 1):
                if i == j:
                    continue

                lit = self.model.NewBoolVar('%i follows %i' % (j, i))
                arcs.append([i, j, lit])
                self.arc_literals[i, j] = lit

                obj_vars.append(lit)
                obj_coeffs.append(self.total_latency[i][j])

        self.model.AddCircuit(arcs)

        # Minimize weighted sum of arcs. Because this s
        self.model.Minimize(sum(obj_vars[i] * obj_coeffs[i] for i in range(len(obj_vars))))

    def solve(self):
        """ Creates a solver & model and solves the model """
        self.solver = cp_model.CpSolver()
        
        # Apply constraints and objectives
        if self.two_dim:
            self.make_2d_model()
        else:
            self.make_1d_model()   
        
        # Sets CpSolver parameters
        if self.time_limit != -1:
            self.solver.parameters.max_time_in_seconds = self.time_limit
        self.solver.parameters.random_seed = 0
        # TODO test
        import multiprocessing
        #self.solver.parameters.num_search_workers = multiprocessing.cpu_count()
        self.solver.parameters.num_search_workers = 8
        
        self.solver.parameters.log_search_progress = False
        if not self.two_dim:
            # To benefit from the linearization of the circuit constraint.
            self.solver.parameters.linearization_level = 2
        
        # Solve the model with solution callback
        solution_callback = SolutionCallback(self)
        self.status = self.solver.Solve(model = self.model, solution_callback = solution_callback)
        #self.status = self.solver.SolveWithSolutionCallback(self.model, solution_callback) # Deprecated
        
        # Check the final result
        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:
            
            if self.two_dim:
                pipeline_result = [[self.solver.Value(x) for x in pipeline] for pipeline in self.nodes] 
            else:
                current_node = 0
                pipeline_result = []
                route_is_finished = False
                while not route_is_finished:
                    for i in range(self.node_size + 1):
                        if i == current_node:
                            continue
                        if self.solver.BooleanValue(self.arc_literals[current_node, i]):
                            pipeline_result.append(i-1)
                            current_node = i
                            if current_node == 0:
                                route_is_finished = True
                            break
                pipeline_result = pipeline_result[:-1]
            
            print(f'Status: {self.solver.StatusName(self.status)}')
            print(f'Wall time: {np.round(self.solver.WallTime(), 2)} s')
            print("Pipeline Result")
            print(pipeline_result)
            
            if self.two_dim:
                print("\nObjective variables: Pipeline Cost")
                print("PP cost list:", [self.solver.Value(x) for x in self.pp_cost_list])
                print("Inter PP cost:", self.solver.Value(self.cost_inter_pp))
                
                print("\nObjective variables: Data Parallel Cost")
                print("Inter DP latency list:", [self.solver.Value(x) for x in self.inter_dp_list])
                print("Inner DP latency list:", [self.solver.Value(x) for x in self.inner_dp_list])
                
                print("Final Inter DP cost:", self.solver.Value(self.cost_inter_dp)) 
                print("Final Inner DP cost:", self.solver.Value(self.cost_inner_dp)) 
                      
            # Save Final Result
            result_dict = {}
            result_dict['optim_result'] = pipeline_result
            result_dict['ortools_status'] = self.solver.StatusName(self.status)
            if self.two_dim:
                result_dict['pipeline_cost'] = [self.solver.Value(x) for x in self.pp_cost_list]
                result_dict['cost_inter_pp'] = self.solver.Value(self.cost_inter_pp)
                result_dict['cost_inter_dp'] = self.solver.Value(self.cost_inter_dp)
                result_dict['cost_inner_dp'] = self.solver.Value(self.cost_inner_dp)

            result_dict['objective_value'] = self.solver.ObjectiveValue()
            result_dict['wall_time'] = self.solver.WallTime()
            
            result = [pipeline_result, result_dict]

            with open(self.output_file, 'w') as f:
                json.dump(result, f, indent=2)

            return result_dict
        else:
            print('No solution found.')
            print(f'{self.solver.SufficientAssumptionsForInfeasibility()}')
            
            for var_index in self.solver.ResponseProto().sufficient_assumptions_for_infeasibility:
                print(var_index, self.model.VarIndexToVarProto(var_index))

class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """ Intermediate solution Callback & Save the logs """
    def __init__(self, save_vars):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.save_vars = save_vars
        self.solutions = []

    def on_solution_callback(self):
        print('Solution %i, time = %f s, objective = %i' %
          (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1
        
        # save intermediate solutions
        if self.save_vars.two_dim:
            pipeline_result = [[self.Value(x) for x in pipeline] for pipeline in self.save_vars.nodes]  
        else:
            current_node = 0
            pipeline_result = []
            route_is_finished = False
            while not route_is_finished:
                for i in range(self.save_vars.node_size + 1):
                    if i == current_node:
                        continue
                    if self.BooleanValue(self.save_vars.arc_literals[current_node, i]):
                        pipeline_result.append(i-1)
                        current_node = i
                        if current_node == 0:
                            route_is_finished = True
                        break  
            pipeline_result = pipeline_result[:-1]
              
        print(pipeline_result)
            
        result_dict = {}
        result_dict['optim_result'] = pipeline_result
        if self.save_vars.two_dim:
            result_dict['pipeline_cost'] = [self.Value(x) for x in self.save_vars.pp_cost_list]
            result_dict['cost_inter_pp'] = self.Value(self.save_vars.cost_inter_pp)
            result_dict['cost_inter_dp'] = self.Value(self.save_vars.cost_inter_dp)
            result_dict['cost_inner_dp'] = self.Value(self.save_vars.cost_inner_dp)

        result_dict['objective_value'] = self.ObjectiveValue()
        result_dict['wall_time'] = self.WallTime()
        
        result = [pipeline_result, result_dict]

        with open(self.save_vars.output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
