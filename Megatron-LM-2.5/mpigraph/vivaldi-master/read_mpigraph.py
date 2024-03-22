#!/usr/bin/python

from Graph import Graph
from Configuration import Configuration
from Vivaldi import Vivaldi

import sys
from matplotlib.pyplot import *
import json

def buildgraph(rows):
	latency_scale_factor = 1000
	MB_to_GB = 1000
	g = Graph(len(rows))
	for node in range(len(rows)):
		arr = rows[node]
		rtts = [latency_scale_factor*MB_to_GB/float(x) for x in arr if x > 0]
		for neighbor in range(len(rtts)):
			g.addVertex(node,neighbor,rtts[neighbor])
	return g
	
def table(data, title=""):
	length = len(data)
	data = sorted(data)

	
if __name__== "__main__":
	if len(sys.argv) != 2:
		print( "Usage: %s <rtt_file>"%sys.argv[0])
		sys.exit(0)
	
	rttfile = sys.argv[1]
	infile = open(rttfile, 'r')
	mpigraph_data = json.load(infile)
	host_name = mpigraph_data[str(0)]['gpus']
	p2p_recv_bw = mpigraph_data[str(0)]['bandwidth']
	num_nodes = len(p2p_recv_bw)
    #print(num_nodes)
	
	# These parameters are part of the Configuration.
	# Modify them according to your need.
	num_neighbors  = 8
	num_iterations = 1000
	num_dimension = 3
	
	# build a configuration and load the matrix into the graph
	c = Configuration(num_nodes, num_neighbors, num_iterations, d=num_dimension)
	init_graph = buildgraph(p2p_recv_bw)
	
	# # run vivaldi: here, only the CDF of the relative error is retrieved. 
	# # Modify to retrieve what's requested.
	v = Vivaldi(init_graph, c)
	v.run()
	predicted = v.getRTTGraph()
 
	rerr = v.getRelativeError(predicted)
	table([100*x for x in rerr], "RELATIVE ERROR (%)")

	count = 0
	for i in v.positions:
		print("{} {} {}".format(str(host_name[count]),str(i[0]),str(i[1])))
		count = count + 1 
  
	# # # Example (using pylab plotting function):
	# #x = [i[0] for i in v.positions]
	# #y = [i[1] for i in v.positions]
 
	

	# x,y = v.computeCDF(rerr)
	# print(x)
	# print(y)
	# # plot(x,y)
	# # show()
