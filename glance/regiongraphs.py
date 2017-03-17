import networkx as nx

def make_graph(vertices, edges): 
	G=nx.Graph()
	G.add_nodes_from(vertices)
	G.add_edges_from(edges)
	"""
	for i in xrange(len(vertices)):
		G.add_node(vertices[i])
	for i in xrange(edges.shape[0]):
		G.add_edge(edges[i,0],edges[i,1])
	"""
	return G


def add_clique(G, vertices):
	for i in xrange(len(vertices)):
		for j in xrange(i,len(vertices)):
			G.add_edge(vertices[i],vertices[j])
	
def delete_bipartite(G, vertices1, vertices2):
	for v1 in vertices1:
		for v2 in vertices2:
			if G.has_edge(v1,v2):
				#print "deleted edge", v1, v2
				G.remove_edge(v1,v2)

def expand_list(G,l):
	return set(_plain_bfs(G,l))

def _plain_bfs(G, sources):
    seen = set()
    nextlevel = set(sources)
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(G[v])
"""
def expand_list(G,l):
	s=set()
	return list(s.union(*[nx.node_connected_component(G, x) for x in l]))
"""
