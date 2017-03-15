function unordered(x,y)
	return (min(x,y),max(x,y))
end

function compute_regiongraph{T,S}(raw::Array{T,3}, machine_labels::Array{S,3})
	X,Y,Z=size(raw)
	vertices = unique(raw)
	edges = Set{Tuple{T,T}}()
	for k in 1:Z-1, j in 1:Y, i in 1:X
		if raw[i,j,k] != raw[i,j,k+1] && machine_labels[i,j,k] == machine_labels[i,j,k+1]
			push!(edges,unordered(raw[i,j,k],raw[i,j,k+1]))
		end
	end
	for k in 1:Z, j in 1:Y-1, i in 1:X
		if raw[i,j,k] != raw[i,j+1,k] && machine_labels[i,j,k] == machine_labels[i,j+1,k]
			push!(edges,unordered(raw[i,j,k],raw[i,j+1,k]))
		end
	end
	for k in 1:Z, j in 1:Y, i in 1:X-1
		if raw[i,j,k] != raw[i+1,j,k] && machine_labels[i,j,k] == machine_labels[i+1,j,k]
			push!(edges,unordered(raw[i+1,j,k],raw[i,j,k]))
		end
	end
	flat_edges = zeros(T,(2,length(edges)))
	for (i,e) in enumerate(edges)
		flat_edges[1,i] = e[1]
		flat_edges[2,i] = e[2]
	end
	return vertices, flat_edges
end


#What is our affinities convention?
function compute_regiongraph{T,S,U}(raw::Array{T,3}, machine_labels::Array{S,3}, affinities::Array{U,4}; threshold=0.3)
	X,Y,Z=size(raw)
	vertices = unique(raw)
	edges = Set{Tuple{T,T}}()
	for k in 1:Z-1, j in 1:Y, i in 1:X
		if raw[i,j,k] != raw[i,j,k+1] && machine_labels[i,j,k] == machine_labels[i,j,k+1] && affinities[i,j,k+1,3] > threshold
			push!(edges,unordered(raw[i,j,k],raw[i,j,k+1]))
		end
	end
	for k in 1:Z, j in 1:Y-1, i in 1:X
		if raw[i,j,k] != raw[i,j+1,k] && machine_labels[i,j,k] == machine_labels[i,j+1,k] && affinities[i,j+1,k,3] > threshold
			push!(edges,unordered(raw[i,j,k],raw[i,j+1,k]))
		end
	end
	for k in 1:Z, j in 1:Y, i in 1:X-1
		if raw[i,j,k] != raw[i+1,j,k] && machine_labels[i,j,k] == machine_labels[i+1,j,k] && affinities[i+1,j,k,3] > threshold
			push!(edges,unordered(raw[i+1,j,k],raw[i,j,k]))
		end
	end
	flat_edges = zeros(T,(2,length(edges)))
	for (i,e) in enumerate(edges)
		flat_edges[1,i] = e[1]
		flat_edges[2,i] = e[2]
	end
	return vertices, flat_edges
end
