using PyCall
using Save
using DataStructures

@pyimport scipy.spatial as sp

function flatten{N,T}(A::Vector{NTuple{N,T}})
	A_flat=fill(zero(T),(N,length(A)))
	for i in 1:length(A)
		for j in 1:N
			A_flat[j,i]=A[i][j]
		end
	end
	return A_flat
end
function unordered(x,y)
	return (min(x,y),max(x,y))
end

function compute_fullgraph{T}(raw::Array{T,3}; resolution=Int[4,4,40], radius=130)
	voxel_radius = round(Int, Int[radius,radius,radius] ./ resolution, RoundUp)

	patch_size = max(Int[100,100,10], 2*voxel_radius)
	step_size = patch_size - voxel_radius
	println(voxel_radius)
	println(patch_size)
	println(step_size)
	edges=Set{Tuple{T,T}}()

	rk = 0:step_size[3]:size(raw,3)
	rj = 0:step_size[2]:size(raw,2)
	ri = 0:step_size[1]:size(raw,1)
	N=prod(map(length, [ri,rj,rk]))
	n=0
	for k in rk
		for j in rj
			for i in ri
				n+=1
				println("$(n)/$(N)")
				union!(edges,compute_fullgraph_direct(
										raw[i+1:min(i+patch_size[1],size(raw,1)),
									  	j+1:min(j+patch_size[2],size(raw,2)),
										k+1:min(k+patch_size[3],size(raw,3))],
										resolution=resolution, 
										radius=radius))
			end
		end
	end
	return flatten(collect(edges))
end

#Computes all pairs of supervoxels whose minimum distance is less than a fixed distance
function compute_fullgraph_direct{T}(raw::Array{T,3}; resolution=Int[4,4,40], radius=130)
	point_lists=DefaultDict{T,Vector{Tuple{Int32,Int32,Int32}}}(()->Tuple{Int32,Int32,Int32}[])

	#accumulating points
	for k in 1:size(raw,3), j in 1:size(raw,2), i in 1:size(raw,1)
		if raw[i,j,k] != 0 &&
			(
				(i > 1 && raw[i,j,k] != raw[i-1,j,k]) ||
				(j > 1 && raw[i,j,k] != raw[i,j-1,k]) ||
				(k > 1 && raw[i,j,k] != raw[i,j,k-1]) ||

				(i < size(raw,1) && raw[i,j,k] != raw[i+1,j,k]) ||
				(j < size(raw,2) && raw[i,j,k] != raw[i,j+1,k]) ||
				(k < size(raw,3) && raw[i,j,k] != raw[i,j,k+1])
			)

			push!(point_lists[raw[i,j,k]],(i*resolution[1],j*resolution[2],k*resolution[3]))
		end
	end
	
	#generate trees
	trees = Dict(i => sp.cKDTree(transpose(flatten(points))) for (i,points) in point_lists)

	#compute distances
	edges = Tuple{T,T}[]
	vertices = collect(keys(trees))
	for i in 1:length(vertices)
		for j in i+1:length(vertices)
			t1=trees[vertices[i]]
			t2=trees[vertices[j]]
			if t1[:count_neighbors](t2,r=radius) > 0
				push!(edges,unordered(vertices[i],vertices[j]))
			end
		end
	end
	return edges
end
basename = expanduser("~/mydatasets/3_3_1/")
raw_labels = load(joinpath(basename, "raw.h5"))#[1:1024,1:1024,1:128]
Save.save(joinpath(basename, "full_raw_edges.h5"), compute_fullgraph(raw_labels))
