using PyCall
using Save

@pyimport scipy.spatial as sp

function flatten{T<:Vector}(A::Vector{T};k=3)
	A_flat=fill(0,(k,length(A)))
	for i in 1:length(A)
		A_flat[:,i]=A[i]
	end
	return A_flat
end
function unordered(x,y)
	return (min(x,y),max(x,y))
end

function compile_candidates{T}(raw::Array{T,3}, step_size, patch_size)
	s=Set{Tuple{T,T}}()

	rk = 0:step_size[3]:size(raw,3)
	rj = 0:step_size[2]:size(raw,2)
	ri = 0:step_size[1]:size(raw,1)
	for k in rk
		for j in rj
			for i in ri
				l=unique(raw[i+1:min(i+patch_size[1],size(raw,1)), j+1:min(j+patch_size[2],size(raw,2)), k+1:min(k+patch_size[3], size(raw,3))])
				for I in 1:length(l)
					for J in I+1:length(l)
						if l[I]!=0 && l[J] != 0
							push!(s,unordered(l[I],l[J]))
						end
					end
				end
			end
		end
	end
	return s
end

#Computes all pairs of supervoxels whose minimum distance is less than a fixed distance
function compute_fullgraph{T}(raw::Array{T,3}; resolution=Int[4,4,40], radius=130)
	point_lists = Vector{Vector{Int32}}[Vector{Int32}[] for i in 1:maximum(raw)]

	println("accumulating points...")
	@time for k in 1:size(raw,3), j in 1:size(raw,2), i in 1:size(raw,1)
		if raw[i,j,k] != 0 &&
			(
				(i > 1 && raw[i,j,k] != raw[i-1,j,k]) ||
				(j > 1 && raw[i,j,k] != raw[i,j-1,k]) ||
				(k > 1 && raw[i,j,k] != raw[i,j,k-1]) ||

				(i < size(raw,1) && raw[i,j,k] != raw[i+1,j,k]) ||
				(j < size(raw,2) && raw[i,j,k] != raw[i,j+1,k]) ||
				(k < size(raw,3) && raw[i,j,k] != raw[i,j,k+1])
			)

			push!(point_lists[raw[i,j,k]],Int[i*resolution[1],j*resolution[2],k*resolution[3]])
		end
	end

	println("sparsity: $(sum(map(length, point_lists))) / $(size(raw,1)*size(raw,2)*size(raw,3))")

	println("compiling candidates...")
	patch_size = round(Int, 1.6 .* [radius, radius, radius] ./ resolution, RoundUp)
	step_size = round(Int, 0.5 .* [radius, radius, radius] ./ resolution, RoundDown)
	@time s=compile_candidates(raw, step_size, patch_size)

	println("$(length(s)) candidate edges")

	println("generating trees...")
	@time trees = [length(points)>0 ? sp.cKDTree(transpose(flatten(points))) : nothing for points in point_lists]

	edges = Vector{Int32}[]
	println("computing distances...")
	n=0
	@time for (i,j) in s
		n+=1
		if mod(n,100)==0
			println("$(n)/$(length(s))")
		end
		t1=trees[i]
		t2=trees[j]
		if t1 != nothing && t2 != nothing
			tmp=t1[:count_neighbors](t2,r=radius)
			if tmp > 0
				push!(edges,Int32[i,j])
			end
		end
	end
	println("$(length(edges)) edges")
	return flatten(edges, k=2)
end
basename = expanduser("~/mydatasets/3_3_1/")
raw_labels = load(joinpath(basename, "raw.h5"))#[1:1024,1:1024,1:128]
Save.save(joinpath(basename, "full_raw_edges.h5"), compute_fullgraph(raw_labels))
