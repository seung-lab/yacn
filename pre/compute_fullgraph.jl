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

#Computes all pairs of supervoxels whose minimum distance is less than a fixed distance
function compute_fullgraph{T}(raw::Array{T,3}; resolution=Int[8,8,40])
	point_lists = Vector{Vector{Int}}[Vector{Int}[] for i in 1:maximum(raw)]

	println("accumulating points...")
	for k in 1:size(raw,3), j in 1:size(raw,2), i in 1:size(raw,1)
		if raw[i,j,k] != 0
			push!(point_lists[raw[i,j,k]],Int[i*resolution[1],j*resolution[2],k*resolution[3]])
		end
	end

	println("compiling candidates...")
	s=Set{Tuple{T,T}}()
	patch_size = round(Int,[240,240,240] ./ resolution)
	for i in 0:round(Int,patch_size[1]/3):size(raw,1)+patch_size[1]
		for j in 0:round(Int,patch_size[2]/3):size(raw,2)+patch_size[2]
			for k in 0:round(Int,patch_size[3]/3):size(raw,3)+patch_size[3]
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

	println("$(length(s)) candidate edges")

	println("generating trees...")
	trees = [length(points)>0 ? sp.cKDTree(transpose(flatten(points))) : nothing for points in point_lists]

	edges = Vector{Int}[]
	println("computing distances...")
	n=0
	for (i,j) in s
		n+=1
		if mod(n,100)==0
			println("$(n)/$(length(s))")
		end
		t1=trees[i]
		t2=trees[j]
		if t1 != nothing && t2 != nothing
			tmp=t1[:count_neighbors](t2,150)
			if tmp > 0
				push!(edges,Int[i,j])
			end
		end
	end
	println("$(length(edges)) edges")
	return flatten(edges, k=2)
end
basename = expanduser("~/mydatasets/3_3_1/ds/")
raw_labels = load(joinpath(basename, "raw.h5"))#[1:500,1:500,1:30]
Save.save(joinpath(basename, "full_raw_edges.h5"), compute_fullgraph(raw_labels))
