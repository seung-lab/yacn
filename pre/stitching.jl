using DataFrames
using Save
using Base.Mmap

include("utils.jl")

xindices = [14977:17024,16513:18560,18049:20096]
yindices = [27265:29312,28801:30848,30337:32384]
zindices = [4003:4258]

xendpoints = cat(1,[xindices[1].start],[round(Int,(xindices[i].stop+xindices[i+1].start)/2) for i in 1:length(xindices)-1],[xindices[end].stop+1])
yendpoints = cat(1,[yindices[1].start],[round(Int,(yindices[i].stop+yindices[i+1].start)/2) for i in 1:length(yindices)-1],[yindices[end].stop+1])

xindices_restricted = [xendpoints[i] : (xendpoints[i+1]-1) for i in 1:length(xendpoints)-1]
yindices_restricted = [yendpoints[i] : (yendpoints[i+1]-1) for i in 1:length(yendpoints)-1]
zindices_restricted = zindices

println(xindices_restricted)
println(yindices_restricted)
println(zindices_restricted)

prefix = expanduser("~/seungmount/Omni/TracerTasks/pinky/proofreading/")
omni=expanduser("~/titan_data/data02/ranl/omni.play/bin/release/omni.desktop")

type Region
	size::Int64
	bbox_upper::Array{Int,1}
	bbox_lower::Array{Int,1}
end

immutable ID
	volume_id::Int64
	segment_id::Int64
end

@inline function push_edge!(d::Dict,key,i,j,k)
	if !haskey(d,key)
		d[key]=Region(1,Int[i,j,k],Int[i,j,k])
	else
		e=d[key]
		e.size+=1
		e.bbox_upper[:] = min(Int[i,j,k],e.bbox_upper)
		e.bbox_lower[:] = max(Int[i,j,k],e.bbox_lower)
	end
end

@inline function interval_intersection(A,B)
	return max(A.start,B.start):min(A.stop,B.stop)
end

function intersection(A,B,id1,id2)
	d=Dict{Tuple{ID,ID},Region}()
	dA=Dict{ID,Region}()
	dB=Dict{ID,Region}()
	rxA,ryA,rzA=indices(A)
	rxB,ryB,rzB=indices(B)
	for i in interval_intersection(rxA,rxB),
		j in interval_intersection(ryA,ryB),
		k in interval_intersection(rzA,rzB)
		lA=ID(id1,A[i,j,k])
		lB=ID(id2,B[i,j,k])

		push_edge!(d,(lA,lB),i,j,k)
		push_edge!(dA,lA,i,j,k)
		push_edge!(dB,lB,i,j,k)
	end
	return d,dA,dB
end

function merge_to!(A,B)

end


function get_valid_set(seg,raw,valid,rx,ry,rz)
	s=Set{Int}()
	for k in rz, j in ry, i in rx
		if valid[raw[i,j,k]]==2
			push!(s,seg[i,j,k])
		end
	end
	return s
end


using OffsetArrays
using DataStructures

segs=[]
raws=[]
valid_sets=[]
for rx in xindices[1:2], ry in yindices[1:1], rz in zindices
	base=prefix*"chunk_$(rx.start)-$(rx.stop)_$(ry.start)-$(ry.stop)_$(rz.start)-$(rz.stop).omni.files"
	seg_file="$(base)/proofread.h5"
	raw_file="$(base)/raw.h5"
	valid_file = "$(base)/valid.txt"
	
	seg=OffsetArray(load(seg_file),rx,ry,rz)
	raw=OffsetArray(load(raw_file),rx,ry,rz)
	valid=parse_valid_file(valid_file)
	valid_set = get_valid_set(seg,raw,valid,rx,ry,rz)

	push!(segs,seg)
	push!(raws,raw)
	push!(valid_sets,valid_set)
end

ds=DisjointSets{ID}([])

for i in 1:length(valid_sets)
	for j in valid_sets[i]
		push!(ds, ID(i,j))
	end
end


function decision_value(r_intersect, r_a, r_b)
	return r_intersect.size > 10000 && (r_intersect.size/r_a.size > 0.33 || r_intersect.size/r_b.size > 0.33)
end

for i in 1:length(segs)
	for j in i+1:length(segs)
		d,di,dj = intersection(segs[i],segs[j],i,j)
		for k in d
			if k[1][1].segment_id in valid_sets[i] && k[1][2].segment_id in valid_sets[j]
				if decision_value(k[2], di[k[1][1]], dj[k[1][2]])
					union!(ds, k[1][1], k[1][2])
					println("merged")
				else
					println("not merged")
				end
			end
		end
	end
end
println(h)
