using Base.Collections
import Base: start, done, next

immutable EdgeIterator
	steps::Array{Int,1}
	len::Int
end
immutable EdgeIteratorState
	i::Int
	k::Int
end

edges(A)=EdgeIterator(Int[1,size(A,1),size(A,1)*size(A,2)],length(A))
start(e::EdgeIterator)=EdgeIteratorState(1,1)
done(iter::EdgeIterator,state::EdgeIteratorState)=(state.i > iter.len)
function next(iter::EdgeIterator, state::EdgeIteratorState)
	obj = (state.i, mod1(state.i+iter.steps[state.k], iter.len))
	newstate = EdgeIteratorState(state.k==3 ? state.i+1: state.i, mod1(state.k+1,3))
	return (obj, newstate)
end

function neighbours(A,i)
	return Int[mod1(i+1,length(A)), mod1(i+size(A,1),length(A)), mod1(i + size(A,1) * size(A,2), length(A)),
			mod1(i-1,length(A)), mod1(i-size(A,1),length(A)), mod1(i - size(A,1) * size(A,2), length(A))]
end

const directions = Array{Int,1}[
								Int[6,0,0],
								Int[0,6,0],
								Int[0,0,30],
								Int[-6,0,0],
								Int[0,-6,0],
								Int[0,0,-30]
								]

function distance_transform(A)
	distances = fill(Inf,size(A))
	history = [Int[0,0,0] for i in A]
	dump(history)
	pq = Collections.PriorityQueue(Int, Float64)
	for i in eachindex(A, distances)
		if A[i] == 0
			distances[i]=0
			pq[i]=0
		end
	end
	for (i,j) in edges(A)
		if A[i] != A[j]
			distances[i]=A[i]
			distances[j]=A[j]
			pq[i]=0
			pq[j]=0
		end
	end

	resolution=Int[6,6,30,6,6,30]
	while !isempty(pq)
		i = dequeue!(pq)
		d = distances[i]
		for (j,dir) in zip(neighbours(A,i),Int[1,2,3,4,5,6])
			if distances[j] > d + resolution[dir]
				distances[j] = d + resolution[dir]
				history[j] = history[i] + directions[dir]
				pq[j] = min(haskey(pq,j) ? pq[j] : Inf, d + resolution[dir])
			end
		end
	end
	
	normalized_dirs = zeros(size(A)..., 3)
	for k in 1:size(A,3), j in 1:size(A,2), i in 1:size(A,1)
		normalized_dirs[i,j,k,:]=history[i,j,k]/(norm(history[i,j,k])+0.001)
	end

	return normalized_dirs
end

function affinitize(labels, ws)
	masks = [
		  0x00000001, 
		  0x00000010, 
		  0x00000100,
		  0x00001000,
		  0x00010000,
		  0x00100000]
	otpt = zeros(Int32, size(ws))
	for i in eachindex(ws)
		for (j,dir) in zip(neighbours(ws,i),Int[1,2,3,4,5,6])
			if ws[j] >= ws[i] && labels[i] == labels[j] && labels[i] != 0
				otpt[i]|=masks[dir]
			end
		end
	end
	return otpt
end


using Save
x=load("~/datasets/AC3/test/human_labels.h5")[1:500,1:500,1:50]
tmp=distance_transform(x)
save("tmp.h5",tmp)
