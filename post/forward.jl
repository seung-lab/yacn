using Save
using Iterators
function bump(x,y,z)
	return exp(bump_logit(x,y,z))
end
function bump(x,y,z, max_logit)
	return exp(bump_logit(x,y,z)-max_logit)
end
function bump_map(win_size)
	[bump(i/(1+win_size[1]),j/(1+win_size[2]), k/(1+win_size[3]))
	for i in 1:win_size[1], j in 1:win_size[2], k in 1:win_size[3]]
end
function bump_map(win_size, max_logit)
	[bump(i/(1+win_size[1]),j/(1+win_size[2]), k/(1+win_size[3]),max_logit[i,j,k])
	for i in 1:win_size[1], j in 1:win_size[2], k in 1:win_size[3]]
end

function bump_logit_map(win_size)
	[bump_logit(i/(1+win_size[1]),j/(1+win_size[2]), k/(1+win_size[3]))
	for i in 1:win_size[1], j in 1:win_size[2], k in 1:win_size[3]]
end

@inline function bump_logit(x,y,z)
	t=1.5
	-(x*(1-x))^(-t)-(y*(1-y))^(-t)-(z*(1-z))^(-t)
end


#Inputs:
#inpt: a (X,Y,Z,...) array. For example, it could be of size (X,Y,Z) for an input image
#otpt: a preallocated (X,Y,Z,...) array. For example, it could be of size (X,Y,Z,3) for 
#an affinity map.
#patch_size: size of the patch on which to run forward pass
#function f which takes as input an array of size (x,y,z,...) where (x,y,z) = patchsize
#and outputs an array of size (x,y,z,...)
#step is the overlap percentage for the patches
function forward!{M,N,T}(inpt::Array{T,M}, otpt::Array{Float32,N}, patch_size, f; step=0.5)
	sz = size(inpt)[1:3]
	normalization = zeros(Float32,sz)
	max_logit = fill(-Inf,sz)
	max_logit_window = bump_logit_map(tuple(patch_size...))

	#This is an iterator which returns ranges for all the patches on which we will run inference
	range_iterator = product([
		map(corner->corner:corner+patch_size[i]-1,
		begin
			last=size(inpt,i)-patch_size[i]+1
			corner = map(x->round(Int,x), 1f0:(last-1)/ceil(last/(step*patch_size[i])):last)
		end
		)
		for i in 1:3]...)
	
	for ranges in range_iterator
		max_logit[ranges...]=max(max_logit_window,sub(max_logit,ranges...))
	end

	inpt_colons = [Colon() for i in 1:(M-3)]
	otpt_colons = [Colon() for i in 1:(N-3)]
	for (t,ranges) in enumerate(range_iterator)
		println(t)
		println(ranges)

		#mask is the bump function for this patch.
		mask = bump_map(patch_size, max_logit[ranges...])
		otpt[ranges...,otpt_colons...] += mask .* f(inpt[ranges...,inpt_colons...])
		normalization[ranges...]+=mask
	end
	
	#We need to divide by the normalization factor at each pixel.
	otpt[:] = otpt ./ normalization
	return otpt
end
function forward_max!{M,N}(inpt::Array{Float32,M}, otpt::Array{Float32,N}, patch_size, f; step=0.5)
	sz = size(inpt)[1:3]
	#This is an iterator which returns ranges for all the patches on which we will run inference
	range_iterator = product([
		map(corner->corner:corner+patch_size[i]-1,
		begin
			last=size(inpt,i)-patch_size[i]+1
			corner = map(x->round(Int,x), 1f0:(last-1)/ceil(last/(step*patch_size[i])):last)
		end
		)
		for i in 1:3]...)
	

	inpt_colons = [Colon() for i in 1:(M-3)]
	otpt_colons = [Colon() for i in 1:(N-3)]
	for (t,ranges) in enumerate(range_iterator)
		println(t)
		println(ranges)
		otpt[ranges...,otpt_colons...] = max(otpt[ranges...,otpt_colons...],f(inpt[ranges...,inpt_colons...]))
	end
	
	return otpt
end
