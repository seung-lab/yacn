function downsample_volume(x)
	return x[1:2:end, 1:2:end, :]
end

#the input should be zyx, zero based indices
function downsample_samples(A)
	dump(A)
	A_filtered = Vector{Int}[]
	for i in 1:size(A,2)
		tmp = A[:,i]
		if tmp[2]%2==0 && tmp[3]%2==0
			push!(A_filtered, Int[tmp[1],tmp[2]/2,tmp[3]/2])
		end
	end
	return flatten_samples(A_filtered)
end
