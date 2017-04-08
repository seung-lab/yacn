function filter_samples(A, limit=200000, full_size = Int[256,2048,2048], padding = Int[33,318+25,318+25])
	A_filtered = Vector{Int}[]
	for i in 1:size(A,2)
		tmp = A[:,i]
		if all([p < x < c-p for (p,c,x) in zip(padding, full_size, tmp)])
			push!(A_filtered, tmp)
		end
	end
	A_filtered = A_filtered[randperm(length(A_filtered))][1:200000]
	return flatten_samples(A_filtered)
end
