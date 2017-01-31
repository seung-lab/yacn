using Save
DIR=expanduser("~/datasets/AC3/train")

patch_size=[314,314,48]
weights = load("$(DIR)/mean_weights.h5")
weights *= (100000/sum(weights))

A=Vector{Int}[]

for k in 1:size(weights,3), j in 1:size(weights,2), i in 1:size(weights,1)
	if rand() <= weights[i,j,k]
		push!(A,Int[i,j,k])
	end
end

println(length(A))
A=collect(filter(x-> reduce(*, true, [patch_size[i]/2 < x[i] < size(weights,i)-patch_size[i]/2 for i in 1:3]), A))
println(length(A))

A_flat=fill(0,(3,length(A)))
for i in 1:length(A)
	A_flat[:,i]=A[i]
end

save("$(DIR)/mean_samples.h5", A_flat)
