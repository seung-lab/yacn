FFTW.set_num_threads(6)

function relabel(labels,relabels)
	return map(i->relabels[i], labels)
end

function gen_weights(labels; M=30)
	labels+=1
	N=maximum(labels)
	tmp = randn(N,M)
	relabels = tmp ./ sqrt(sum(tmp.^2,2))
	relabels[1,:]=0

	kernel = fill(0f0, size(labels))
	kernel_size=[100,100,20]
	kernel[[1:x for x in kernel_size]...]=1
	kernel = circshift(kernel,[round(Int,-x/2) for x in kernel_size]) / sum(kernel)

	plan = plan_fft(kernel, [1,2,3])
	iplan = inv(plan)
	fft_kernel=plan*(kernel)

	weights=fill(0f0, size(labels))
	for i in 1:M
		println(i)
		indicator = relabel(labels, relabels[:,i])
		smoothed = iplan * ((plan*(indicator)).*fft_kernel)
		weights[:,:,:] += real(smoothed.*indicator)
	end
	for i in eachindex(weights,labels)
		if labels[i]==1
			weights[i]=0
		else
			weights[i] = 1 ./ max(0.05, weights[i])
		end
	end
	return weights
end

function gen_samples(labels; patch_size=Int[314,314,48], N=100000)
	weights = gen_weights(labels)
	weights *= (N/sum(weights))

	A=Vector{Int}[]

	for k in 1:size(weights,3), j in 1:size(weights,2), i in 1:size(weights,1)
		if rand() <= weights[i,j,k]
			push!(A,Int[i,j,k])
		end
	end

	println(length(A))
	A=collect(filter(x-> reduce(*, true, [patch_size[i]/2 < x[i] < size(weights,i)-patch_size[i]/2 for i in 1:3]), A))
	println("$(length(A)) examples collected")

	A_flat=fill(0,(3,length(A)))
	for i in 1:length(A)
		A_flat[:,i]=A[i]
	end
	return A_flat
end

#DIR=expanduser("~/datasets/AC3/test")
#weights = main(load("$(DIR)/oracle_labels.h5"))
#save("$(DIR)/oracle_approximate_weights.h5", weights)
