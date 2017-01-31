using Save
DIR=expanduser("~/datasets/AC3/test")
FFTW.set_num_threads(6)

function main(labels)
	N=maximum(labels)
	kernel = fill(0f0, size(labels))
	kernel_size=[100,100,20]
	kernel[[1:x for x in kernel_size]...]=1
	kernel = circshift(kernel,[round(Int,-x/2) for x in kernel_size])

	plan = plan_fft(kernel, [1,2,3])
	iplan = inv(plan)
	fft_kernel=plan*(kernel)

	weights=fill(0f0, size(labels))
	S=sum(kernel)
	for i in 1:N
		begin
			println(i)
			indicator = map(x->x==i ? 1f0 : 0f0, labels)
			smoothed = iplan * (conj!(plan*(indicator)).*fft_kernel)
			weights[:,:,:] += real(smoothed.*indicator)
		end
	end
	for i in eachindex(weights)
		if weights[i] > 0
			weights[i] = S./weights[i]
		end
	end
	return weights
end

weights = main(load("$(DIR)/oracle_labels.h5"))
save("$(DIR)/oracle_weights.h5", weights)
