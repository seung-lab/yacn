using Save
DIR=expanduser("~/datasets/AC3/test")
FFTW.set_num_threads(6)

function f(labels)
	kernel = fill(0f0, size(labels))
	kernel_size=[100,100,20]
	kernel[[1:x for x in kernel_size]...]=1
	kernel = circshift(kernel,[-x/2 for x in kernel_size])

	plan = plan_fft(kernel, [1,2,3])
	iplan = inv(plan)
	fft_kernel=plan*(kernel)

	weights=fill(0f0, size(labels))
	S=sum(kernel)
	println(length(unique(labels)))
	f1=(x,y) -> conj(x)*y
	f2=(x,y) -> real(x*y)
	for i in unique(labels)
		begin
			println(i)
			indicator = map(x->x==i ? 1f0 : 0f0, labels)
			smoothed = iplan \ map(f1,plan*(indicator),fft_kernel)
			weights .+= map(f2, smoothed,indicator)
		end
	end
	return weights
end

include("forward.jl")

const patch_size = Int[300,300,50]

inpt = load("$(DIR)/oracle_labels.h5")
otpt = zeros(Float32, size(inpt))

forward!(inpt,otpt,patch_size, f, step=0.5)

for i in eachindex(otpt)
	if otpt[i] > 0
		otpt[i] = S./otpt[i]
	end
end

save("$(DIR)/experimental_oracle_weights.h5", otpt)
