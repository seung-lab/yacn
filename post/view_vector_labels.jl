using Base.Threads
using FixedSizeArrays
using ArrayUtils
using Util
using Save


using GLVisualize,Reactive,GeometryTypes,GLAbstraction
using Colors

ranges=(1:500,1:500,1:50)
vector_labels = Save.load("$(ARGS[1])")#[ranges...,:]
nvec_labels = size(vector_labels,4)
const vec_labels = Vec{nvec_labels,Float32}[Vec{nvec_labels,Float32}(vector_labels[i,j,k,:]) for i in 1:size(vector_labels,1), j in 1:size(vector_labels,2), k in 1:size(vector_labels,3)]

using Colors
cmap=colormap("RdBu")

function torgb(x)
	cmap[max(min(round(Int,x*100),100),1)]
end
function affinity(u,v)
	tmp=u-v
	return exp(-0.5*dot(tmp,tmp))
end

function field_affinity(u,vec_labels)
	tmp=zeros(Float32,size(vec_labels))
	@threads for i in 1:length(tmp)
		tmp[i] = affinity(vec_labels[i],u)
	end
	return tmp
end

const focus_vec=Signal(Vec{nvec_labels,Float32}(0))
const raw_image = map(v->field_affinity(v,vec_labels),focus_vec)
const coloured_image = map(x->map(torgb,x),raw_image)
x,y,z = size(vec_labels)
const zmax = z

window = glscreen("viewer", background = RGBA(0.1,0.1,0.1,1))
slice_index=Signal(round(Int,value(zmax)/2))

slice = map((A,i)->A[:,:,i], coloured_image, slice_index)
preserve(map(window.inputs[:scroll]) do s
	if Int(s[2]) != 0
		push!(slice_index, max(1,min(value(zmax),value(slice_index)-Int(s[2]))))
	end
	nothing
end)
tm = map(window.inputs[:window_size]) do s
	scalematrix(Vec{3,Float32}(s[1]/x, s[2]/y, 0f0))
end

mouse_buttons_pressed = window.inputs[:mouse_buttons_pressed]
key_pressed = const_lift(GLAbstraction.singlepressed, mouse_buttons_pressed, GLFW.MOUSE_BUTTON_LEFT)
preserve(map(key_pressed) do lp
	if lp
		mx,my=value(window.inputs[:mouseposition])./(value(window.inputs[:window_size]))
		mx=round(Int,mx*x)
		my=round(Int,(1-my)*y)
		push!(focus_vec,vec_labels[mx,my,value(slice_index)])
	end
	nothing
end)

slice_context = visualize(slice, model=tm, boundingbox=nothing)
view(slice_context, window, camera=:fixed_pixel)
renderloop(window)
