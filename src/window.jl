# This file contains the definition of the interface to allow dispatch on the
# type of desired window.

# define abstract widnowing type to allow dispatch on specific window used
abstract type WindowMethod end

# these types immitate the options available from Matlab's spectrum.welch method
window_types = [:NoWindow,       :Hamming,     :Bartlett,   :BartlettHann,
                :BlackmanHarris, :Bohman,      :Chebyshev,  :FlatTop,
                :Gaussian,       :Hann,        :Kaiser,     :Nuttall,
                :Parzen,         :Rectangular, :Triangular, :Tukeym,
                :Welch,          :Sine]
for window in window_types
    @eval begin
        struct $window <: WindowMethod end
        export $window
    end
end

# define window functions
window_func(x::Float64, ::NoWindow) = 1.0
window_func(x::Float64, ::Hann) = 0.5*(1 - cos(2π*x))
window_func(x::Float64, ::Welch) = 1 - (2*x - 1)^2
window_func(x::Float64, ::Sine) = sin(π*x)
window_func(x::Float64, ::Hamming) = 0.53836 + 0.46164*cos(2π*t)

apply_window!(Q::AbstractMatrix, ::NoWindow) = Q
function apply_window!(Q::AbstractMatrix, window::WindowMethod)
    for n in axes(Q, 2)
        @view(Q[:, n]) .*= window_func(n/size(Q, 2), window)
    end

    return Q
end

window_factor(Nf::Int, ::NoWindow) = Nf
window_factor(Nf::Int, window::WindowMethod) = sum(window_func(n/Nf, window)^2 for n in 0:Nf)

# define fallback methods for unimplemented windows that throw errors
implemented_windowing = [:NoWindow, :Hann, :Welch, :Sine, :Hamming]
for window in window_types
    if window ∉ implemented_windowing
        @eval begin
            window_func(::Float64, ::$window) = throw(ArgumentError(string($window)*" windowing is not implemented yet!"))
        end
    end
end
