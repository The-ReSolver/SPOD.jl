# This file contains the definition of the interface to allow dispatch on the
# type of desired window.

# define abstract widnowing type to allow dispatch on specific window used
abstract type WindowMethod end

# these types immitate the options available from Matlab's spectrum.welch method
window_types = [:NoWindow,       :Hamming,     :Bartlett,   :BartlettHann,
                :BlackmanHarris, :Bohman,      :Chebyshev,  :FlatTop,
                :Gaussian,       :Hann,        :Kaiser,     :Nuttall,
                :Parzen,         :Rectangular, :Triangular, :Tukeym]
for window in window_types
    @eval begin
        struct $window <: WindowMethod end
    end
end

apply_window!(Q::AbstractMatrix, ::NoWindow) = Q

# define fallback methods for unimplemented windows that throw errors
# TODO: check this fuckery works
implemented_windowing = [:NoWindow]
for window in window_types
    if window ∉ implemented_windowing
        @eval begin
            apply_window!(::AbstractMatrix, ::$window) = throw(ArgumentError($window, " windowing is not implement yet!"))
        end
    end
end
