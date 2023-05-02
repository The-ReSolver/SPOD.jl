module SPOD

using LinearAlgebra, FFTW

# ! Ways to improve the code:
# !     - include windowing;
# !     - type to treat snapshot array as vector;
# !     - use FFT plans to spead up loop processes;
# !     - clearly some parts of this are dying for views of arrays instead of assignment;
# !     - use in place mul! for the matrix multiplication

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

# TODO: check this fuckery works
apply_window!(Q::AbstractMatrix, ::NoWindow) = Q
implemented_windowing = [:NoWindow]
for window in window_types
    if window ∉ implemented_windowing
        @eval begin
            apply_window!(::AbstractMatrix, ::$window) = throw(ArgumentError($window, " windowing is not implement yet!"))
        end
    end
end


"""
    Split the (Fourier transformed) snapshot matrix into Nf blocks in the time
    direction, with a specified overlap.
"""
function split_into_blocks(Q::M, Nf::Int, No::Int) where {M <: AbstractMatrix}
    # compute the number of blocks
    Nb = floor(Int, (size(Q, 2) - No)/(Nf - No))

    # initialise vector to hold block arrays
    Q_blocks = [M(undef, size(Q, 1), Nf) for _ in 1:Nb]

    # loop over the blocks and take views of snapshot matrix
    for nb in 1:Nb
        offset = (nb - 1)*(Nf - No)
        # TODO: change to copyto! so that the original matrix isn't affected by windowing
        Q_blocks[nb] = @view(Q[:, (1 + offset):(Nf + offset)])
    end

    return Q_blocks, Nb
end

# ! Normalisation???
"""
    Compute the Fourier transform of a single block of a total snapshot matrix. 
"""
fft_time(Q::AbstractMatrix) = FFTW.fft(Q, 2)

"""
    spod(
        Q::Matrix{Float64},
        quad_weights::Vector{Float64},
        Nf::Int,
        No::Int;
        [window]::WindowMethod=NoWindow(),
        [eigrange]::Union{Nothing, UnitRange}=nothing,
    ) -> (AbstractVector{Float64}, AbstractMatrix{CompleF64})

    Compute the SPOD modes for a given snapshot matrix. Windowing is not
    currently supported. The decomposition can be truncated by passing a unit
    range to eigrange.
"""
function spod(Q::M, quad_weights::AbstractVector, dt::Float64, Nf::Int, No::Int=0; window::WindowMethod=NoWindow(), eigrange::Union{Nothing, Int, UnitRange}=nothing) where {M <: AbstractMatrix}
    # get size of snapshot vectors
    N = size(Q, 1)

    # split Q into blocks with overlap
    Q_blocks, Nb = split_into_blocks(Q, Nf, No)

    # initialise matrix to hold FFT of snapshot matrix blocks
    Q̂_blocks = Vector{typeof(Q)}(undef, length(Q_blocks))

    # loop over blocks and perform the time-wise FFT
    for (i, block) in enumerate(Q_blocks)
        Q̂_blocks[i] = fft_time(apply_window!(block, window))
    end
    Nω = size(Q̂_blocks[1], 2)

    # initialise useful arrays
    Qfk = M(undef, N, Nb)
    Mfk = M(undef, Nb, Nb)
    eigvals = Matrix{Float64}(undef, Nb, Nω)
    spod_modes = Array{ComplexF64, 3}(undef, N, Nb, Nω)

    # construct the weight matrix (quadrature + windowing)
    Z = zeros((Int(N/3), Int(N/3)))
    W1 = Diagonal(quad_weights)
    W =    [W1 Z  Z;
            Z  W1 Z;
            Z  Z  W1]

    # compute the realisation scaling constant
    # FIXME: including κ in the computation messes up the mode magnitude
    sqrt_κ = sqrt(dt/(Nf*Nb))

    # loop over the frequencies of all the blocks
    for fk in 1:Nω
        # construct fourier realisation matrices for each block
        for nb in 1:Nb
            Qfk[:, nb] .= sqrt_κ.*@view(Q̂_blocks[nb][:, fk])
        end

        # compute the weights cross-spectral density matrix for each frequency
        Mfk .= Qfk'*W*Qfk

        # compute the eigenvalue decomposition
        eigvals[:, fk], eigvecs = eigen(Hermitian(Mfk), sortby=(x -> -x))

        # take range if needed
        # FIXME: truncation is broken due to change in size not being compatible with size of original array
        truncate_eigen!(@view(eigvals[:, fk]), eigvecs, eigrange)

        # convert the eivenvectors to the correct SPOD modes
        spod_modes[:, :, fk] .= Qfk*eigvecs*Diagonal(@view(eigvals[:, fk]).^-0.5)
    end

    return eigvals, spod_modes
end

truncate_eigen!(::AbstractVector, ::AbstractMatrix, ::Nothing) = nothing
truncate_eigen!(eigvals::AbstractVector, eigvecs::AbstractMatrix, eigrange::UnitRange) = (eigvals = eigvals[eigrange]; eigvecs = eigvecs[:, eigrange]; return nothing)
truncate_eigen!(eigvals::AbstractVector, eigvecs::AbstractMatrix, eigrange::Int) = (eigvals = [eigvals[eigrange]]; eigvecs = eigvecs[:, eigrange]; return nothing)
end
