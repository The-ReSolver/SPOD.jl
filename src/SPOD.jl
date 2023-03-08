module SPOD

using LinearAlgebra, FFTW

# ! Ways to improve the code:
# !     - include windowing;
# !     - type to treat snapshot array as vector;
# !     - use FFT plans to spead up loop processes.
# !     - clearly some parts of this are dying for views of arrays instead of assignment

# ! Julia points to outline before discussing code:
# !     - type of arguments in function definitions;
# !     - container type syntax;
# !     - indexing starts at 1;
# !     - unicode can be included;
# !     - nested loop syntax;
# !     - let them know what @view does;
# !     - explain the dot syntax.

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


snap2field!(u, q) = (u[:] = q[:]; return u)
field2snap(u) = u[:]

"""
    Take a set of DNS data and convert it to a snapshot matrix.
"""
function construct_snapshot_matrix(loc::String); end

"""
    Fourier transform the snapshot matrix in the homogeneous directions, and
    return the result as a similar snapshot matrix.
"""
function fft_homogeneous_directions(Q::Matrix, S::NTuple{3, Int})
    # initialise useful arrays
    Q_new = zeros(ComplexF64, prod((S[1], (S[2] >> 1) + 1, S[3])), size(Q, 2))
    u = zeros(Float64, S...)

    # loop over time dimensions
    for ti in 1:size(Q, 2)
        # convert snapshot vector to field array
        snap2field!(u, @view(Q[:, ti]))

        # fourier transform the data in streamwise and spanwise directions
        # TODO: does this need normalisation?
        û = FFTW.rfft(u, [1, 3])

        # convert back into snapshot vector for spectral space and assign to new snapshot matrix
        Q_new[:, ti] = field2snap(û)
    end

    return Q_new
end

"""
    Split the (Fourier transformed) snapshot matrix into Nf blocks in the time
    direction, with a specified overlap.
"""
function split_into_blocks(Q::Matrix, Nf::Int, No::Int)
    # compute the number of blocks
    Nb = Int(floor((size(Q, 2) - No)/(Nf - No)))

    # initialise vector to hold block arrays
    Q_blocks = [Matrix{ComplexF64}(undef, size(Q, 1), Nf) for _ in Nb]

    # loop over the blocks and take views of snapshot matrix
    for nb in 1:Nb
        block_index_dist = (nb - 1)*(Nf - No)
        Q_blocks[nb] = @view(Q[:, (1 + block_index_dist):(Nf + block_index_dist)])
    end

    return Q_blocks
end

"""
    Compute the Fourier transform of a single block of a total snapshot matrix. 
"""
fft_time(Q::Matrix) = FFTW.fft(Q, 2)

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
function spod(Q::Matrix{ComplexF64}, quad_weights::Vector{Float64}, Nf::Int, No::Int=0; window::WindowMethod=NoWindow(), eigrange::Union{Nothing, UnitRange}=nothing)
    # get size of snapshot vectors
    N = size(Q, 1)

    # split Q into blocks with overlap
    Q_blocks = split_into_blocks(Q, Nf, No)
    Nb = length(Q_blocks)

    # initialise matrix to hold FFT of snapshot matrix blocks
    Q̂_blocks = Vector{Matrix{ComplexF64}}(undef, length(Q_blocks))
    Nω = size(Q̂_blocks[1], 2)

    # loop over blocks and perform the time-wise FFT
    for (i, block) in enumerate(Q_blocks)
        Q̂_blocks[i] = fft_time(block)
    end

    # initialise useful arrays
    Qfk = zeros(ComplexF64, N, Nb)
    Mfk = zeros(ComplexF64, Nb, Nb)
    spod_modes = [Matrix{ComplexF64}(undef, N, Nb) for _ in 1:Nω]

    # # construct the weight matrix (quadrature + windowing)
    W = diag(quad_weights)

    # loop over the frequencies of all the blocks
    for fk in 1:Nω
        # construct fourier realisation matrices for each block
        for nb in 1:Nb
            Qfk[:, nb] .= @view(Q̂_blocks[nb][:, fk])
        end

        # compute the weights cross-spectral density matrix for each frequency
        Mfk .= adjoint(Qfk)*W*Qfk

        # compute the eigenvalue decomposition
        eigvals, eigvecs = eigen(Hermitian(Mfk), sortby=(x -> -real(x)))

        # take range if needed
        truncate_eigen!(eigvals, eigvecs, eigrange)

        # convert the eivenvectors to the correct SPOD modes
        spod_modes[fk] .= Qfk*eigvecs*Diagonal(eigvals.^-0.5)
    end

    return eigvals, spod_modes
end

truncate_eigen!(::Vector{T}, ::Matrix{Complex{T}}, ::Nothing) where {T} = nothing
truncate_eigen!(eigvals::Vector{T}, eigvecs::Matrix{Complex{T}}, eigrange::UnitRange) where {T} = (eigvals[eigrange]; eigvecs[:, eigrange]; return nothing)
end
