module SPOD

using LinearAlgebra, FFTW

export spod

# ! Ways to improve the code:
# !     - include windowing;
# !     - type to treat snapshot array as vector;
# !     - use FFT plans to spead up loop processes;
# !     - clearly some parts of this are dying for views of arrays instead of assignment;
# !     - use in place mul! for the matrix multiplication

include("utils.jl")
include("window.jl")

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
function spod(Q::M, ws::AbstractVector, dt::Float64, Nf::Int, No::Int=0; verbose::Bool=false, window::WindowMethod=NoWindow(), eigrange::Union{Nothing, Int, UnitRange}=nothing) where {M <: AbstractMatrix}
    # get size of snapshot vectors
    N = size(Q, 1)

    # split Q into blocks with overlap
    verbose && print("Splitting snapshot matrix into blocks...")
    Q_blocks, Nb = split_into_blocks(Q, Nf, No)
    verbose && println("Done!")

    # initialise matrix to hold FFT of snapshot matrix blocks
    Q̂_blocks = Vector{typeof(Q)}(undef, length(Q_blocks))

    # loop over blocks and perform the time-wise FFT
    verbose && print("Fourier transforming snapshot blocks... ")
    for (i, block) in enumerate(Q_blocks)
        Q̂_blocks[i] = fft_time(apply_window!(block, window))
    end
    verbose && println("Done!")
    Nω = size(Q̂_blocks[1], 2)

    # initialise useful arrays
    Qfk = M(undef, N, Nb)
    Mfk = M(undef, Nb, Nb)
    eigvals = Matrix{Float64}(undef, Nb, Nω)
    spod_modes = Array{ComplexF64, 3}(undef, N, Nb, Nω)

    # construct the weight matrix (quadrature + windowing)
    W = construct_weight_matrix(ws, N)

    # compute the realisation scaling constant
    # FIXME: including κ in the computation messes up the mode magnitude
    sqrt_κ = sqrt(dt/(window_factor(Nf, window)*Nb))

    # loop over the frequencies of all the blocks
    for fk in 1:Nω
        verbose && print("Solving Eigenproblem for every frequency... fk = ", fk, "/", Nω, "\r")
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
    verbose && println("Solving Eigenproblem for every frequency... Done!                     ")

    return eigvals, spod_modes
end

end
