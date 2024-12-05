module SPOD

using LinearAlgebra, FFTW

export spod, spod!

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
function spod!(eigvals::AbstractMatrix{Float64}, spod_modes::AbstractArray{ComplexF64, 3}, Q::M, ws::AbstractVector, dt::Float64, Nf::Int, No::Int=0; verbose::Bool=false, window::WindowMethod=NoWindow(), eigrange::Union{Nothing, Int}=nothing) where {M<:AbstractMatrix}
    # get size of snapshot vectors
    N = size(Q, 1)

    # split Q into blocks with overlap
    verbose && print("Splitting snapshot matrix into blocks...")
    Q_blocks, Nb = split_into_blocks(Q, Nf, No)
    eigrange = isnothing(eigrange) ? Nb : eigrange
    verbose && println("Done!"); flush(stdout)

    # initialise matrix to hold FFT of snapshot matrix blocks
    Q̂_blocks = Vector{typeof(Q)}(undef, length(Q_blocks))

    # loop over blocks and perform the time-wise FFT
    verbose && print("Fourier transforming snapshot blocks... ")
    for (i, block) in enumerate(Q_blocks)
        Q̂_blocks[i] = fft_time(apply_window!(block, window))
    end
    verbose && println("Done!"); flush(stdout)
    Nω = size(Q̂_blocks[1], 2)

    # initialise useful arrays
    Qfk = M(undef, N, Nb)
    Mfk = M(undef, Nb, Nb)

    # construct the weight matrix (quadrature + windowing)
    W = construct_weight_matrix(ws, N)

    # compute the realisation scaling constant
    sqrt_κ = sqrt(dt/(window_factor(Nf, window)*Nb))

    # loop over the frequencies of all the blocks
    for fk in 1:Nω
        verbose && print("Solving Eigenproblem for every frequency... fk = ", fk, "/", Nω, "\r"); flush(stdout)
        # construct fourier realisation matrices for each block
        for nb in 1:Nb
            Qfk[:, nb] .= sqrt_κ.*@view(Q̂_blocks[nb][:, fk])
        end

        # compute the weights cross-spectral density matrix for each frequency
        Mfk .= Qfk'*W*Qfk

        # compute the eigenvalue decomposition
        vals, eigvecs = eigen(Hermitian(Mfk), sortby=(x->-abs(x)))
        eigvals[:, fk] .= @view(vals[1:eigrange])
        spod_modes[:, :, fk] .= Qfk*@view(eigvecs[:, 1:eigrange])*Diagonal(abs.(@view(eigvals[:, fk])).^-0.5)
    end
    verbose && println("Solving Eigenproblem for every frequency... Done!                     "); flush(stdout)

    return eigvals, spod_modes
end
spod(Q::M, ws::AbstractVector, dt::Float64, Nf::Int, No::Int=0; verbose::Bool=false, window::WindowMethod=NoWindow(), eigrange::Union{Nothing, Int}=nothing) where {M<:AbstractMatrix} = spod!(genEigvalues(Nb, Nω, eigrange), genModeArray(N, Nb, Nω, eigrange), Q, ws, dt, Nf, No; verbose=verbose, window=window, eigrange=eigrange)

end
