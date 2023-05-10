# This file contains a number of useful functions and constructs that can be
# used to make the SPOD function more modular.

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
        copyto!(Q_blocks[nb], @view(Q[:, (1 + offset):(Nf + offset)]))
    end

    return Q_blocks, Nb
end

# ! Normalisation???
"""
    Compute the Fourier transform of a single block of a total snapshot matrix. 
"""
fft_time(Q::AbstractMatrix) = FFTW.fft(Q, 2)

truncate_eigen!(::AbstractVector, ::AbstractMatrix, ::Nothing) = nothing
truncate_eigen!(eigvals::AbstractVector, eigvecs::AbstractMatrix, eigrange::UnitRange) = (eigvals = eigvals[eigrange]; eigvecs = eigvecs[:, eigrange]; return nothing)
truncate_eigen!(eigvals::AbstractVector, eigvecs::AbstractMatrix, eigrange::Int) = (eigvals = [eigvals[eigrange]]; eigvecs = eigvecs[:, eigrange]; return nothing)
