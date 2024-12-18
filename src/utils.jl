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

"""
    Compute the Fourier transform of a single block of a total snapshot matrix. 
"""
fft_time(Q::AbstractMatrix) = FFTW.fft(Q, [2])./size(Q, 2)

function construct_weight_matrix(ws::AbstractVector, N::Int)
    Z = zeros((Int(N/3), Int(N/3)))
    W1 = Diagonal(ws)
    W =    [W1 Z  Z;
            Z  W1 Z;
            Z  Z  W1]

    return W
end

genModeArray(N, Nb, Nω, ::Nothing) = Array{ComplexF64, 3}(undef, N, Nb, Nω)
genModeArray(N, Nb, Nω, eigrange::Int) = Array{ComplexF64, 3}(undef, N, eigrange, Nω)

genEigvaluesArray(Nb, Nω, ::Nothing) = Matrix{Float64}(undef, Nb, Nω)
genEigvaluesArray(Nb, Nω, eigrange::Int) = Matrix{Float64}(undef, eigrange, Nω)

truncate_eigen!(::AbstractVector, ::AbstractMatrix, ::Nothing) = nothing
truncate_eigen!(eigvals::AbstractVector, eigvecs::AbstractMatrix, eigrange::Int) = (eigvals = eigvals[eigrange]; eigvecs = eigvecs[:, eigrange]; return nothing)
