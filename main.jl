
include("src/engine.jl")
include("src/plot.jl")


# arguments for the `train` function
Base.@kwdef mutable struct Args
    heta = 3e-4                             # learning rate
    lambda = 0                              # L2 regularizer param, implemented as weight decay
    batchsize = 128                         # batch size
    epochs = 10                             # number of epochs
    seed = 0                                # set seed > 0 for reproducibility
    use_cuda = true                         # if true use cuda (if available)
    infotime = 1                            # report every `infotime` epochs
    checktime = 5                           # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    logger = true                           # log training
    savepath = joinpath(@__DIR__, "data")   # results path
end


function main()
    args = Args()
    rm(joinpath(@__DIR__, "data", "log.txt"), force=true)
    train(args)
    plot_results(joinpath(@__DIR__, "data"))
end
