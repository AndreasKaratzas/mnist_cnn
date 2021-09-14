
using Flux
using Logging
using Statistics, Random
using Flux: onecold, logitcrossentropy
using Flux.Optimise: Optimiser, WeightDecay
using ProgressMeter: @showprogress
using CUDA

import BSON

include("dataloader.jl")
include("model.jl")
include("utils.jl")


function train(args)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet5() |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"    
    
    ps = Flux.params(model)

    opt = ADAM(args.heta) 
    if args.lambda > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.lambda), opt)
    end

    mkpath(args.savepath)
    
    ## LOGGING UTILITIES
    if args.logger
        @info "Logging at \"$(args.savepath)\""
    end
    
    function report(epoch, logging)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if logging
            logger_filepath = joinpath(args.savepath, "log.txt")
            io = open(logger_filepath, "a")
            logger = SimpleLogger(io)
            with_logger(logger) do
                @info "train" epoch=epoch loss=train.loss  acc=train.acc
                @info "test"  epoch=epoch loss=test.loss   acc=test.acc
            end
            close(io)
        end
    end
    
    ## TRAINING
    @info "Start Training"
    report(0, args.logger)
    for epoch in 1:args.epochs
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                    y_hat = model(x)
                    loss(y_hat, y)
                end

            Flux.Optimise.update!(opt, ps, gs)
        end
        
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch, args.logger)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson") 
            let model = cpu(model) #return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end
