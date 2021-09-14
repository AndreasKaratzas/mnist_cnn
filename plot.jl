
using Plots

include("parser.jl")


function plot_results(logger_filepath :: String = "")
    
    if isempty(logger_filepath)
        logger_filepath = joinpath(@__DIR__, "data", "log.txt")
    end

    epochs, training_loss, test_loss, training_accuracy, test_accuracy = parser(logger_filepath)
    loss = hcat(training_loss, test_loss)
    acc = hcat(training_accuracy, test_accuracy)

    xlabel!("Epochs")
    plot(epochs, loss, title = "Model Loss", label = ["Training" "Test"], lw = 2)
    savefig(joinpath(@__DIR__, "data", "loss.png"))

    xlabel!("Epochs")
    plot(epochs, acc, title = "Model Accuracy", label = ["Training" "Test"], lw = 2)
    savefig(joinpath(@__DIR__, "data","accuracy.png"))
end
