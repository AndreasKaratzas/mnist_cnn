
using Plots

include("parser.jl")


function plot_results(data_path :: String = "")
    
    logger_filepath = ""

    if isempty(data_path)
        logger_filepath = joinpath(@__DIR__, "data", "log.txt")
    else
        logger_filepath = joinpath(data_path, "log.txt")
    end

    epochs, training_loss, test_loss, training_accuracy, test_accuracy = parser(logger_filepath)
    loss = hcat(training_loss, test_loss)
    acc = hcat(training_accuracy, test_accuracy)

    xlabel!("Epochs")
    plot(epochs, loss, title = "Model Loss", label = ["Training" "Test"], lw = 2)
    savefig(joinpath(data_path, "loss.png"))

    xlabel!("Epochs")
    plot(epochs, acc, title = "Model Accuracy", label = ["Training" "Test"], lw = 2)
    savefig(joinpath(data_path, "accuracy.png"))
end
