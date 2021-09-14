
using Flux: onecold
using Flux.Losses: logitcrossentropy

loss(y_hat, y) = logitcrossentropy(y_hat, y)
num_params(model) = sum(length, Flux.params(model)) 
round4(x) = round(x, digits=4)

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        y_hat = model(x)
        l += loss(y_hat, y) * size(x)[end]        
        acc += sum(onecold(y_hat |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end
