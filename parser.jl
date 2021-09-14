
function parser(logger_filepath :: String)
    
    epoch_value = 0

    training_loss = zeros(0)
    test_loss = zeros(0)

    training_accuracy = zeros(0)
    test_accuracy = zeros(0)
    
    open(logger_filepath, "r") do f
 
        # line_number
        line = 0
        is_training_loss = true
        is_training_acc = true

        # read till end of file
        while ! eof(f)

            # read a new / next line for every iteration          
            s = readline(f)

            words = split(s, " ", limit = 2)

            epoch = findall(x -> occursin("epoch", x), words)
            loss = findall(x -> occursin("loss", x), words)
            acc = findall(x -> occursin("acc", x), words)

            if ! isempty(epoch)
                epoch_substring = split(words[epoch][1], " ")[end]
                epoch_value = parse(Int, epoch_substring)
            end

            if ! isempty(loss)
                loss_substring = split(words[loss][1], " ")[end]
                loss_value = parse(Float64, loss_substring)

                if is_training_loss
                    append!(training_loss, loss_value)
                    is_training_loss = false
                else
                    append!(test_loss, loss_value)
                    is_training_loss = true
                end
            end

            if ! isempty(acc)
                acc_substring = split(words[acc][1], " ")[end]
                acc_value = parse(Float64, acc_substring)

                if is_training_acc
                    append!(training_accuracy, acc_value)
                    is_training_acc = false
                else
                    append!(test_accuracy, acc_value)
                    is_training_acc = true
                end
            end

            line += 1
        end

    end

    epochs = 1 : epoch_value + 1;

    return epochs, training_loss, test_loss, training_accuracy, test_accuracy

end
