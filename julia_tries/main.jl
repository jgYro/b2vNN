using Flux
using Flux: onehotbatch, logitcrossentropy, params
using Images

function get_test_data()
    path = "test"
    x_test = []
    y_test = Float32[]
    for label in readdir(path)
        for file in readdir(joinpath(path, label))
            img = load(joinpath(path, label, file))
            img = imresize(img, (256, 256))
            data = Float32.(reshape(channelview(Gray.(img)), 256, 256))
            push!(x_test, data)
            push!(y_test, parse(Float32, label))
        end
    end
    return x_test, y_test
end

function get_train_data()
    path = "train"
    x_train = []
    y_train = Float32[]
    for label in readdir(path)
        for file in readdir(joinpath(path, label))
            img = load(joinpath(path, label, file))
            img = imresize(img, (256, 256))
            data = Float32.(reshape(channelview(Gray.(img)), 256, 256))
            push!(x_train, data)
            push!(y_train, parse(Float32, label))
        end
    end
    return x_train, y_train
end

train_x, train_y = get_train_data()
train_x = [reshape(vec(matrix), 65536) for matrix in train_x]

test_x, test_y = get_test_data()
test_x = [reshape(vec(matrix), 65536) for matrix in test_x]

# Define your model
model = Chain(
    Dense(65536 => 15, relu),
    Dense(15 => 2, sigmoid),
    softmax
)


loss(x, y) = Flux.crossentropy(model(x), onehotbatch(y, 0:1))

optimizer = ADAM()

function train!(model, x, y, optimizer)
    gs = gradient(params(model)) do
        _ = loss(x, y)
    end
    Flux.update!(optimizer, params(model), gs)
end

ps = Flux.params(model)

num_epochs = 10
for epoch in 1:num_epochs
    Flux.train!(loss, ps, zip(train_x, train_y), optimizer)
    println("Epoch $epoch completed")
    # Calculate accuracy on test data
    correct_count = 0
    total_count = 0
    for i in 1:length(test_x)
        output = model(test_x[i])
        prediction = argmax(output) - 1  # Adjust prediction index to match label encoding
        true_label = test_y[i]
        println("Output: $output, Prediction: $prediction, True Label: $true_label")
        if prediction == true_label
            correct_count += 1
        end
        total_count += 1
    end
    accuracy = correct_count / total_count
    println("Accuracy at epoch $epoch: $accuracy")
end

bin = load("./bin.png")
bin = imresize(bin, (256, 256))
bin = Float32.(reshape(channelview(Gray.(bin)), 256, 256))
bin = reshape(vec(bin), 65536)
println("Printing Result for Bin.png")
predict = model(bin)

pic = load("./pic.png")
pic = imresize(pic, (256, 256))
pic = Float32.(reshape(channelview(Gray.(pic)), 256, 256))
pic = reshape(vec(pic), 65536)
println("Printing Result for Pic.png")
predict = model(pic)
