using Flux
using Flux: Conv, relu, MaxPool, Dense, BatchNorm, Chain, params
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, update!
using Flux: binarycrossentropy, Momentum
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
train_x = [reshape(vec(matrix), (256, 256, 1, 1)) for matrix in train_x]


test_x, test_y = get_test_data()
test_x = [reshape(vec(matrix), (256, 256, 1, 1)) for matrix in test_x]

# Define the CNN model
function create_model()
    return Chain(
        Conv((3, 3), 1=>32, relu),
        Conv((3, 3), 32=>64, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 64=>128, relu),
        Conv((3, 3), 128=>256, relu),
        MaxPool((2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(952576, 128, relu),
        BatchNorm(128),
        Dense(128, 1)
    )
end

# Initialize model
model = create_model()

# Define loss function
loss(x, y) = binarycrossentropy(model(x), y)

# Initialize optimizer
opt = Momentum()

# DataLoader for training and test data
train_data = zip(train_x, train_y)
test_data = zip(test_x, test_y)

# Training loop
function train_model!(model, data, opt)
    for (x, y) in data
        gs = gradient(params(model)) do
            _ = loss(x, y)
        end
        update!(opt, params(model), gs)
    end
end

# Evaluation function
function evaluate(model, data)
    accs = []
    for (x, y) in data
        pred = model(x) .> 0.5
        acc = pred .== y / length(y)
        push!(accs, acc)
    end
    return accs / length(accs)
end

# Training
epochs = 10
for epoch in 1:epochs
    train_model!(model, train_data, opt)
    train_acc = evaluate(model, train_data)
    test_acc = evaluate(model, test_data)
    println("Epoch $epoch: Train accuracy = $train_acc, Test accuracy = $test_acc")
end

