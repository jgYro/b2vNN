using Flux
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
test_x, test_y = get_test_data()

train_y = Flux.onehotbatch(train_y, 0:1)
test_y = Flux.onehotbatch(test_y, 0:1)



σ(x) = 1 / (1 + exp(-x))

methods(Flux.mse)

model = Chain(
    x -> reshape(x, :, size(x, 4)),  # Flatten the input data into a vector
    Dense(65536=>15, relu),
    Dense(15=>2, sigmoid),
    softmax
)

function loss(model, x, y)
	return Flux.crossentropy(model(x),Flux.onehotbatch(y,0:1))
end


optimizer = Flux.Optimise.ADAM()

predict = model.(train_x)

