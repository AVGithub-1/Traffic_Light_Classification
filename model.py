def model_builder(hp):
  """
  custom model building function for hyperparameter tuning that creates a Keras Sequential model object and helps tune the number of layers,
  number of units in each layer, and learning rate of the model
  hp()
  """
  model = models.Sequential()

  #hyperparameter for number of layers
  #hyperparameter for filters in each layer
  #hyperparameter for window size in each conv2D layer

  hp_num_conv = 2

  hp_layers_list = []
  for i in range(hp_num_conv):
    hp_layer = hp.Choice(name = 'layer_'+str(i), values = [32, 64])
    hp_layers_list.append(hp_layer)


  for i in hp_layers_list:
    model.add(layers.Conv2D(filters = i, kernel_size = (3, 3), activation='relu', input_shape=(192,256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Flatten())
  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=128, max_value=512, step=128)
  model.add(layers.Dense(units=hp_units, activation='relu'))
  model.add(layers.Dense(3))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01 or 0.001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

  model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model