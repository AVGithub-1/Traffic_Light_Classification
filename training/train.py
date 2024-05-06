
#initialize hyperparameter tuner object
tuner_1 = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     factor=3,
                     project_name='hyperparam_tuning_results')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#make sure you're using GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPU devices:", physical_devices)

print(tuner_1.search_space_summary(extended=False))

with tf.device('/GPU:0'):
    tuner_1.search(train_set, batch_size = 32, validation_data=val_set, callbacks=[stop_early])


# Get the optimal hyperparameters
best_hps=tuner_1.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

best_hps_dict = best_hps.values
print(best_hps.values)



#create the final model using the best hyperparameters, then train it on the entire training set
final_model = models.Sequential()
final_model.add(layers.Conv2D(filters = best_hps['layer_0'], kernel_size = (3, 3), activation='relu', input_shape=(192,256, 3)))
final_model.add(layers.MaxPooling2D((2, 2)))
final_model.add(layers.Conv2D(filters = best_hps['layer_1'], kernel_size = (3, 3), activation='relu', input_shape=(192,256, 3)))
final_model.add(layers.MaxPooling2D((2, 2)))
final_model.add(layers.Flatten())
final_model.add(layers.Dense(units=best_hps['units'], activation='relu'))
final_model.add(layers.Dense(3))

final_model.compile(optimizer=optimizers.Adam(learning_rate=best_hps['learning_rate']),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

with tf.device('/GPU:0'):
    final_model.fit(train_set, epochs = 10, callbacks = stop_early, validation_data = test_set)


#plot the training and validation accuracy for each epoch of training
plt.plot(final_model.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#print out the accuracy on the test set
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)