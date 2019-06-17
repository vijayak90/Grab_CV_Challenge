import keras
from resnet_152 import resnet152_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

# Setting the image metadata along with the training metadata
iw, ih = 224, 224
num_channels = 3
train_data = 'data/train'
valid_data = 'data/valid'
g_classes = 196
cnt_train_samples = 6549
cnt_valid_samples = 1595
verbose = 1
batch_size = 8
num_epochs = 10
patience = 1

if __name__ == '__main__':
   # Classifier model based on resnet_152 model
   model = resnet152_model(ih, iw, num_channels, g_classes)

   # Configuration of Data Augumentation
   train_data_gen = ImageDataGenerator(rotation_range=20.,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
   valid_data_gen = ImageDataGenerator()
   # Defining the callbacks and storing the models
   tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
   log_file_path = 'logs/training.log'
   csv_logger = CSVLogger(log_file_path, append=False)
   # Early stopping of training to reduce overfitting, and improve generalization
   early_stop = EarlyStopping('val_acc', patience=patience)
   # Reduce learning rate when a metric has stopped improving to save tuning time
   reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)
   #Path to store the trained models
   trained_models_path = 'models/model'
   model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
   model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
   callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

   # training and validation generators
   train_generator = train_data_gen.flow_from_directory(train_data, (iw, ih), batch_size=batch_size,
                                                        class_mode='categorical')
   valid_generator = valid_data_gen.flow_from_directory(valid_data, (iw, ih), batch_size=batch_size,
                                                        class_mode='categorical')
   print('Starting to fine tune the model!')
   #Fine tuning the model
   model.fit_generator(
       train_generator,
       steps_per_epoch=cnt_train_samples / batch_size,
       validation_data=valid_generator,
       validation_steps=cnt_valid_samples / batch_size,
       epochs=num_epochs,
       callbacks=callbacks,
       verbose=verbose)
