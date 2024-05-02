import tensorflow as tf

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./225,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(150, 150),
    batch_size = 32,
    class_mode='categorical',
    subset='training',
    save_to_dir='processed_images/',
    save_prefix='aug_',
    save_format='png'
)

for i in range(10):
    img = next(train_generator)

validation_generator = train_datagen.flow_from_directory(
    'processed_images/',
    target_size=(150, 150),
    batch_size = 32,
    class_mode='categorical',
    subset='validation' 
)