# Load libraries
library(keras)
library(tidyr)
library(caret)
library(jpeg)

# Specify the path to the directory containing your image files
image_directory <- "path/to/your/images"

# List all files in the directory with a .jpg extension (adjust for other formats)
image_files <- list.files(image_directory, pattern = ".jpg", full.names = TRUE)

# Read the images into a list
pneumonia_X_Ray <- lapply(image_files, readJPEG)

# To visualize an image, you can use the 'image()' function from the 'jpeg' package
image(pneumonia_X_Ray[[1]], axes = FALSE, col = grey.colors(256))

pneumonia_X_Ray <- "pneumonia_X_Ray"
train_path <- file.path(pneumonia_X_Ray, "train")
test_path <- file.path(pneumonia_X_Ray, "test")

# Image dimensions and batch size
img_width <- 712
img_height <- 712
batch_size <- 439

# Data preprocessing and augmentation
train_datagen <- image_data_generator(
  rescale = 1/255,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_path,
  train_datagen,
  target_size = c(img_width, img_height),
  batch_size = batch_size,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  test_path,
  test_datagen,
  target_size = c(img_width, img_height),
  batch_size = batch_size,
  class_mode = "binary"
)

# CNN model
cnn_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(img_width, img_height, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

cnn_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001),
  metrics = c("accuracy")
)

# Train CNN model
history <- cnn_model %>% fit(
  train_generator,
  steps_per_epoch = as.integer(2000 / batch_size),
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = as.integer(500 / batch_size)
)

# Evaluate the model
evaluation <- cnn_model %>% evaluate(validation_generator, steps = as.integer(500 / batch_size))
cat("Accuracy on the validation set:", evaluation$accuracy, "\n")
