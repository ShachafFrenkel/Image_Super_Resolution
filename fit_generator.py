# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
#from pyimagesearch.minivggnet import MiniVGGNet
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import cv2
from smallvggnet import SmallVGGNet
import VGG
from simple_NN import create_cnn
# def csv_image_generator(inputPath, bs, mode="train", aug=None):
#     f = open(inputPath, "r")
#     while True:
# # initialize our batches of images and labels
#         images_blur = []
#         images = []
# # keep looping until we reach our batch size
#         while len(images_blur) < bs:
# # attempt to read the next line of the CSV file
#             line = f.readline()
# # check to see if the line is empty, indicating we have
# # reached the end of the file
#             if line == "":
# # reset the file pointer to the beginning of the file
# # and re-read the line
#                 f.seek(0)
#                 line = f.readline()
# # if we are evaluating we should now break from our
# # loop to ensure we don't continue to fill up the
# # batch from samples at the beginning of the file
#                 if mode == "eval":
#                     break
# # extract the label and construct the image
#             #line = line.strip().split(",")
#             #label = line[0]
#             image = np.array([int(x) for x in line[1:]], dtype="float32")
#             rand_int_1 = rand_int_1(1,1800)
#             rand_int_2 = rand_int_2(1,1800)
#             image = image[rand_int_1:rand_int_1+50,rand_int_2:rand_int_2+50]
#             image = image.reshape((50, 50, 1))
#             blur_image = cv2.GaussianBlur((Image.fromarray(image)).reshape((50, 50, 1)),(5,5),cv2.BORDER_DEFAULT)
# # update our corresponding batches lists
#             images.append(image)
#             images_blur.append(blur_image)


def image_generator_train( bs, mode="train", aug=None):
    while True:
        n=0
    #while n<20:
# initialize our batches of images and labels
        images_blur = []
        images = []
        i=0
# keep looping until we reach our batch size
        while len(images_blur) < bs:
            if mode == "eval":
                break
            image = cv2.imread(f'C:\\Users\\DavidS10\\PycharmProjects\\pythonProject\\original\\image_{i}_cv2.jpg',
                               cv2.IMREAD_UNCHANGED)
            rand_int_1 = random.randint(1,1600)
            rand_int_2 = random.randint(1,1600)
            image = image[rand_int_1:rand_int_1+128,rand_int_2:rand_int_2+128]
            image = image.reshape((128, 128))
            pil_img = Image.fromarray(image)
            cv2_img = np.array(pil_img)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
            blur_image = cv2.GaussianBlur(cv2_img,(15,15),cv2.BORDER_DEFAULT)
            gray_blur_img = cv2.cvtColor(blur_image, cv2.COLOR_RGB2GRAY)
            blur_array = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
# update our corresponding batches lists
            image = image.reshape((128,128,1))
            gray_blur_img = gray_blur_img.reshape((128, 128,1))
            images.append(image)
            images_blur.append(gray_blur_img)
            print("train", i)
            i+=1
            # print(len(images_blur))
            # print(np.array(images_blur).shape)

            # return images[0],images_blur[0]
            # one-hot encode the labels
            #images = lb.transform(np.array(images))
            #if the data augmentation object is not None, apply it
            # if aug is not None:
            #     ( images_blur, images) = next(aug.flow(images_blur, images, batch_size=bs))
            # # yield the batch to the calling function
        n += 1
        #shape = np.array(images_blur).shape
        yield np.array(images_blur),np.array(images)

def image_generator_test( bs, mode="train", aug=None):
    while True:
        n=0
    #while n<20:
# initialize our batches of images and labels
        images_blur = []
        images = []
        i=0
# keep looping until we reach our batch size
        while len(images_blur) < bs:
            # if mode == "eval":
            #    break
            image = cv2.imread(f'C:\\Users\\DavidS10\\PycharmProjects\\pythonProject\\original\\image_{i}_cv2.jpg',
                               cv2.IMREAD_UNCHANGED)
            rand_int_1 = random.randint(1,1600)
            rand_int_2 = random.randint(1,1600)
            image = image[rand_int_1:rand_int_1+128,rand_int_2:rand_int_2+128]
            image = image.reshape((128, 128))
            pil_img = Image.fromarray(image)
            cv2_img = np.array(pil_img)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
            blur_image = cv2.GaussianBlur(cv2_img,(15,15),cv2.BORDER_DEFAULT)
            gray_blur_img = cv2.cvtColor(blur_image, cv2.COLOR_RGB2GRAY)
            blur_array = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
# update our corresponding batches lists
            image = image.reshape((128,128,1))
            gray_blur_img = gray_blur_img.reshape((128, 128,1))
            images.append(image)
            images_blur.append(gray_blur_img)
            print("val",i)
            i+=1
            # print(len(images_blur))
            # print(np.array(images_blur).shape)

            # return images[0],images_blur[0]
            # one-hot encode the labels
            #images = lb.transform(np.array(images))
            # if the data augmentation object is not None, apply it
            #if aug is not None:
            #    ( images_blur,images) = next(aug.flow(np.array(images_blur),
            #                                     np.array(images), batch_size=bs))
            # yield the batch to the calling function
        n += 1

        if mode=="eval":
            tifffile.imsave("x_tif.tiff", np.array(images_blur))

        yield np.array(images_blur),np.array(images)

# A different dataset for test: (need to add)
# def image_generator_test_1( bs, mode="train", aug=None):
#     print("here")
#     while True:
#     # n=0
#     # while n<20:
# # initialize our batches of images and labels
#         images_blur = []
#         images = []
#         i=0
# # keep looping until we reach our batch size
#         while len(images_blur) < bs:
#             # if mode == "eval":
#             #     break
#             image = cv2.imread(f'C:\\Users\\DavidS10\\PycharmProjects\\pythonProject\\original\\image_{i}_cv2.jpg',
#                                cv2.IMREAD_UNCHANGED)
#             rand_int_1 = random.randint(1,1800)
#             rand_int_2 = random.randint(1,1800)
#             image = image[rand_int_1:rand_int_1+64,rand_int_2:rand_int_2+64]
#             image = image.reshape((64, 64,1))
#             pil_img = Image.fromarray(image)
#             cv2_img = np.array(pil_img)
#             cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
#             blur_image = cv2.GaussianBlur(cv2_img,(5,5),cv2.BORDER_DEFAULT)
#             blur_array = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
# # update our corresponding batches lists
#             images.append(image)
#             images_blur.append(blur_image)
#             i+=1
#     #         print(i)
#     #     n+=1
#     # return images[0],images_blur[0]
#             # one-hot encode the labels
#             images = lb.transform(np.array(images))
#             # if the data augmentation object is not None, apply it
#             if aug is not None:
#                 (images_blur, images) = next(aug.flow(np.array(images_blur),
#                                                  images, batch_size=bs))
#             # yield the batch to the calling function
#         yield (np.array(images_blur), images)


# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 1000
BS = 64
# initialize the total number of training and testing image
NUM_TRAIN_IMAGES = 1024
NUM_TEST_IMAGES = 1024
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
# initialize both the training and testing image generators
trainGen = image_generator_train( BS,
	mode="train", aug=aug)
testGen = image_generator_test( BS,
	mode="train", aug=None)
# initialize our Keras model and compile it (NEED TO WRITE DIFFERENT NN AND CALL IT HERE)
#model = create_cnn(64, 64,1)
#model = SmallVGGNet.build(64,64,1)
model = VGG.create_net(1,1,1,2,2,(128,128,1))
opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-5, momentum=0.9, decay=1e-2 / NUM_EPOCHS)
model.compile(loss="mean_squared_error", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training w/ generator...")
H = model.fit(
	trainGen,
	steps_per_epoch=NUM_TRAIN_IMAGES // BS,
	validation_data=testGen,
	validation_steps=NUM_TEST_IMAGES // BS,
	epochs=NUM_EPOCHS)
# re-initialize our testing data generator, this time for evaluating
testGen = image_generator_test( BS,
	mode="eval", aug=None)
print(model.summary())
# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability
#predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
# print("[INFO] evaluating network...")
# print(classification_report(testLabels.argmax(axis=1), predIdxs,
# 	target_names=lb.classes_))
# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
#plt.ylim(0,min(H.history["loss"])*10)
plt.legend(loc="lower left")
plt.ylim(0,min(H.history["loss"])*10)
plt.savefig("plot_train_loss.png")
plt.figure()
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.ylim(0,min(H.history["val_loss"])*10)
plt.legend(loc="lower left")
plt.savefig("plot_val_loss.png")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
#plt.ylim(0,min(H.history["loss"])*10)
plt.legend(loc="lower left")
plt.savefig("plot_train_acc.png")
plt.figure()
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
#plt.ylim(0,min(H.history["loss"])*10)
plt.legend(loc="lower left")
plt.savefig("plot_val_acc.png")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# #plt.ylim(0,min(H.history["loss"])*10)
# plt.legend(loc="lower left")
# plt.savefig("plot.png")
print("min train loss", min(H.history["loss"]))
print("min val loss", min(H.history["val_loss"]))
predIdxs = model.predict(x=testGen, steps=(NUM_TEST_IMAGES // BS) + 1)
pred_array=np.asarray(predIdxs)
print("pred array shape",pred_array.shape)
tifffile.imsave("predict_tif.tiff",pred_array)
img_for_test = cv2.imread('C:\\Users\\DavidS10\\PycharmProjects\\pythonProject\\original\\image_0_cv2.jpg',cv2.IMREAD_UNCHANGED)
image = img_for_test[1600:1728,1600:1728]
print(image.shape)
image = image.reshape((128, 128))
pil_img = Image.fromarray(image)
cv2_img = np.array(pil_img)
cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
blur_image = cv2.GaussianBlur(cv2_img,(15,15),cv2.BORDER_DEFAULT)
gray_blur_img = cv2.cvtColor(blur_image, cv2.COLOR_RGB2GRAY)
blur_array = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
# update our corresponding batches lists
# image = image.reshape((1,64,64))
#gray_blur_img = gray_blur_img.reshape((1,1,64, 64))
image = image.reshape((128,128,1))
gray_blur_img = gray_blur_img.reshape((1,128, 128,1))
predIdxs = model.predict(gray_blur_img)
tifffile.imsave("predict.tiff",predIdxs)
tifffile.imsave("x_to_predict.tiff",cv2_img)
model.save("my_model")

#csv_image_generator("C:\\Users\\DavidS10\\PycharmProjects\\pythonProject\\image_csv.csv",bs=32)
# image = np.array(image_generator_train(bs=32))
# cv2.imwrite('blur_img.jpg', image[0][0])
# cv2.imshow("image", image[0][0])
# cv2.waitKey()
# cv2.imwrite('non_blur_img.jpg', image[1][0])
# cv2.imshow("image", image[1][0])
# cv2.waitKey()


