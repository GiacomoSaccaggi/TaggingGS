# -*- coding: utf-8 -*-
"""
SUPERVISED IMG
"""


def easy_net(to_tag, tagged, analysis_token):
    """
    \033[95m
    This function uses a pre-trained convolutional neural network for image classification,
    from which we remove the last two layers. Then thanks to a PCA we reduce the dimensionality of the vector
    and we calculate the clusters optimizing the number for MSE.
    ...
    Questa funzione utilizza una rete neurale convoluionale preaddestrata per la classificazione di immagini,
    dalla quale togliamo gli ultimi due layers. Poi grazie ad una PCA riduciamo la dimensionalità del vettore
    e calcoliamo i cluster ottimizzando la numerosità per MSE.
    \033[0m

    ################################################################################################################

    Reimportare il modello salvato:


    from keras.models import model_from_json

    # Importo l'architettura del modello
    json_file = open('model_trained.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("CNNW.h5")
    print("Modello importato!!!")
    """
    from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.layers import Activation, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras import losses
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from PIL import Image
    import numpy as np
    import base64
    import shutil
    import json
    import os
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path) + '\\Uploads'
    folder_from = f'{dir_path}/{analysis_token}/img_supervised'
    folder_to = f'{dir_path}/{analysis_token}/supervised'
    with open(f'{dir_path}/{analysis_token}/metadata.json', 'r', encoding='utf8') as f:
        metadata = json.load(f)
    try:
        os.mkdir(folder_from)
    except FileExistsError:
        pass
    immagini = []
    if metadata['tagging_type'] == 'img':
        for k, v in to_tag.items():
            imgdata = base64.b64decode(v.split(';base64,')[-1])
            filename = f'{folder_from}/{k.replace("upload_tmp/", "")}'
            with open(filename, 'wb') as f:
                f.write(imgdata)
            immagini.append(filename)

    base_model = VGG16(weights="imagenet")
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    # h, w = load_img(folder_from + '/' + immagini[0]).size
    h = 224
    w = 224

    def aumenta_tutte_uguali(image_pil, width, height):
        ratio_w = width / image_pil.width
        ratio_h = height / image_pil.height
        if ratio_w < ratio_h:
            resize_width = width
            resize_height = round(ratio_w * image_pil.height)
        else:
            resize_width = round(ratio_h * image_pil.width)
            resize_height = height
        image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
        background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
        background.paste(image_resize, offset)
        return background.convert('RGB')

    def extract_features(file_, model_, h_, w_):
        # load the image as a 224x224 array
        img_ = load_img(folder_from + '/' + file_)
        img_ = aumenta_tutte_uguali(img_, h_, w_)
        img_.save(folder_from + '/' + file_)
        # convert from 'PIL.Image.Image' to numpy array
        img_ = np.array(img_)
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = img_.reshape(1, h, w, 3)
        # prepare image for model
        imgx = preprocess_input(reshaped_img)
        # get the feature vector
        features = model_.predict(imgx, use_multiprocessing=True)
        return features

    data = {}

    # lop through each image in the dataset
    for imm in tqdm(immagini):
        # try to extract the features and update the dictionary
        try:
            feat = extract_features(imm, model, h, w)
            data[imm] = feat
        # if something fails, save the extracted features as a pickle file (optional)
        except Exception as e:
            print(f"Errore sull'immagine \n {imm}")
            print(e)
            pass

    os.mkdir(folder_to)
    os.mkdir(folder_to + "/Training")
    numero_classi = 0
    num_per_categ = {}
    for categ in set(tagged.values()):
        numero_classi += 1
        num_per_categ[categ] = 0
        os.mkdir(folder_to + f"/Training/{categ}")

    for file_i, categ in tagged.items():
        num_per_categ[categ] += 1
        shutil.copyfile(folder_from + '/' + file_i,
                        folder_to + "/Training/" + str(categ) + '/' + file_i)

    shutil.rmtree(folder_from)
    # model_new = MaxPooling2D()(base_model.layers[-2].output)
    model_new = Dense(64)(base_model.layers[-2].output)
    model_new = Activation('relu')(model_new)
    model_new = Dropout(0.2)(model_new)
    # Impostare funzione di attivazione e compilare modello
    if numero_classi == 2:
        model_new = Dense(numero_classi - 1)(model_new)
        model_new = Activation('sigmoid')(model_new)
        head_model = Model(inputs=base_model.inputs, outputs=model_new)
        head_model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
    else:

        model_new = Dense(numero_classi)(model_new)
        model_new = Activation('softmax')(model_new)
        head_model = Model(inputs=base_model.inputs, outputs=model_new)
        head_model.compile(loss=losses.categorical_crossentropy,
                           optimizer=Adam(),
                           metrics=['accuracy'])
    # FOR TRANSFER LEARNING
    for i in range(len(head_model.layers) - 6):
        head_model.layers[i].trainable = False

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3, rotation_range=90,
                                       width_shift_range=[-200, 200], brightness_range=[0.2, 1.0],
                                       height_shift_range=0.5,
                                       zoom_range=(0.5, 0.2), horizontal_flip=True, vertical_flip=True)

    if numero_classi == 2:
        train_generator = train_datagen.flow_from_directory(
            folder_to + "/Training",
            target_size=(h, w),
            batch_size=int(min(num_per_categ) / 5) + 1,
            class_mode='binary',
            subset='training')  # set as training data

    else:
        train_generator = train_datagen.flow_from_directory(
            folder_to + "/Training",
            target_size=(h, w),
            batch_size=int(min(num_per_categ) / 5) + 1,
            class_mode='categorical',
            subset='training')  # set as training data

    es = EarlyStopping(monitor='accuracy', patience=20, verbose=1, restore_best_weights=True)

    head_model.fit(
        train_generator,
        steps_per_epoch=10,
        epochs=100,
        callbacks=[es]
    )

    # Salvare l'architettura del modello
    model_json = head_model.to_json()
    with open(folder_to + "/model_trained.json", "w") as json_file:
        json_file.write(model_json)
    # Serializzare i pesi in formato HDF5
    head_model.save_weights(folder_to + "/CNNW.h5")
    with open(folder_to + "/model_information.txt", "w") as f:
        f.write(f'classes={numero_classi}\nepochs=50\nsteps_per_epoch=60\n'
                f'target_size=({h}, {w})\nbatch_size={int(min(num_per_categ) / 5 ) +1}\n')
        f.write(f'\n\nImageDataGenerator(rescale=1. / 255, validation_split=0.3, rotation_range=90,\n'
                '\twidth_shift_range=[-200, 200], brightness_range=[0.2, 1.0],\n'
                '\theight_shift_range=0.5,\n'
                '\tzoom_range=(0.5, 0.2), horizontal_flip=True, vertical_flip=True)\n')

    train_loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    train_acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']
    xc = range(len(model.history.history['val_accuracy']))

    # Training loss vs Validation loss
    fig_performance, ax = plt.subplots(nrows=1, ncols=2, figsize=(19, 11))
    ax[0].plot(xc, train_loss)
    ax[0].plot(xc, val_loss)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training loss vs Validation loss')
    ax[0].grid(True)
    ax[0].legend(['Training', 'Validation'], loc=0)
    # Training accuracy vs Validation accuracy
    ax[1].plot(xc, train_acc)
    ax[1].plot(xc, val_acc)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training accuracy vs Validation accuracy')
    ax[1].grid(True)
    ax[1].legend(['Training', 'Validation'], loc=0)

    return f"\tLe immagini che erano nella cartella\n\t{folder_from}\n\tsono state salvate in" \
           f"\n\t{folder_to}\n\tdivide in {num_per_categ} clusters."
