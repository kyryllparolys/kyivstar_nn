import pandas as pd
import tensorflow as tf
import csv


def get_dataset():
    dataset = pd.read_csv('./tabular_data.csv')
    labels = pd.read_csv('./train.csv')

    labels.pop('id')  # delete id column from X_test

    dataset = dataset.fillna(0)  # deleting all nan values
    dataset = dataset.drop_duplicates('id')  # deleting duplicate rows

    # delete id, period
    dataset.pop('id')
    dataset.pop('period')
    types = list()  # list of types of hashed strings, there're only 8 of them in feature_25
    for index in dataset.index:
        type = dataset['feature_25'][index]
        if type not in types:
            types.append(type)
        dataset['feature_41'][index] = 0.1 * types.index(dataset['feature_25'][index])
    dataset.pop('feature_25')

    dataset = tf.keras.utils.normalize(dataset)

    y_test = dataset.drop(dataset.index[:4085])
    dataset = dataset.drop(dataset.index[4084:])
    dataset = tf.data.Dataset.from_tensor_slices((dataset.values, labels.values))


    return dataset, y_test


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(52, activation='relu'),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dense(52, activation='relu'),
        # tf.keras.layers.Embedding(input_dim=48, output_dim=10),
        # tf.keras.layers.GRU(256, return_sequences=True),
        # tf.keras.layers.SimpleRNN(128),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',  # adam
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def save_predictions(y_test, model):
    predictions = model.predict(y_test)
    counter = 4084
    with open("final.csv", 'w', newline="") as csvfile:
        fieldnames = ['id', 'score']
        writer = csv.DictWriter(csvfile, fieldnames)

        for i in predictions:
            writer.writerow({'id': counter, 'score': i[0]})
            counter += 1


if __name__ == '__main__':
    dataset, y_test = get_dataset()

    model = get_compiled_model()

    train_dataset = dataset.shuffle(len(dataset)).batch(1)
    history = model.fit(train_dataset, epochs=10)

    model.save('bigdata_model_10epochs.h5')
    save_predictions(y_test, model)
