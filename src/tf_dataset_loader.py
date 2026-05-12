import tensorflow as tf
import yaml

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

def create_tf_dataset(X,y,batch_size=32,shuffle=True):
    dataset=tf.data.Dataset.from_tensor_slices((X,y))
    if shuffle:
        dataset=dataset.shuffle(buffer_size=len(X))
    dataset=dataset.batch(batch_size=batch_size)
    dataset=dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_train_valid_test_datasets(X_train, y_train, X_valid, y_valid, X_test, y_test, config):
    batch_size=config["TRAINING"]["batch_size"]
    train_ds = create_tf_dataset(X_train, y_train, batch_size, shuffle=True)
    valid_ds = create_tf_dataset(X_valid, y_valid, batch_size, shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size, shuffle=False)
    return train_ds,valid_ds,test_ds