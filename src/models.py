import tensorflow as tf 
from .custom_cells import CustomRNNCell, CustomLSTMCell

layers=tf.keras.layers
models=tf.keras.models

def build_rnn_model(config, input_shape):
    units=config["MODELS"]["rnn_units"]
    dropout=config["MODELS"]["dropout"]
    dense_units=config["MODELS"]["dense_units"]
    
    cell1=CustomRNNCell(units=units[0])
    cell2=CustomRNNCell(units=units[1])
    
    rnn_layer1=tf.keras.layers.RNN(cell1,return_sequences=True)
    rnn_layer2=tf.keras.layers.RNN(cell2,return_sequences=False)
    
    inp=layers.Input(shape=input_shape)
    x=rnn_layer1(inp)
    x=layers.Dropout(dropout)(x)
    x=rnn_layer2(x)
    x=layers.Dropout(dropout)(x)
    x=layers.Dense(dense_units,activation='relu')(x)
    out=layers.Dense(1,activation='relu')(x)
    
    model=models.Model(inputs=inp,outputs=out)
    model.compile(optimizer=config["TRAINING"]["optimizer"],loss=config["TRAINING"]["loss"],metrics=["mae"])
    return model

def build_lstm_model(config,input_shape):
    units = config["MODELS"]["lstm_units"]
    dropout = config["MODELS"]["dropout"]
    dense_units = config["MODELS"]["dense_units"]
        
    cell1=CustomLSTMCell(units[0])
    cell2=CustomLSTMCell(units[1])
    
    rnn_layer1=tf.keras.layers.RNN(cell1, return_sequences=True)
    rnn_layer2=tf.keras.layers.RNN(cell2,return_sequences=False)
    
    inp=layers.Input(shape=input_shape)
    x=rnn_layer1(inp)
    x=layers.Dropout(dropout)(x)
    x=rnn_layer2(x)
    x=layers.Dropout(dropout)(x)
    x=layers.Dense(dense_units,activation='relu')(x)
    out=layers.Dense(1,activation='relu')(x)
    
    model=models.Model(inputs=inp,outputs=out)
    model.compile(
        optimizer=config["TRAINING"]["optimizer"],
        loss=config["TRAINING"]["loss"],
        metrics=["mae"]
    )
    
    return model