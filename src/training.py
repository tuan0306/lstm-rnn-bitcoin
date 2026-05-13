import time
import tensorflow as tf

def train_model(model,train_ds,val_ds,config,model_path):
    epochs=config["TRAINING"]["epochs"]
    early_stopping_patience=config["TRAINING"]["early_stopping_patience"]
    
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    start_time=time.time()
    history=model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=callbacks,
        epochs=epochs,
        verbose=1
    )
    
    training_time=time.time()-start_time
    return history,training_time