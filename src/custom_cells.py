import tensorflow as tf

K=tf.keras.backend
sigmoid=tf.keras.activations.sigmoid
tanh=tf.keras.activations.tanh

class CustomRNNCell(tf.keras.layers.Layer):
    def __init__(self,units,**kwargs):
        super().__init__(**kwargs)
        self.units=units
        self.state_size=units
        self.output_size=units
        
    def build(self,input_shape):
        input_dim=input_shape[-1]
        self.kernel=self.add_weight(shape=(input_dim,self.units),initializer='glorot_uniform',name='kernel')
        self.recurrent_kernel=self.add_weight(shape=(self.units,self.units),initializer='orthogonal',name='recurrent_kernel')
        self.bias=self.add_weight(shape=(self.units,),initializer='zeros',name='bias')
        super().build(input_shape)
        
    def call(self,inputs,states):
        h_tm1=states[0]
        h=K.dot(inputs,self.kernel)+K.dot(h_tm1,self.recurrent_kernel)+self.bias
        h=K.tanh(h)
        return h,[h]
    
class CustomLSTMCell(tf.keras.layers.Layer):
    def __init__(self,units,**kwargs):
        super().__init__(**kwargs)
        self.units=units
        self.state_size=[units,units]
        self.output_size=units
        
    def build(self,input_shape):
        input_dim=input_shape[-1]
        self.kernel=self.add_weight(shape=(input_dim,4*self.units),initializer='glorot_uniform',name='kernel')
        self.recurrent_kernel=self.add_weight(shape=(self.units,4*self.units),initializer='orthogonal',name='recurrent_kernel')
        self.bias=self.add_weight(shape=(4*self.units,),initializer='zeros',name='bias')
        super().build(input_shape)
        
    def call(self,inputs,states):
        h_tmp1=states[0]
        c_tmp1=states[1]
        z=K.dot(inputs,self.kernel)+K.dot(h_tmp1,self.recurrent_kernel)+self.bias
        z0,z1,z2,z3=tf.split(z,num_or_size_splits=4,axis=1)
        
        i=sigmoid(z0)
        f=sigmoid(z1)
        c_hat=tanh(z2)
        o=sigmoid(z3)
        
        c=f*c_tmp1+i*c_hat
        h=o*tanh(c)
        
        return h,[h,c]