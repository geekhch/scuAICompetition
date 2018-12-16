+ full MAXLEN=35  lsrm_drop=0.1  w2c  b_lstm+droupout(0.5) 0.705
+ 15000 0.65 mask=false, maxlen=35
```
c1 = Conv1D(64,5,activation='relu')(em)
    c1 = MaxPooling1D(strides=1)(c1)
    x1 = Bidirectional(LSTM(50, recurrent_dropout=0.1))(c1)
    # x1 = Dense(128)(x1)
    
    x = Dropout(0.5)(x1)
```