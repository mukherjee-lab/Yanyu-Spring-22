# %% [Algorithm 1c Loop]
# # MUSHROOMS

# %% [markdown]
# ## Binary Classification

# %% [markdown]
# ### Imports

# %%
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

# %% [markdown]
# ### Load Data
dataset = pd.read_csv(r"C:\Users\yxie367\Documents\GitHub\Mushrooms\DATA\mushrooms.csv")
#dataset = pd.read_csv(r"C:\Users\xieya\Documents\GitHub\Mushrooms\DATA\mushrooms.csv")

# %% [markdown]
# ### View Data and Informations

# %%
dataset.head()

# %%
dataset.info()

# %%
edible, poisonous = dataset['class'].value_counts()

# print("Edible:\t  ", edible,"\nPoisonous:", poisonous)

# %%
# Categorical to numerical
labels = {'e': 0, 'p': 1}
dataset['class'].replace(labels, inplace=True)

edible, poisonous = dataset['class'].value_counts()
#print("0 - Edible:   ", edible,"\n1 - Poisonous:", poisonous)

# %% [markdown]
# # NN1 Gill Color - Black (k)

# %% [markdown]
# ### Split Dataset

# %% [markdown]
# #### Get the Labels

# %%
X, y =  dataset.drop('class', axis=1), dataset['class'].copy()

#print("X:",X.shape,"\ny:",y.shape)

# %% [markdown]
# #### Train Set and Test Set
total_error_1 = 0
total_error_2 = 0
total_error_comb = 0
randnum = np.arange(2,44,4)
num_trials = len(randnum)
record = ""
wrong_record = ""
run = 1

# %% Data cleaning
from sklearn.model_selection import train_test_split
X_white = pd.DataFrame()
X_not_white = pd.DataFrame()
y_white = pd.Series(dtype='float64')
y_not_white = pd.Series(dtype='float64')
for i in range(0,len(X)):
    if X.loc[i,"stalk-root"] == "r":
        X_white = X_white.append(X.iloc[i,:])
        y_white = y_white.append(pd.Series(y.iloc[i]))
    else:
        X_not_white = X_not_white.append(X.iloc[i,:])
        y_not_white = y_not_white.append(pd.Series(y.iloc[i]))


# %% Data cleaning pt2
X_green = pd.DataFrame()
X_not_green = pd.DataFrame()
y_green = pd.Series(dtype='float64')
y_not_green = pd.Series(dtype='float64')
for i in range(0,len(X)):
    if X.loc[i,"odor"] == "a":
        X_green = X_green.append(X.iloc[i,:])
        y_green = y_green.append(pd.Series(y.iloc[i]))
    else:
        X_not_green = X_not_green.append(X.iloc[i,:])
        y_not_green = y_not_green.append(pd.Series(y.iloc[i]))

# %%

for j in randnum:
    X_train_not_white, X_test_not_white, y_train_not_white, y_test_not_white = train_test_split(X_not_white, y_not_white, test_size=1-(6905/(8124-len(X_white))), random_state=j)
    X_train_not_green, X_test_not_green, y_train_not_green, y_test_not_green = train_test_split(X_not_green, y_not_green, test_size=1-(6905/(8124-len(X_green))), random_state=j)


    X_train_green = (X_train_not_green)
    y_train_green = (y_train_not_green)
    X_train_white = (X_train_not_white)
    y_train_white = (y_train_not_white)
    # %%
    from sklearn.utils import shuffle
    X_train_full1 = shuffle(X_train_white, random_state=j)
    X_test = shuffle(X, random_state=j).iloc[4000:8000]
    y_train_full1 = shuffle(y_train_white, random_state=j)
    y_test = shuffle(y, random_state=j).iloc[4000:8000]

    # %% [markdown]
    # #### Validation Set

    # %%
    X_valid1, X_train1 = X_train_full1[:500], X_train_full1[500:]
    y_valid1, y_train1 = y_train_full1[:500], y_train_full1[500:]

    # print("X_train:", X_train1.shape[0], "y_train", y_train1.shape[0])
    # print("X_valid: ", X_valid1.shape[0], "y_valid ", y_valid1.shape[0])
    # print("X_test: ", X_test.shape[0], "y_test ", X_test.shape[0])


    # %% [markdown]
    # ### Prepare the Data

    # %% [markdown]
    # #### Data Transformation

    # %%
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer

    cat_attr_pipeline = Pipeline([
                            ('encoder', OrdinalEncoder())
                        ])

    cols = list(X)
    pipeline = ColumnTransformer([
                    ('cat_attr_pipeline', cat_attr_pipeline, cols)
                ])


    X_train1 = pipeline.fit_transform(X_train1)
    X_valid1 = pipeline.fit_transform(X_valid1)
    X_test1  = pipeline.fit_transform(X_test)

    # %% [markdown]
    # ### Neural Network

    # %% [markdown]
    # #### Model

    # %%
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer, Dense

    # %%
    # tf.random.set_seed(j)
    tf.random.set_random_seed(j)

    # %%
    model1 = Sequential([
        InputLayer(input_shape=(22,)),    # input  layer
        Dense(45, activation='relu'),     # hidden layer
        Dense(1,   activation='sigmoid')  # output layer
    ])

    # %%
    #model1.summary()

    # %% [markdown]
    # #### Compile the Model

    # %%
    model1.compile(loss='binary_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])

    # %% [markdown]
    # #### Prepare Callbacks

    # %%
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    checkpoint_cb = ModelCheckpoint('../SavedModels/best_model.h5',
                                    save_best_only=True)

    early_stopping_cb = EarlyStopping(patience=3,
                                    restore_best_weights=True)

    # %% [markdown]
    # ### Training

    # %%
    train_model1 = model1.fit(X_train1, y_train1,
                            epochs=100,
                            validation_data=(X_valid1, y_valid1),
                            callbacks=[checkpoint_cb, early_stopping_cb])



    # %% [markdown]
    # ### Evaluate the Best Model on Test Set

    # %%
    results1 = model1.evaluate(X_test1, y_test)
    # print("test loss, test acc:", results1)

    # %% [markdown]
    # ### Make Some Predictions

    # %%
    X_new1 = X_test1[:5]
    y_prob1 = model1.predict(X_new1)
    # print(y_prob.round(3))

    # %%
    y_pred1 = (model1.predict(X_new1) > 0.5).astype("int32")
    # print(y_pred)
    y_test_pred = (model1.predict(X_test1) > 0.5).astype("int32")

    # %% [markdown]
    # ## KL Divergence

    # %%
    # X_new = X_test[:5]
    X_df1 = pd.DataFrame(model1.predict(X_test1))
    y_test_pred1 = pd.DataFrame(y_test_pred).reset_index(drop=True)
    X_df1 = pd.concat([X_df1, y_test_pred1], axis=1)
    y_test1 = y_test.reset_index(drop=True)
    X_df1 = pd.concat([X_df1, y_test1], axis=1)
    X_df1.columns = ["X_pred","y_pred","y_actual"]
    #print(X_df1)

    # %%
    import math
    table1 = pd.DataFrame(columns=["KL_div","abs_distance","correctness"])
    for i in range(0,len(X_df1)):
        # KL divergence
        p = X_df1.loc[i,"X_pred"]
        try:
            kl = -(p*math.log(p) + (1-p)*math.log(1-p))
        except:
            kl = 0
        table1.loc[i,"KL_div"] = kl
        # absolute distance
        abs_dist = 2*abs(0.5-p)
        table1.loc[i,"abs_distance"] = abs_dist
        # correctness
        y_pred1 = X_df1.loc[i,"y_pred"]
        y_act1 = X_df1.loc[i,"y_actual"]
        if y_pred1 == y_act1:
            table1.loc[i,"correctness"] = 1 # correct prediction
        else:
            table1.loc[i,"correctness"] = 0 # wrong prediction
        table1.loc[i,"y_pred"] = y_pred1

    #print(table1)

    # %%
    table1["count"] = 1
    correctness1 = table1[["correctness","count"]].groupby(pd.cut(table1["KL_div"], np.arange(0, 0.8, 0.1))).apply(sum)
    correctness1["percent"] = 100*(correctness1["correctness"]/correctness1["count"])
    #print(correctness1)

    # %%
    index = []
    for i in (correctness1.index):
        index.append(str(i))
    plt.bar(index,correctness1["percent"], width=0.7)
    for index,data in enumerate(correctness1["percent"]):
        plt.text(x=index , y =data+1 , s=f"{round(data,2)}" , fontdict=dict(fontsize=15),ha='center')
    plt.ylim(0,120)
    plt.xlabel("KL Divergence")
    plt.ylabel("% correct")

    # %% [markdown]
    # ### Confidence

    # %%
    kl1 = table1[["correctness","count"]].groupby(pd.cut(table1["KL_div"], np.arange(0, 0.80, 0.1))).apply(sum)
    kl1["percent"] = (kl1["correctness"]/kl1["count"])
    kl1.dropna(inplace=True)
    plt.scatter(np.arange(0, 0.70, 0.1), kl1["percent"])
    plt.xlabel("KL Divergence")
    plt.ylabel("% correct")

    # %%
    # Linear Regression
    from sklearn.linear_model import LinearRegression

    x_reg1 = np.arange(0, 0.70, 0.1).reshape((-1, 1))
    y_reg1 = kl1["percent"]
    reg_model1 = LinearRegression().fit(x_reg1,y_reg1)

    # %%
    # print('intercept(alpha):', reg_model1.intercept_)
    # print('slope(theta):', reg_model1.coef_)

    # %% [markdown]
    # # NN2 Odor - Almond (a)

    # %% [markdown]
    # #### Train Set and Test Set

        # %%
    from sklearn.utils import shuffle
    X_train_full2 = shuffle(X_train_green, random_state=j)
    # X_test2 = shuffle(X_test_green, random_state=j)
    y_train_full2 = shuffle(y_train_green, random_state=j)
    # y_test2 = shuffle(y_test_green, random_state=j)

    # %% [markdown]
    # #### Validation Set

    # %%
    X_valid2, X_train2 = X_train_full2[:500], X_train_full2[500:]
    y_valid2, y_train2 = y_train_full2[:500], y_train_full2[500:]

    # print("X_train:", X_train2.shape[0], "y_train", y_train2.shape[0])
    # print("X_valid: ", X_valid2.shape[0], "y_valid ", y_valid2.shape[0])
    # print("X_test: ", X_test.shape[0], "y_test ", X_test.shape[0])


    # %% [markdown]
    # ### Prepare the Data

    # %% [markdown]
    # #### Data Transformation

    # %%
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer

    cat_attr_pipeline = Pipeline([
                            ('encoder', OrdinalEncoder())
                        ])

    cols = list(X)
    pipeline = ColumnTransformer([
                    ('cat_attr_pipeline', cat_attr_pipeline, cols)
                ])


    X_train2 = pipeline.fit_transform(X_train2)
    X_valid2 = pipeline.fit_transform(X_valid2)
    X_test2  = pipeline.fit_transform(X_test)
    y_test2 = y_test

    # %% [markdown]
    # ### Neural Network

    # %% [markdown]
    # #### Model

    # %%
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer, Dense

    tf.random.set_random_seed(j)

    # %%
    model2 = Sequential([
        InputLayer(input_shape=(22,)),    # input  layer
        Dense(45, activation='relu'),     # hidden layer
        Dense(1,   activation='sigmoid')  # output layer
    ])

    # %%
    #model2.summary()

    # %% [markdown]
    # #### Compile the Model

    # %%
    model2.compile(loss='binary_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])

    # %% [markdown]
    # #### Prepare Callbacks

    # %%
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    checkpoint_cb = ModelCheckpoint('../SavedModels/best_model.h5',
                                    save_best_only=True)

    early_stopping_cb = EarlyStopping(patience=3,
                                    restore_best_weights=True)

    # %% [markdown]
    # ### Training

    # %%
    train_model2 = model2.fit(X_train2, y_train2,
                            epochs=100,
                            validation_data=(X_valid2, y_valid2),
                            callbacks=[checkpoint_cb, early_stopping_cb])



    # %% [markdown]
    # ### Evaluate the Best Model on Test Set

    # %%
    results2 = model2.evaluate(X_test2, y_test2)
    # print("test loss, test acc:", results2)

    # %% [markdown]
    # ### Make Some Predictions


    # %%
    # y_pred2 = (model2.predict(X_new2) > 0.5).astype("int32")
    # print(y_pred2)
    y_test_pred2 = (model2.predict(X_test2) > 0.5).astype("int32")

    # %% [markdown]
    # ## KL Divergence

    # %%
    # X_new = X_test[:5]
    X_df2 = pd.DataFrame(model2.predict(X_test2))
    y_test_pred2 = pd.DataFrame(y_test_pred2).reset_index(drop=True)
    X_df2 = pd.concat([X_df2, y_test_pred2], axis=1)
    y_test2 = y_test2.reset_index(drop=True)
    X_df2 = pd.concat([X_df2, y_test2], axis=1)
    X_df2.columns = ["X_pred","y_pred","y_actual"]
    #print(X_df2)

    # %%
    import math
    table2 = pd.DataFrame(columns=["KL_div","abs_distance","y_pred","correctness"])
    for i in range(0,len(X_df2)):
        # KL divergence
        p = X_df2.loc[i,"X_pred"]
        if p > 0:
            kl = -(p*math.log(p) + (1-p)*math.log(1-p))
        else:
            kl = 1
        table2.loc[i,"KL_div"] = kl
        # absolute distance
        abs_dist = 2*abs(0.5-p)
        table2.loc[i,"abs_distance"] = abs_dist
        # correctness
        y_pred = X_df2.loc[i,"y_pred"]
        y_act = X_df2.loc[i,"y_actual"]
        if y_pred == y_act:
            table2.loc[i,"correctness"] = 1 # correct prediction
        else:
            table2.loc[i,"correctness"] = 0 # wrong prediction
        table2.loc[i,"y_pred"] = y_pred

    #print(table2)

    # %%
    table2["count"] = 1
    correctness2 = table2[["correctness","count"]].groupby(pd.cut(table2["KL_div"], np.arange(0, 0.8, 0.1))).apply(sum)
    correctness2["percent"] = 100*(correctness2["correctness"]/correctness2["count"])
    #print(correctness2)

    # %%
    index = []
    for i in (correctness2.index):
        index.append(str(i))
    plt.bar(index,correctness2["percent"], width=0.7)
    for index,data in enumerate(correctness2["percent"]):
        plt.text(x=index , y =data+1 , s=f"{round(data,2)}" , fontdict=dict(fontsize=15),ha='center')
    plt.ylim(0,120)
    plt.xlabel("KL Divergence")
    plt.ylabel("% correct")

    # %% [markdown]
    # ### Confidence

    # %%
    kl2 = table2[["correctness","count"]].groupby(pd.cut(table2["KL_div"], np.arange(0, 0.8, 0.1))).apply(sum)
    kl2["percent"] = (kl2["correctness"]/kl2["count"])
    kl2.dropna(inplace=True)
    plt.scatter(np.arange(0, 0.70, 0.1), kl2["percent"])
    # print(kl)
    # print(np.arange(0, 0.7, 0.05))

    # %%
    # Linear Regression
    from sklearn.linear_model import LinearRegression

    x_reg2 = np.arange(0, 0.7, 0.1).reshape((-1, 1))
    y_reg2 = kl2["percent"]
    reg_model2 = LinearRegression().fit(x_reg2,y_reg2)

    # %%
    # print('intercept(alpha):', reg_model2.intercept_)
    # print('slope(theta):', reg_model2.coef_)

    # %% [markdown]
    # ## Algorithm C: It = argmax(Ct,i)

    # %%
    # Correct answer
    ans = pd.DataFrame(X_df2["y_actual"])

    # NN1
    alpha1 = reg_model1.intercept_
    theta1 = reg_model1.coef_

    # NN2
    alpha2 = reg_model2.intercept_
    theta2 = reg_model2.coef_

    # %%
    # Creating NN tables
    nn1 = table1.drop(["abs_distance","correctness"], axis=1)
    nn1["conf"] = alpha1 + theta1 * nn1["KL_div"]

    nn2 = table2.drop(["abs_distance","correctness"], axis=1)
    nn2["conf"] = alpha2 + theta2 * nn2["KL_div"]

    # nn2

    # %%
    # Determing higher confidence NN and choosing that arm

    for i in range(0,len(nn1)):
        if nn1.loc[i,"conf"] > nn2.loc[i,"conf"]:
            ans.loc[i,"y_pred"] = nn1.loc[i,"y_pred"]
            ans.loc[i,"NN"] = 1
            ans.loc[i,"conf"] = nn1.loc[i,"conf"]
        else:
            ans.loc[i,"y_pred"] = nn2.loc[i,"y_pred"]
            ans.loc[i,"NN"] = 2
            ans.loc[i,"conf"] = nn2.loc[i,"conf"]


    # ans

    # %% [markdown]
    # #### Comparing performance

    # %%
    # NN1 performance
    cost1 = 0
    for i in range(0,len(nn1)):
        if nn1.loc[i,"y_pred"] != ans.loc[i,"y_actual"]:
            cost1 += 1
        else:
            pass

    # NN2 performance
    cost2 = 0
    for i in range(0,len(nn2)):
        if nn2.loc[i,"y_pred"] != ans.loc[i,"y_actual"]:
            cost2 += 1
        else:
            pass

    # Combined performance
    cost3 = 0
    for i in range(0,len(nn1)):
        nn = ans.loc[i,"NN"]
        nn_conf = ans.loc[i,"conf"]
        if ans.loc[i,"y_pred"] != ans.loc[i,"y_actual"]:
            cost3 += 1
            wrong_record = wrong_record + (f"Run:{run} - Wrong NN:{nn}, Conf:{nn_conf}") + "\n"
        else:
            pass

    # %%
    record = record+(f"Run:{run} - Error count for NN1:{cost1}, NN2:{cost2}, Combined:{cost3}") + "\n"

    total_error_1 += cost1
    total_error_2 += cost2
    total_error_comb += cost3

    print(f"Run {run} complete!")
    run+=1

print(record)
print(f"Average error count for NN1:{total_error_1/num_trials}, NN2:{total_error_2/num_trials}, Combined:{total_error_comb/num_trials}")

#%%
print(wrong_record)
# %%
