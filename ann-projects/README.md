# 🤖 ANN Deep Learning Projects

A collection of deep learning projects built using **Artificial Neural Networks (ANN)** 
covering real-world classification and regression problems — fully documented and 
beginner friendly.

---

## 📚 What is an ANN?

An **Artificial Neural Network (ANN)** is a machine learning model inspired by the 
human brain. Just like our brain has neurons connected to each other, an ANN has 
**nodes (artificial neurons)** connected in layers.


### 🧠 How does it learn?
1. **Forward Pass** — Data flows through the network and makes a prediction
2. **Loss Calculation** — It checks how wrong the prediction was
3. **Backpropagation** — It goes backward and adjusts the weights to reduce error
4. **Repeat** — This process repeats for every epoch until the model gets better

### 🔑 Key Terms for Beginners

| Term | Simple Meaning |
|---|---|
| **Neuron** | A single unit that takes input, does math, gives output |
| **Layer** | A group of neurons working together |
| **Weight** | A number that decides how important an input is |
| **Bias** | Helps the model shift the output up or down |
| **Activation Function** | Decides whether a neuron should "fire" or not |
| **Epoch** | One full pass through the entire training data |
| **Loss** | How wrong the model's prediction is |
| **Optimizer** | The method used to reduce loss (we use Adam) |
| **Accuracy** | How often the model gets the right answer |

---

## 📂 Projects

| # | Project | Type | Algorithm | Accuracy |
|---|---|---|---|---|
| 1 | [🚢 Titanic Survival Prediction](1-titanic-survival) | Classification | ANN + Sigmoid | ~81% |
| 2 | [🏠 House Price Prediction](2-house-price) | Regression | ANN + Linear | MAE ~$22K |

---

## 🗂️ Repository Structure

```text
ann-projects/
+-- 1-titanic-survival/
|   +-- data/
|   |   +-- train.csv
|   |   +-- test.csv
|   +-- notebooks/
|   |   +-- Ann_titanic_project.ipynb
|   +-- screenshots/
|   |   +-- screenshot (1).png
|   |   +-- screenshot (2).png
|   |   +-- screenshot (3).png
|   +-- README.md
+-- 2-house-price/
|   +-- data/
|   |   +-- train.csv
|   |   +-- test.csv
|   +-- notebooks/
|   |   +-- Ann_house_Project.ipynb
|   +-- screenshots/
|   |   +-- screenshot (1).png
|   |   +-- screenshot (2).png
|   |   +-- screenshot (3).png
|   +-- README.md
+-- extras/
|   +-- Backpropagation_GradientDescent.ipynb
|   +-- Optimizers_DeepLearning.ipynb
+-- README.md
```

---

## 🚢 Project 1 — Titanic Survival Prediction

### 🎯 Goal
Predict whether a passenger **survived or not** based on features like age, gender, 
ticket class, and fare.

### 📊 Dataset
- **Source:** Kaggle Titanic Dataset
- **Samples:** 891 passengers
- **Features:** 8 (after cleaning)
- **Target:** Survived (1 = Yes, 0 = No)

### 🔄 Step by Step Process

**Step 1 — Load Data**
```python
titanic_df = pd.read_csv('data/train.csv')
```
We load the CSV file into a Pandas DataFrame to explore and manipulate the data.

**Step 2 — Check Missing Values**
```python
titanic_df.isnull().sum()
```
Real-world data is messy! We check which columns have missing values so we can fix them.

**Step 3 — Clean Data**
```python
# Drop columns we don't need
titanic_df = titanic_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing Age with average age
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())

# Fill missing Embarked with most common value
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])
```
We remove useless columns and fill missing values so the model can work properly.

**Step 4 — Encode Categorical Data**
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
titanic_df['Sex'] = le.fit_transform(titanic_df['Sex'])        # male=1, female=0
titanic_df['Embarked'] = le.fit_transform(titanic_df['Embarked'])
```
ANNs only understand **numbers**, not text. So we convert "male/female" → "1/0".

**Step 5 — Split Features & Target**
```python
X = titanic_df.drop('Survived', axis=1)   # Input features
y = titanic_df['Survived']                 # Target (what we want to predict)
```

**Step 6 — Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
We use 80% of data to **train** the model and 20% to **test** it on unseen data.

**Step 7 — Feature Scaling**
```python
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
```
ANNs work best when all numbers are in the **same range**. Scaling brings all features 
between -1 and 1.

> ⚠️ Important: We use `fit_transform` on training data but only `transform` on test 
data — because we don't want the model to "see" test data during training!

**Step 8 — Build ANN**
```python
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))  # Hidden layer 1
model.add(Dense(8, activation='relu'))                                 # Hidden layer 2
model.add(Dense(1, activation='sigmoid'))                              # Output layer
```

Our ANN architecture:
> 💡 **ReLU** — Used in hidden layers. Keeps positive values, turns negatives to 0.
> 💡 **Sigmoid** — Used in output for binary classification. Gives a value between 0-1 
(probability of surviving).

**Step 9 — Compile Model**
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- **Adam** — Smart optimizer that adjusts learning rate automatically
- **Binary Crossentropy** — Best loss function for yes/no (binary) problems

**Step 10 — Train Model**
```python
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```
We train for **50 epochs** — meaning the model sees the full data 50 times and keeps 
improving.

**Step 11 — Evaluate & Results**
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

### 📊 Screenshots

#### Model Summary
![Model Summary](1-titanic-survival/screenshots/Screenshot%20(1).png)

#### Training Curves
![Training Curves](1-titanic-survival/screenshots/Screenshot%20(2).png)

#### Final Accuracy
![Final Accuracy](1-titanic-survival/screenshots/Screenshot%20(3).png)

---

## 🏠 Project 2 — House Price Prediction

### 🎯 Goal
Predict the **sale price of a house** based on features like size, location, quality, 
and number of rooms.

### 📊 Dataset
- **Source:** Kaggle House Prices Dataset
- **Samples:** 1460 houses
- **Features:** 80+ (after encoding)
- **Target:** SalePrice (in USD)

### 🔄 Step by Step Process

**Step 1 — Load Data**
```python
house_df = pd.read_csv('data/train.csv')
```

**Step 2 — Check Missing Values**
```python
house_df.isnull().sum()
```
This dataset has many missing values across multiple columns — we need to handle all of them.

**Step 3 — Clean Data**
```python
# Drop columns with too many missing values
house_df = house_df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# Fill numeric columns with mean
numeric_cols = house_df.select_dtypes(include=['int64', 'float64']).columns
house_df[numeric_cols] = house_df[numeric_cols].fillna(house_df[numeric_cols].mean())

# Fill categorical columns with mode
cat_cols = house_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    house_df[col] = house_df[col].fillna(house_df[col].mode()[0])
```

**Step 4 — One-Hot Encoding**
```python
house_df = pd.get_dummies(house_df, drop_first=True)
```
This dataset has many text columns (like "Neighborhood", "RoofStyle"). We convert them 
all to numbers using **One-Hot Encoding** — this creates a new column for each category 
with 0 or 1.

> 💡 **Why One-Hot instead of Label Encoding?**
> Because categories like neighborhoods have no order — "Sawyer" is not greater than 
"NAmes". One-Hot treats them equally.

**Step 5 — Split Features & Target**
```python
X = house_df.drop('SalePrice', axis=1)
y = house_df['SalePrice']
```

**Step 6 — Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Step 7 — Feature Scaling**
```python
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
```

**Step 8 — Build ANN**
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))  # Hidden layer 1
model.add(Dense(32, activation='relu'))                                # Hidden layer 2
model.add(Dense(1, activation='linear'))                               # Output layer
```

Our ANN architecture:
> 💡 **Linear activation** — Used in output for regression. Unlike Sigmoid (0-1), 
Linear can output any number — perfect for predicting a price!

**Step 9 — Compile Model**
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```
- **MSE (Mean Squared Error)** — Loss function for regression. Penalizes big errors more
- **MAE (Mean Absolute Error)** — How far off predictions are on average in dollars

**Step 10 — Train Model**
```python
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
```
We use **100 epochs** here (more than Titanic) because house prices are more complex 
to learn.

**Step 11 — Evaluate & Results**
```python
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: ${mae:,.2f}")
```
MAE tells us — on average, our prediction is off by this many dollars.

### 📊 Screenshots

#### Model Summary
![Model Summary](2-house-price/screenshots/Screenshot%20(1).png)

#### Training Curves
![Training Curves](2-house-price/screenshots/Screenshot%20(2).png)

#### Final MAE
![Final MAE](2-house-price/screenshots/Screenshot%20(3).png)

---

## 🔑 Key Differences Between Both Projects

| | 🚢 Titanic | 🏠 House Price |
|---|---|---|
| **Problem Type** | Classification | Regression |
| **Output** | 0 or 1 (Survived?) | A number ($Price) |
| **Output Activation** | Sigmoid | Linear |
| **Loss Function** | Binary Crossentropy | MSE |
| **Metric** | Accuracy % | MAE in $ |
| **Encoding** | Label Encoding | One-Hot Encoding |
| **Epochs** | 50 | 100 |

---

## 📚 Extras — Theory Notebooks

| Notebook | What You Learn |
|---|---|
| `Backpropagation_GradientDescent.ipynb` | How ANN learns from scratch using math |
| `Optimizers_DeepLearning.ipynb` | Difference between SGD and Adam optimizer |

These notebooks explain the **"why"** behind everything above — great for beginners!

---

## 🛠️ Tech Stack

- Python, NumPy, Pandas
- TensorFlow, Keras
- Scikit-learn
- Matplotlib
- Jupyter Notebook / Google Colab

## 👤 Author
[Harshit Sharma](https://github.com/harshitsharma200377-spec)