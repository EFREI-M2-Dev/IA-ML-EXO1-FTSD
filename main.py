import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error
import random

# on charge les données
df = pd.read_csv("./datasets/housedata.csv")

# ici on supprime les colonnes qui ne sont pas utiles pour le podel
df.drop(['street', 'statezip', 'country'], axis=1, inplace=True)

# on convertit la colonne 'date' en datetime panda
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df.drop('date', axis=1, inplace=True)

# on convertit la collonne city en entier
le = LabelEncoder()
df['city'] = le.fit_transform(df['city'].astype(str))

# on supprime les valeurs manquantes et les doublons
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# on supprime les valeurs incohérentes
df = df[df['bedrooms'] > 0]
df = df[df['price'] < 3_000_000] 

# on separe les donnée d'entraiement et de prédiction
X = df.drop('price', axis=1)
y = df['price']

# on normalise les données c'est a dire que on les met a la meme echelle
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# on divise les données en données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# on crée le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# on évalue le modèle
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE : {round(rmse, 2)} €") # a ce que j'ai compris c'est l'erreur moyenne de la prédiction
print(f"✅ R²   : {round(r2, 4)}") # et la c'est le coefficient de détermination

# on fait une validation croisée pour avoir une idée de la performance du modèle
# % ± % c'est la moyenne R² et l'ecart type
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f"✅ R² moyen (CV 5 folds) : {round(scores.mean(), 4)} ± {round(scores.std(), 4)}")

# on teste le modèle sur une maison réelle
random_index = random.randint(0, len(X_scaled) - 1)

maison = X_scaled.iloc[random_index]
maison_input = pd.DataFrame([maison], columns=X.columns)
predicted = model.predict(maison_input)[0]
real = y.iloc[random_index]

print("\n> Test sur une maison réelle :")
print(f"Caractéristiques de la maison (ligne {random_index}) :")
print(maison)

print("\n> Résultat de la prédiction :")
print(f"- Prix prédit : {int(predicted):,} €")
print(f"- Prix réel   : {int(real):,} €")
