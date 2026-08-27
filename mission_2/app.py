"""
Documentation for the model results.

so basically this program loads the model and then uses it, after creating the ui elements with streamlit and taking the raw values
I had to basically map the values into numerical values and save it inside a dataframe, since the model only can understand numerical values

here's the passengers I used to predict

Passenger Class: 1
Sex: female
Age: 33
Siblings/Spouses aboard: 1
Parents/Children aboard: 1
Fare: 32.00
Embarked: C
Had a cabin listed?: ✓ (checked)
Title: Mr

Result: Did not survive (confidence: 60.6%)

___

Passenger Class: 1
Sex: male
Age: 33
Siblings/Spouses aboard: 1
Parents/Children aboard: 1
Fare: 100
Embarked: S
Had a cabin listed?: ✓ (checked)
Title: Rare

Result: did survive 

---

Passenger Class: 1
Sex: female
Age: 20
Siblings/Spouses aboard: 1
Parents/Children aboard: 1
Fare: 32.00
Embarked: Q
Had a cabin listed?: ✓ (checked)
Title: Mr


Result: Unknown — model isn't confident either way (50.2% survival probability)


for the unknown part, I basically said that if the confidence score is between 40-60 percent 

in terms of the explication, 

1. I accessed the model, using joblib which I saved the trained model on
2. the entrees are the follow, the build_features function basically transforms the raw text values into numerical values to a dataframe
and the proba[0] is the sortie which means not survival and the proba[1].

3. the components neccesary is the following:
    - loading the model
    - created the ui and with their appropriate type of input
    - transforming the raw data values into a dataframe
    - using the trained model to predict a surivival rate.
    
    
4. final test

I ran the final test against my model

Mystery Passengers — Predictions

Pclass | Sex    | Age | SibSp | Parch | Fare  | Embarked | Guessed Title | Prediction      | Survival Probability
1      | female | 29  | 0     | 0     | 85.00 | C        | Miss          | Survived        | 81.2%
3      | male   | 31  | 0     | 0     | 7.90  | S        | Mr            | Did not survive | 18.7%
3      | male   | 7   | 1     | 1     | 20.50 | S        | Master        | Survived        | 75.4%
2      | female | 42  | 1     | 0     | 28.00 | Q        | Miss          | Did not survive | 39.6%
1      | male   | 58  | 0     | 0     | 72.00 | C        | Mr            | Unknown         | 42.7%


"""


import joblib
import streamlit as st
import pandas as pd

# this basically loads the trained model inside of the file
clf = joblib.load('logic_regression_mode.joblib')

# this is ui stuff so the title and the difference options based on the data provided
st.title("Titanic Survival Predictor")

# each option
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
cabin = st.checkbox("Had a cabin listed?")
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

# what this does it takes the raw input that was used in the form, to basically turn into numerical values that the model was trained on.
def build_features(pclass, sex, age, sibsp, parch, fare, embarked, has_cabin, title):
    sex_encoded = 1 if sex == "male" else 0
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    row = {
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "HasCabin": 1 if has_cabin else 0,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Embarked_Q": embarked == "Q",
        "Embarked_S": embarked == "S",
        "Title_Miss": title == "Miss",
        "Title_Mr": title == "Mr",
        "Title_Mrs": title == "Mrs",
        "Title_Rare": title == "Rare",
    }
    # then returns it as a dataframe
    return pd.DataFrame([row])

def predict_with_label(input_df, margin=0.10):
    """Runs a prediction and returns (label, survive_proba)."""
    proba = clf.predict_proba(input_df)[0]
    survive_proba = proba[1]

    if abs(survive_proba - 0.5) < margin:
        return "Unknown", survive_proba
    elif survive_proba > 0.5:
        return "Survived", survive_proba
    else:
        return "Did not survive", survive_proba



if st.button("Predict"):
    input_df = build_features(pclass, sex, age, sibsp, parch, fare, embarked, cabin, title)
    label, survive_proba = predict_with_label(input_df)

    if label == "Unknown":
        st.warning(f"Unknown — model isn't confident either way ({survive_proba:.1%} survival probability)")
    elif label == "Survived":
        st.success(f"Survived! (confidence: {survive_proba:.1%})")
    else:
        st.error(f"Did not survive (confidence: {1 - survive_proba:.1%})")



mystery_passengers = [
    {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 85.0, "Embarked": "C"},
    {"Pclass": 3, "Sex": "male",   "Age": 31, "SibSp": 0, "Parch": 0, "Fare": 7.9,  "Embarked": "S"},
    {"Pclass": 3, "Sex": "male",   "Age": 7,  "SibSp": 1, "Parch": 1, "Fare": 20.5, "Embarked": "S"},
    {"Pclass": 2, "Sex": "female", "Age": 42, "SibSp": 1, "Parch": 0, "Fare": 28.0, "Embarked": "Q"},
    {"Pclass": 1, "Sex": "male",   "Age": 58, "SibSp": 0, "Parch": 0, "Fare": 72.0, "Embarked": "C"},
]


def infer_title(sex, age):
    if sex == "male" and age < 15:
        return "Master"
    elif sex == "male":
        return "Mr"
    else:
        return "Miss"


st.header("Mystery Passengers")

if st.button("Predict all mystery passengers"):
    results = []
    for p in mystery_passengers:
        guessed_title = infer_title(p["Sex"], p["Age"])
        input_df = build_features(
            p["Pclass"], p["Sex"], p["Age"], p["SibSp"], p["Parch"],
            p["Fare"], p["Embarked"], has_cabin=False, title=guessed_title
        )
        label, survive_proba = predict_with_label(input_df)

        results.append({
            **p,
            "Guessed Title": guessed_title,
            "Prediction": label,
            "Survival Probability": f"{survive_proba:.1%}",
        })

    st.table(pd.DataFrame(results))