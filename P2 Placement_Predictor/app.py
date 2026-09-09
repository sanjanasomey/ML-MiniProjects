from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Create history file if it doesn't exist
if not os.path.exists("prediction_history.csv"):
    df = pd.DataFrame(
        columns=[
            "CGPA",
            "IQ",
            "Profile Score",
            "Prediction"
        ]
    )
    df.to_csv("prediction_history.csv", index=False)


@app.route("/")
def home():

    history = pd.read_csv("prediction_history.csv")

    recent_history = history.tail(5).values.tolist()

    return render_template(
        "index.html",
        history=recent_history
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:

        cgpa = float(request.form["cgpa"])
        iq = int(request.form["iq"])
        profile_score = int(request.form["profile_score"])

        # Validation

        if cgpa < 0 or cgpa > 10:
            raise ValueError(
                "CGPA must be between 0 and 10"
            )

        if iq <= 0:
            raise ValueError(
                "IQ must be greater than 0"
            )

        if profile_score < 0 or profile_score > 100:
            raise ValueError(
                "Profile Score must be between 0 and 100"
            )

        input_data = np.array(
            [cgpa, iq, profile_score]
        ).reshape(1, 3)

        prediction = model.predict(input_data)

        result = (
            "Placed"
            if prediction[0] == 0
            else "Not Placed"
        )

        # Save history

        new_record = pd.DataFrame(
            [[
                cgpa,
                iq,
                profile_score,
                result
            ]],
            columns=[
                "CGPA",
                "IQ",
                "Profile Score",
                "Prediction"
            ]
        )

        new_record.to_csv(
            "prediction_history.csv",
            mode="a",
            header=False,
            index=False
        )

        history = pd.read_csv(
            "prediction_history.csv"
        )

        recent_history = history.tail(5).values.tolist()

        return render_template(
            "index.html",
            result=result,
            history=recent_history
        )

    except Exception as e:

        history = pd.read_csv(
            "prediction_history.csv"
        )

        recent_history = history.tail(5).values.tolist()

        return render_template(
            "index.html",
            error=str(e),
            history=recent_history
        )


if __name__ == "__main__":
    app.run(debug=True)