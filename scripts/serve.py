import os
import numpy as np
import subprocess
from flask import Flask, request
from flask_json import FlaskJSON, JsonError, as_json
from werkzeug.utils import secure_filename
from single_labeling_api import TWilBertLabelClass

# Load the model to be used in inference
processor = TWilBertLabelClass(
    "configs/microservs/config_labelling_single_hateeval19_large.json"
)

# FIXME: the system crashes if it does not perform an analysis outside the post method
processor.predict(["Texto de prueba"])

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
APP_ROOT = "./"
app.config["APPLICATION_ROOT"] = APP_ROOT
json_app = FlaskJSON(app)


@as_json
@app.route("/predict_json", methods=["POST"])
def predict_json():
    data = request.get_json()
    if (data.get("type") != "text") or ("content" not in data):
        output = invalid_request_error(None)
        return output

    content = data["content"]

    try:
        if isinstance(content, list):
            labels, predictions = processor.predict(content)
        else:
            labels, predictions = processor.predict([content])

        output = generate_successful_response(labels[0], np.max(predictions[0]))
        return output
    except Exception as e:
        return generate_failure_response(
            status=404,
            code="elg.service.internalError",
            text=None,
            params=None,
            detail=e,
        )


@json_app.invalid_json_error
def invalid_request_error(e):
    """Generates a valid ELG "failure" response if the request cannot be parsed"""
    raise JsonError(
        status_=400,
        failure={
            "errors": [
                {"code": "elg.request.invalid", "text": "Invalid request message"}
            ]
        },
    )


def generate_successful_response(label, confidence):
    """Generates the dict with the text classification reponse

    :param label: the label of the input after classification
    :param confidence: confidence of the model
    :return: a dict with the response

    """
    response = {
        "type": "classification",
        "classes": [{"class": label, "score": confidence}],
    }
    output = {"response": response}
    return output


def generate_failure_response(status, code, text, params, detail):
    """Generate a wrong response indicating the failure

    :param status: api error code
    :param code: ELG error type
    :param text: not used
    :param params: not used
    :param detail: detail of the exception

    """

    error = {}
    if code:
        error["code"] = code
    if text:
        error["text"] = text
    if params:
        error["params"] = params
    if detail:
        error["detail"] = str(detail)

    raise JsonError(status_=status, failure={"errors": [error]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8866)
