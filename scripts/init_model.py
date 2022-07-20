from single_labeling_api import TWilBertLabelClass


class Initializer:
    def __init__(self):
        processor = TWilBertLabelClass(
            "configs/microservs/config_labelling_single_hateeval19_large.json"
        )
        processor.predict(["Texto de prueba"])
